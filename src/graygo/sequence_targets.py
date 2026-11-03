"""Truncated Retrace targets with bounded inference and start-only losses.

Only actual retained trajectory records enter the trace. The last continuing
transition always bootstraps, even at a replay or 32-round boundary.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from .models import epsilon_greedy_distribution
from .nfsp import OneStepBatch, _bounded, apply_q_update

TRACE_VERSION = 'graygo/nfsp-retrace/1'
TRACE_FIELDS = ('trace_mass', 'sequence_records', 'start_overlap', 'opponent_snapshot_generation_age', 'replay_age', 'continuation_overlap_sum')


@dataclass
class SequenceBatch:
    steps: OneStepBatch
    starts: OneStepBatch
    valid: torch.Tensor
    behavior: torch.Tensor

    @classmethod
    def from_replay(cls, data, device):
        valid = np.asarray(data['valid'])
        if (valid.dtype != np.bool_ or valid.ndim != 2 or not len(valid)
                or not 1 <= valid.shape[1] <= 32 or not valid[:, 0].all()
                or ((~valid[:, :-1]) & valid[:, 1:]).any()):
            raise ValueError('nonempty contiguous sequence prefixes of at most 32 rounds required')
        n, length = valid.shape
        for name in ('action', 'reward', 'terminated', 'behavior_prob', 'value_mask',
                     'stream', 'step', 'opponent_version', 'policy_version', 'mode', 'supplemental', 'ordinal'):
            if np.asarray(data[name]).shape != valid.shape:
                raise ValueError('sequence scalar metadata dimensions disagree')
        if np.asarray(data['value_mask']).dtype != np.bool_ or not data['value_mask'][valid].all():
            raise ValueError('only Q-eligible records may enter a sequence')
        behavior = np.asarray(data['behavior_prob'])
        if (not np.isfinite(behavior[valid]).all() or (behavior[valid] <= 0).any()
                or (behavior[valid] > 1).any()):
            raise ValueError('actual positive conditional behavior probabilities required')
        for name in ('action', 'stream', 'step', 'opponent_version', 'policy_version', 'ordinal', 'mode'):
            value = np.asarray(data[name])
            if not np.issubdtype(value.dtype, np.integer) or (value[valid] < 0).any():
                raise ValueError('nonnegative integral sequence identifiers required')
        if (np.asarray(data['terminated']).dtype != np.bool_
                or np.asarray(data['supplemental']).dtype != np.bool_
                or (data['mode'][valid] > 1).any()):
            raise ValueError('boolean sequence flags and valid modes required')
        linked = valid[:, 1:]
        if (data['terminated'][:, :-1][linked]).any():
            raise ValueError('sequence continues past a true terminal')
        for name in ('stream', 'opponent_version', 'policy_version', 'mode', 'supplemental'):
            if (data[name][:, 1:][linked] != data[name][:, :-1][linked]).any():
                raise ValueError('sequence crosses a player/episode/frozen-policy boundary')
        if ((data['step'][:, 1:][linked] != data['step'][:, :-1][linked] + 1).any()
                or (data['ordinal'][:, 1:][linked] <= data['ordinal'][:, :-1][linked]).any()):
            raise ValueError('sequence chronology differs')
        for name in ('spatial', 'globals', 'legal'):
            if np.asarray(data[name]).shape[:2] != valid.shape or np.asarray(data['next_' + name]).shape != np.asarray(data[name]).shape:
                raise ValueError('sequence observation dimensions disagree')
            if not np.array_equal(data['next_' + name][:, :-1][linked], data[name][:, 1:][linked]):
                raise ValueError('sequence adjacent public observations disagree')
        flat = {name: np.asarray(value)[valid] for name, value in data.items()}
        starts = {name: np.asarray(value)[:, 0] for name, value in data.items()}
        steps = OneStepBatch.from_replay(flat, device)
        first = OneStepBatch.from_replay(starts, device)
        steps.validate(); first.validate()
        return cls(steps, first, torch.as_tensor(valid, device=device),
                   torch.as_tensor(behavior[valid], device=device, dtype=torch.float32))


@torch.no_grad()
def _forward(model, inputs, legal, limit):
    chunks = []
    for start in range(0, len(legal), limit):
        stop = start + limit
        output = model(**{name: value[start:stop] for name, value in inputs.items()})
        _bounded(output, legal[start:stop])
        chunks.append(output)
    return torch.cat(chunks)


@torch.no_grad()
def retrace_targets(online, target, batch, epsilon, *, trace_lambda=.95, target_batch=512):
    if (not math.isfinite(trace_lambda) or not 0 <= trace_lambda <= 1
            or type(target_batch) is not int or target_batch < 1):
        raise ValueError('finite trace lambda and positive bounded forward batch required')
    valid = batch.valid
    n, length = valid.shape
    steps = batch.steps
    device = steps.actions.device
    row, col = valid.nonzero(as_tuple=True)
    eps = torch.as_tensor(epsilon, device=device, dtype=steps.rewards.dtype)
    if eps.ndim and eps.shape != (n,):
        raise ValueError('epsilon must be scalar or one per sequence start')
    if not bool(torch.isfinite(eps).all()) or bool(((eps < 0) | (eps > 1)).any()):
        raise ValueError('invalid epsilon')
    per_step_epsilon = eps if eps.ndim == 0 else eps[row]
    current = _forward(online, steps.inputs, steps.legal, target_batch)
    frozen = _forward(target, steps.inputs, steps.legal, target_batch)
    policy = epsilon_greedy_distribution(current, steps.legal, per_step_epsilon)
    chosen = frozen.gather(1, steps.actions[:, None]).squeeze(1)
    overlap = (policy.gather(1, steps.actions[:, None]).squeeze(1) / batch.behavior).clamp(max=1)
    # Adjacent retained sources are exactly the preceding successors. Reuse
    # these forwards; infer additional successors only at continuing boundaries.
    following = torch.zeros_like(chosen)
    has_next = torch.zeros_like(valid)
    has_next[:, :-1] = valid[:, 1:]
    interior = has_next[valid]
    indices = torch.arange(len(chosen), device=device)
    if bool(interior.any()):
        after = indices[interior] + 1
        following[interior] = (policy[after] * frozen[after].masked_fill(~steps.legal[after], 0)).sum(-1)
    boundary = (~interior) & (~steps.terminated)
    if bool(boundary.any()):
        inputs = {name: value[boundary] for name, value in steps.next_inputs.items()}
        legal = steps.next_legal[boundary]
        online_next = _forward(online, inputs, legal, target_batch)
        target_next = _forward(target, inputs, legal, target_batch)
        boundary_epsilon = per_step_epsilon if eps.ndim == 0 else per_step_epsilon[boundary]
        pi = epsilon_greedy_distribution(online_next, legal, boundary_epsilon)
        following[boundary] = (pi * target_next.masked_fill(~legal, 0)).sum(-1)
    delta = steps.rewards + following - chosen
    residuals = torch.zeros((n, length), device=device, dtype=delta.dtype)
    coefficients = torch.zeros_like(residuals)
    residuals[valid] = delta
    coefficients[valid] = trace_lambda * overlap
    # The first action is already conditioned on: c_t never weights delta_t.
    lengths = valid.sum(1)
    start_indices = torch.cat((torch.zeros(1, device=device, dtype=torch.long), lengths.cumsum(0)[:-1]))
    raw = chosen[start_indices].clone()
    weight = torch.ones(n, device=device, dtype=raw.dtype)
    mass = torch.ones_like(weight)
    for index in range(length):
        if index:
            weight = weight * coefficients[:, index]
            mass += weight
        raw += weight * residuals[:, index]
    if not bool(torch.isfinite(raw).all()):
        raise ValueError('nonfinite raw Retrace target')
    projected = raw.clamp(-1, 1)
    stats = {'raw_target_min': float(raw.min()), 'raw_target_max': float(raw.max()),
             'projected_targets': int((raw != projected).sum()), 'targets': n,
             'q_target': TRACE_VERSION, 'sequence_records': int(valid.sum()),
             'context_records': int(valid.sum()) - n,
             'sequence_length_max': int(lengths.max()), 'trace_mass_sum': float(mass.sum()),
             'trace_mass_min': float(mass.min()), 'trace_mass_max': float(mass.max()),
             'nonterminal_boundaries': int(boundary.sum())}
    continuation_overlap = torch.zeros_like(residuals)
    continuation_overlap[valid] = overlap
    continuation_overlap[:, 0] = 0
    details = {'trace_mass': mass.cpu().double().numpy(),
               'sequence_records': lengths.cpu().numpy(),
               'start_overlap': overlap[start_indices].cpu().double().numpy(),
               'continuation_overlap_sum': continuation_overlap.sum(1).cpu().double().numpy()}
    return projected, stats, details


def update_retrace_q(online, target, optimizer, data, *, device, epsilon, trace_lambda=.95,
                     target_batch=512, max_grad_norm=.5, diagnostics=None, trace_observer=None):
    batch = SequenceBatch.from_replay(data, device)
    labels, stats, details = retrace_targets(online, target, batch, epsilon,
        trace_lambda=trace_lambda, target_batch=target_batch)
    if trace_observer is not None:
        trace_observer(details)
    return apply_q_update(online, optimizer, batch.starts, labels, stats,
        max_grad_norm=max_grad_norm, diagnostics=diagnostics)


class TraceDiagnostics:
    """Cumulative start-weighted diagnostics; context never increases reuse."""
    def __init__(self):
        self.counts = np.zeros((2, 2, 3, 3), dtype=np.int64)
        self.sums = np.zeros((*self.counts.shape, len(TRACE_FIELDS)), dtype=np.float64)

    def observe(self, starts, details, fresh_records, policy_version):
        age = fresh_records - 1 - starts['ordinal'].astype(np.int64)
        # Actor IDs include the acting-policy bit; generation ages do not.
        opponent = starts['opponent_version'] // np.uint64(2)
        if (age < 0).any() or (opponent > policy_version).any():
            raise ValueError('trace diagnostics contain future records or opponent versions')
        overlap = np.asarray(details['start_overlap'])
        indices = (starts['mode'].astype(np.int64), starts['supplemental'].astype(np.int64),
                   np.where(age <= 12800, 0, np.where(age <= 128000, 1, 2)),
                   np.where(overlap <= .1, 0, np.where(overlap < .9, 1, 2)))
        np.add.at(self.counts, indices, 1)
        fields = (details['trace_mass'], details['sequence_records'], overlap,
                  policy_version - opponent.astype(np.int64), age, details['continuation_overlap_sum'])
        for index, values in enumerate(fields):
            np.add.at(self.sums[..., index], indices, values)

    def state_dict(self):
        return {'version': TRACE_VERSION, 'counts': torch.from_numpy(self.counts.copy()),
                'sums': torch.from_numpy(self.sums.copy())}

    def load_state_dict(self, state, expected_starts):
        counts, sums = state['counts'], state['sums']
        if (state['version'] != TRACE_VERSION or counts.dtype != torch.int64
                or sums.dtype != torch.float64 or tuple(counts.shape) != self.counts.shape
                or tuple(sums.shape) != self.sums.shape or bool((counts < 0).any())
                or not bool(torch.isfinite(sums).all()) or bool((sums < 0).any())
                or int(counts.sum()) != expected_starts
                or bool((sums[..., 0] < counts.double() - 1e-6).any())
                or bool((sums[..., 0] > sums[..., 1] + 1e-6).any())
                or bool((sums[..., 1] > 32 * counts.double() + 1e-6).any())
                or bool((sums[..., 2] > counts.double() + 1e-6).any())
                or bool((sums[..., 5] > sums[..., 1] - counts.double() + 1e-6).any())
                or bool((sums[counts == 0] != 0).any())):
            raise ValueError('trace diagnostic schema/counts differ')
        self.counts[:] = counts.cpu().numpy()
        self.sums[:] = sums.cpu().numpy()

    def report(self):
        return {'version': TRACE_VERSION, 'axes': ['mode', 'supplemental', 'replay_age', 'start_target_behavior_overlap'],
                'age_upper_bounds': [12800, 128000, None], 'overlap_bins': ['at_most_0.1', 'between_0.1_and_0.9', 'at_least_0.9'],
                'sum_fields': list(TRACE_FIELDS), 'counts': self.counts.tolist(), 'sums': self.sums.tolist(),
                'loss_starts': int(self.counts.sum()), 'context_is_additional_loss_targets': False,
                'opponent_age_definition': 'current Q+SL update generation minus stored opponent actor ID // 2'}

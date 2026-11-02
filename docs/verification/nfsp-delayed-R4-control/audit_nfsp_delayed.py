"""Read-only independent audit of a completed opaque delayed NFSP trial.

Use the trial's pinned src directory on PYTHONPATH. This never instantiates a
trainer, truncates evidence, or invokes the trainer's task evaluator. Values and
strategy gaps are derived directly from the chain's terminal outcomes.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from graygo.buffers import ReplayBuffer, ReservoirBuffer
from graygo.manifest import source_digest, write_report
from graygo.storage import CheckpointStore, digest
from graygo.task_adapter import TaskAdapter
from graygo.task_training import NFSPTaskConfig, HEADER_DTYPE
from graygo.tasks import BalancedCoverage


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message, atol=1e-12):
    try:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)
    except AssertionError as exc:
        raise ValueError(message) from exc


def q_updates(records, config):
    return max(0, records // config.batch - config.learning_start // config.batch + 1) * config.q_reuse


def partial_header(active):
    episode = active['episode']; choices = [episode['actors'][str(p)]['choice'] for p in (1, 2)]
    row = np.zeros(1, dtype=HEADER_DTYPE)
    saved = episode['state']; branch = saved['branch']; controller = saved['controller']
    row['episode'] = episode['episode_id']; row['start_record'] = active['start_record']; row['steps'] = episode['steps']
    row['root'] = active['root_index']; row['generation'] = active['snapshot']['version']
    row['supplemental'] = episode['supplemental']; row['average_ready'] = active['snapshot']['average_ready']
    forced = episode['forced_root']; row['forced_role'] = 0 if forced is None else forced[0]
    row['forced_action'] = 65535 if forced is None else forced[1]
    for key, field in (('best_response_mode', 'best_response'), ('mode_draw', 'mode_draw'), ('ramp_draw', 'ramp_draw'),
                       ('ramp_probability', 'ramp_probability'), ('epsilon', 'epsilon')):
        row[field] = [[c[key] for c in choices]]
    row['acting'] = [[{'q': 0, 'average': 1, 'uniform': 2}[c['acting_policy']] for c in choices]]
    row['reward_black'] = (1 if branch == 0 else -1) * (1 if controller == 1 else -1)
    return row


def predict(adapter, weights, config, epsilon):
    q, average = adapter.models()
    q.load_state_dict(weights['q']); average.load_state_dict(weights['average'])
    q.eval(); average.eval(); values = []; policies = []; targets = []; greedies = []
    with torch.no_grad():
        for start in range(0, len(adapter.observations), config.eval_batch):
            obs = adapter.observations[start:start + config.eval_batch]
            x = torch.tensor(np.stack([o.spatial for o in obs]), dtype=torch.float32)
            g = torch.tensor(np.stack([o.global_features for o in obs]), dtype=torch.float32)
            legal = torch.tensor(np.stack([o.legal for o in obs]), dtype=torch.bool)
            scores = q(x, g); logits = average(x, g)
            require(bool(torch.isfinite(scores[legal]).all()) and bool((scores[legal].abs() <= 1 + 1e-6).all()), 'archive has invalid Q values')
            maxima = scores.masked_fill(~legal, -torch.inf).max(-1, keepdim=True).values
            tied = (legal & (scores == maxima)).float()
            greedy = tied / tied.sum(-1, keepdim=True)
            uniform = legal.to(scores.dtype) / legal.sum(-1, keepdim=True)
            eps = torch.as_tensor(epsilon, dtype=scores.dtype)
            target = (1 - eps) * greedy + eps * uniform
            policy = torch.log_softmax(logits.masked_fill(~legal, -torch.inf), -1).exp()
            require(bool(torch.isfinite(policy).all()), 'archive has invalid deployed policy')
            values.append(scores.double().numpy()); policies.append(policy.double().numpy())
            targets.append(target.double().numpy()); greedies.append(greedy.double().numpy())
    shape = (len(adapter.states), 2, adapter.actions)
    def normalized(parts):
        table = np.concatenate(parts).reshape(shape)
        return table / table.sum(-1, keepdims=True)
    return np.concatenate(values).reshape(shape), normalized(policies), normalized(targets), normalized(greedies)


def analytical_roots(adapter, q, average, target, greedy, records, ready):
    c = adapter.config; ramp = min(1., max(0., (records - c.reservoir_open) / (c.handoff_end - c.reservoir_open)))
    mass_average = .9 * ramp if ready else 0.
    weights = np.array([.1 if c.uniform_comparison else 1 - mass_average, mass_average,
                        .9 - mass_average if c.uniform_comparison else 0.])
    results = []
    for root_index, root in enumerate(adapter.roots):
        i = adapter.index[root]; controller = root.controller; role_index = controller - 1
        p0 = target[i, role_index, 0]
        other_p0 = weights[0] * p0 + weights[1] * average[i, role_index, 0] + weights[2] * .5
        early = []; late = []; actions = []
        def append(state, role, expected, rows):
            index = adapter.index[state]; legal = tuple(adapter.game.actions(state, role))
            predicted = float(target[index, role - 1, list(legal)] @ q[index, role - 1, list(legal)])
            rows.append({'state_index': index, 'player': role, 'expected': float(expected),
                         'predicted': predicted, 'error': abs(float(expected) - predicted)})
        for role in (1, 2):
            truths = (1., -1.) if role == controller else (-(2 * other_p0 - 1),)
            for action, truth in enumerate(truths):
                prediction = float(q[i, role - 1, action])
                actions.append({'player': role, 'action': action, 'expected': float(truth), 'predicted': prediction,
                                'error': abs(float(truth) - prediction)})
            append(root, role, 2 * p0 - 1 if role == controller else -(2 * other_p0 - 1), early)
        for branch in (0, 1):
            for depth in sorted({d for d in (1, 2, 5) if d < root.horizon}):
                state = type(root)(root.horizon, controller, depth, branch)
                for role in (1, 2):append(state, role, (1 if branch == 0 else -1) * (1 if role == controller else -1), early)
            state = type(root)(root.horizon, controller, root.horizon - 1, branch)
            for role in (1, 2):append(state, role, (1 if branch == 0 else -1) * (1 if role == controller else -1), late)
        def gaps(policy):
            gap = float(2 * (1 - policy[i, role_index, 0]))
            return [gap, 0.] if controller == 1 else [0., gap]
        results.append({'root_index': root_index, 'root_tag': adapter.root_tags[root_index], 'value': 1. if controller == 1 else -1.,
            'native_opponent_weights': weights.tolist(), 'average_gaps': gaps(average), 'greedy_q_gaps': gaps(greedy),
            'root_action_values': actions, 'early_values': early, 'late_values': late,
            'max_early_error': max(row['error'] for row in early + actions), 'max_late_error': max(row['error'] for row in late),
            'root_average_policies': average[i].tolist(), 'root_greedy_policies': greedy[i].tolist()})
    return results


def compare_tree(actual, expected, path='root'):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), path + ' fields differ')
        for key in expected:compare_tree(actual[key], expected[key], path + '/' + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), path + ' length differs')
        for i, value in enumerate(expected):compare_tree(actual[i], value, path + '/' + str(i))
    elif isinstance(expected, float):close(actual, expected, path + ' numerical result differs')
    else:require(actual == expected, path + ' identity differs')


def audit(root, *, require_formal=True, budget_profile=None):
    root = Path(root); manifest = json.loads((root / 'run.json').read_text()); config = NFSPTaskConfig(**manifest['config'])
    require(config.task in ('delayed', 'delayed-forced'), 'auditor requires an opaque delayed task')
    require(manifest['source_sha256'] == source_digest(), 'use the trial\'s pinned source on PYTHONPATH')
    if require_formal:require(config.formal_protocol(), 'trial does not use the formal primary protocol')
    if config.budget_profile_sha256:
        profile = Path(budget_profile) if budget_profile else root.parent / (config.learner + '-budget.json')
        require(digest(profile) == config.budget_profile_sha256, 'funding profile checksum differs')
        funded = json.loads(profile.read_text())
        require(funded['source_sha256'] == source_digest() and funded['learner'] == config.learner and funded['task'] == config.task
                and funded['q_reuse'] == config.q_reuse and funded['complete_pipeline'] and set(funded['phases']) == {'pre', 'post'}, 'funding profile is not this complete learner')
    torch.set_num_threads(1)
    adapter = TaskAdapter(config); require(manifest['adapter'] == json.loads(json.dumps(adapter.manifest())), 'opaque state/codebook/oracle inventory differs')
    store = CheckpointStore(root); bundle = store.latest(); seal = store.verify(bundle)
    state = torch.load(bundle / 'trainer.pt', weights_only=True, map_location='cpu'); counts = state['backend']['counters']; n = config.records
    require(seal['step'] == counts['fresh_q'] == n and state['config'] == config.plan(), 'trial is incomplete or configuration differs')
    require(seal['metadata']['source_sha256'] == source_digest(), 'sealed source differs')
    audit_state = state['episode_audit']; path = root / 'episodes.bin'
    require(path.stat().st_size == audit_state['bytes'] == state['episodes'] * HEADER_DTYPE.itemsize, 'completed episode inventory differs')
    require(digest(path) == audit_state['sha256'], 'episode header checksum differs')
    schema = json.loads((root / 'episodes.schema.json').read_text())
    require(schema['itemsize'] == HEADER_DTYPE.itemsize and schema['dtype'] == json.loads(json.dumps(HEADER_DTYPE.descr)), 'episode header schema differs')
    completed = np.memmap(path, dtype=HEADER_DTYPE, mode='r', shape=(state['episodes'],)) if state['episodes'] else np.empty(0, dtype=HEADER_DTYPE)
    headers = np.concatenate((completed, partial_header(state['active']))) if state['active'] else completed
    replay = ReplayBuffer.load(bundle / 'replay.npz'); reservoir = ReservoirBuffer.load(bundle / 'reservoir.npz')
    require(replay.spec == reservoir.spec == adapter.spec and replay.capacity == reservoir.capacity == config.capacity, 'buffer schema/capacity differs')
    require(replay.insertions == n and replay.size == min(n, config.capacity) and reservoir.opening_record == config.reservoir_open, 'buffer insertion/opening clocks differ')
    expected_records = np.full(config.capacity, -1, dtype=np.int64); expected_versions = np.zeros(config.capacity, dtype=np.uint64)
    rng = {name: np.random.default_rng(config.seed + offset) for name, offset in
           (('mode_black', 101), ('mode_white', 102), ('reservoir', 105), ('q_replay', 106), ('supervised', 107))}
    coverage = BalancedCoverage(config.horizons, seed=config.seed + 108)
    cursor = ordinary = seen = changes = supervised_targets = 0; supervised_events = []; last_accepted = -1
    recent_changes = deque(maxlen=config.batch)
    mode_visits = np.zeros(2, dtype=np.int64); acting_counts = np.zeros(3, dtype=np.int64); root_episodes = np.zeros((len(adapter.roots), 2), dtype=np.uint64)
    forced_counts = np.zeros(2, dtype=np.int64); ordinary_ends = []; ordinary_starts = []; ordinary_before = []
    for episode, row in enumerate(headers):
        start = int(row['start_record']); steps = int(row['steps']); root_state = adapter.roots[int(row['root'])]
        require(int(row['episode']) == episode and start == cursor and 0 < steps <= root_state.horizon, 'episode/record/depth continuity differs')
        if episode < len(completed):require(steps == root_state.horizon and row['termination'] == 1, 'completed delayed episode is truncated')
        entry = coverage.next(); supplemental = entry.forced_root is not None
        require(entry.horizon == root_state.horizon and entry.controller == root_state.controller and supplemental == row['supplemental'], 'balanced randomized coverage differs')
        require(int(row['forced_role']) == (entry.controller if supplemental else 0) and int(row['forced_action']) == (entry.forced_root if supplemental else 65535), 'forced-root origin differs')
        require(row['reward_black'] in (-1., 1.), 'invalid chain outcome')
        if supplemental:
            forced_counts[root_state.controller - 1] += 1
            require(row['reward_black'] == (1 if entry.forced_root == 0 else -1) * (1 if entry.controller == 1 else -1), 'forced root does not produce its terminal outcome')
        ready = bool(supervised_events)
        require(bool(row['average_ready']) == ready, 'average readiness differs from first publication')
        require(int(row['generation']) == q_updates(start, config) + len(supervised_events), 'frozen snapshot generation differs')
        ramp = min(1., max(0., (start - config.reservoir_open) / (config.handoff_end - config.reservoir_open)))
        eps = .01 + .09 * max(0., 1 - start / config.epsilon_decay)
        for role, name in enumerate(('mode_black', 'mode_white')):
            draws = rng[name].random(2)
            close([row['mode_draw'][role], row['ramp_draw'][role]], draws, 'private episode mode/ramp RNG differs', atol=0)
            best = draws[0] < .1
            acting = 1 if not best and ready and draws[1] < ramp else (2 if config.uniform_comparison and not best else 0)
            require(bool(row['best_response'][role]) == best and row['acting'][role] == acting, 'logical/actual episode policy differs')
            close(row['epsilon'][role], eps, 'episode epsilon clock differs', atol=0)
            close(row['ramp_probability'][role], ramp, 'episode hand-off probability differs', atol=0)
            mode_visits[role] += steps * best; acting_counts[acting] += steps
        root_episodes[int(row['root']), int(supplemental)] += 1
        end = start + 2 * steps
        labels = np.empty(0, dtype=np.int64)
        if not supplemental:
            ordinals = start + np.arange(2 * steps)
            labels = ordinals[(ordinals >= config.reservoir_open) & row['best_response'][ordinals % 2].astype(bool)]
            ordinary_starts.append(start); ordinary_ends.append(end); ordinary_before.append(ordinary)
            for boundary_ordinary in range((ordinary // config.batch + 1) * config.batch, ordinary + 2 * steps + 1, config.batch):
                boundary = start + boundary_ordinary - ordinary
                boundary_seen = seen + int(np.searchsorted(labels, boundary))
                if boundary_seen >= config.batch:
                    supervised_events.append((boundary, boundary_seen))
                    supervised_targets += min(boundary_seen, config.capacity) if config.learner == 'tabular' else config.batch
                    if config.learner == 'neural':rng['supervised'].integers(0, min(boundary_seen, config.capacity), config.batch)
            ordinary += 2 * steps
        for record in labels:
            slot = seen if seen < config.capacity else int(rng['reservoir'].integers(0, seen + 1)); seen += 1
            if slot < config.capacity:
                recent_changes.append((int(record), slot, int(expected_records[slot])))
                expected_records[slot] = record; expected_versions[slot] = int(row['generation']) * 2
                changes += 1; last_accepted = int(record)
        cursor = end
    require(cursor == n and ordinary == counts['fresh_ordinary'], 'fresh/ordinary budget accounting differs')
    require(state['completed_rounds'] == sum(int(row['steps']) for row in completed) and state['per_role'] == [n // 2] * 2, 'completed/per-role round counts differ')
    require(coverage.state_dict() == state['coverage'], 'coverage continuation state differs')
    require(mode_visits.tolist() == state['best_response_visits'] and acting_counts.tolist() == [state['acting_counts'][k] for k in ('q', 'average', 'uniform')], 'record-weighted episode mode metrics differ')
    require(counts['q_updates'] == q_updates(n, config) and counts['q_loss_targets'] == counts['q_updates'] * config.batch and counts['target_refreshes'] == n // config.target_every, 'Q/reuse/refresh accounting differs')
    require(counts['supervised_updates'] == len(supervised_events) and counts['supervised_targets'] == supervised_targets, 'ordinary publication/target accounting differs')
    for boundary in range(config.learning_start, n + 1, config.batch):
        for _ in range(config.q_reuse):rng['q_replay'].integers(0, min(boundary, config.capacity), config.batch)
    for name, generator in rng.items():require(generator.bit_generator.state == state['rng'][name], name + ' continuation RNG differs')
    for role, name in enumerate(('action_black', 'action_white')):
        generator = np.random.default_rng(config.seed + 103 + role)
        generator.bit_generator.advance(n // 2 - int(forced_counts[role]))
        require(generator.bit_generator.state == state['rng'][name], name + ' private draw count differs')
    for name, updates in (('q_optimizer', counts['q_updates']), ('average_optimizer', counts['supervised_updates'])):
        optimizer = state['backend'][name]
        if optimizer is not None:require(all(int(value['step'].item()) == updates for value in optimizer['state'].values()), 'Adam update count differs')
    require(reservoir.seen == seen and reservoir.revisions == changes, 'eligible labels/Algorithm R replacements differ')
    np.testing.assert_array_equal(reservoir.arrays['record'][:reservoir.size], expected_records[:reservoir.size])
    np.testing.assert_array_equal(reservoir.arrays['policy_version'][:reservoir.size], expected_versions[:reservoir.size])
    with np.load(bundle / 'metrics.npz', allow_pickle=False) as metrics:
        require(int(metrics['visits'].sum()) == n and int(metrics['visits'][..., 0].sum()) == ordinary, 'visit/origin totals differ')
        require(int(metrics['replay_starts'].sum()) == int(metrics['overlap_counts'].sum()) == counts['q_loss_targets'], 'sample/overlap target totals differ')
        np.testing.assert_array_equal(metrics['root_episodes'], root_episodes)
    # Decode retained records in bounded chunks, independent of the task evaluator.
    ids = adapter.game.encoding.ids; token_keys = {token: (key, role) for (key, role), token in ids.items()}
    starts = headers['start_record'].astype(np.int64)
    token_fields = np.zeros((adapter.spec.spatial_shape[0], 5), dtype=np.int64)
    for token, (key, role) in token_keys.items():token_fields[token] = (*key, role)
    root_horizons = np.array([root.horizon for root in adapter.roots])
    root_controllers = np.array([root.controller for root in adapter.roots])
    def check_observations(arrays, indices, records, *, next_state=False):
        prefix = 'next_' if next_state else ''
        x, g, legal = adapter.spec.unpack(arrays[prefix + 'spatial'][indices], arrays[prefix + 'globals'][indices], arrays[prefix + 'legal'][indices])
        positions = np.searchsorted(starts, records, side='right') - 1
        rows = headers[positions]; depths = (records - rows['start_record'].astype(np.int64)) // 2 + int(next_state)
        horizons = root_horizons[rows['root']]; controllers = root_controllers[rows['root']]; roles = records % 2 + 1
        branches = np.where(rows['reward_black'] == np.where(controllers == 1, 1, -1), 0, 1)
        terminal = depths == horizons
        np.testing.assert_array_equal(x.sum(-1), (~terminal).astype(np.uint64))
        expected = np.stack((horizons, controllers, depths, np.where(depths == 0, -1, branches), roles), axis=-1)
        np.testing.assert_array_equal(token_fields[x[~terminal].argmax(-1)], expected[~terminal])
        action_counts = np.where(terminal, 0, np.where(depths == 0, np.where(roles == controllers, 2, 1), adapter.game.equivalent_actions))
        np.testing.assert_array_equal(legal, np.arange(adapter.actions)[None, :] < action_counts[:, None])
        close(g[:, 0], ((horizons - depths) / horizons).astype(np.float32), 'retained public countdown differs', atol=0)
        return rows, depths, legal
    retained = replay.size; ordinals = replay.arrays['ordinal'][:retained].astype(np.int64)
    np.testing.assert_array_equal(np.sort(ordinals), np.arange(n - retained, n))
    for start in range(0, retained, 2048):
        selection = slice(start, min(start + 2048, retained)); records = ordinals[selection]; a = replay.arrays
        rows, depths, legal = check_observations(a, selection, records)
        check_observations(a, selection, records, next_state=True)
        role = records % 2; local = np.arange(len(records)); episode_index = rows['episode'].astype(np.int64)
        np.testing.assert_array_equal(a['stream'][selection], 2 * episode_index + role)
        np.testing.assert_array_equal(a['step'][selection], depths)
        np.testing.assert_array_equal(a['mode'][selection], rows['best_response'][local, role])
        np.testing.assert_array_equal(a['policy_version'][selection], 2 * rows['generation'] + (rows['acting'][local, role] == 1))
        np.testing.assert_array_equal(a['opponent_version'][selection], 2 * rows['generation'] + (rows['acting'][local, 1 - role] == 1))
        np.testing.assert_array_equal(a['supplemental'][selection], rows['supplemental'])
        np.testing.assert_array_equal(a['actor_mask'][selection], ~rows['supplemental'])
        require(a['value_mask'][selection].all() and legal[local, a['action'][selection]].all(), 'retained action/value eligibility differs')
        terminal = np.array([int(depth) + 1 == adapter.roots[int(row['root'])].horizon for row, depth in zip(rows, depths)])
        np.testing.assert_array_equal(a['terminated'][selection], terminal)
        expected_reward = np.where(terminal, rows['reward_black'] * np.where(role == 0, 1, -1), 0).astype(np.float32)
        np.testing.assert_array_equal(a['reward'][selection], expected_reward)
        probabilities = a['behavior_prob'][selection]
        require(np.isfinite(probabilities).all() and (probabilities > 0).all() and (probabilities <= 1).all(), 'retained behavior probabilities invalid')
        controllers = root_controllers[rows['root']]
        root_actions = (depths == 0) & (role + 1 == controllers)
        branches = np.where(rows['reward_black'] == np.where(controllers == 1, 1, -1), 0, 1)
        np.testing.assert_array_equal(a['action'][selection][root_actions], branches[root_actions])
        require((probabilities[root_actions & rows['supplemental']] == 1).all(), 'forced-root marginal probability differs')
    mean_check = config.learner == 'tabular' and bool(supervised_events)
    published_records = expected_records.copy()
    pending_changes = []
    if mean_check:
        last_publication = supervised_events[-1][0]
        pending_changes = [change for change in recent_changes if change[0] >= last_publication]
        require(len(pending_changes) < config.batch, 'publication tail exceeds ordinary cadence')
        for record, slot, old_record in reversed(pending_changes):published_records[slot] = old_record
    mean_totals = np.zeros((adapter.spec.spatial_shape[0], adapter.actions), dtype=np.float64) if mean_check else None
    mean_counts = np.zeros(adapter.spec.spatial_shape[0], dtype=np.int64) if mean_check else None
    for start in range(0, reservoir.size, 2048):
        selection = slice(start, min(start + 2048, reservoir.size)); a = reservoir.arrays
        records = a['record'][selection].astype(np.int64); rows, depths, legal = check_observations(a, selection, records)
        greedy = np.unpackbits(a['greedy'][selection], axis=-1, count=adapter.actions, bitorder='little').astype(bool)
        require(greedy.any(-1).all() and not (greedy & ~legal).any(), 'stored frozen greedy sets are empty/illegal')
        if mean_check:
            x = np.unpackbits(a['spatial'][selection], axis=-1, count=adapter.spec.spatial_shape[0], bitorder='little')
            tokens = x.argmax(-1)
            unchanged = published_records[selection] == records
            np.add.at(mean_totals, tokens[unchanged], greedy[unchanged] / greedy[unchanged].sum(-1, keepdims=True))
    exact_mean_states = root_mean_states = 0; ambiguous_mean_states = 0
    if mean_check:
        # Reverse only the sub-batch unpublished Algorithm-R tail. Unknown old
        # targets stay bounded by their legal simplex; unchanged targets are exact.
        published = published_records[published_records >= 0]
        positions = np.searchsorted(starts, published, side='right') - 1
        rows = headers[positions]; depths = (published - rows['start_record'].astype(np.int64)) // 2
        tokens = np.empty(len(published), dtype=np.int64)
        for k, (record, row, depth) in enumerate(zip(published, rows, depths)):
            root_state = adapter.roots[int(row['root'])]
            branch = 0 if row['reward_black'] == (1 if root_state.controller == 1 else -1) else 1
            key = (root_state.horizon, root_state.controller, int(depth), -1 if depth == 0 else branch)
            tokens[k] = ids[key, int(record % 2) + 1]
        np.add.at(mean_counts, tokens, 1)
        unknown_counts = np.zeros_like(mean_counts)
        changed_slots = np.flatnonzero((published_records >= 0) & (published_records != expected_records))
        for slot in changed_slots:
            record = int(published_records[slot]); row = headers[np.searchsorted(starts, record, side='right') - 1]
            root_state = adapter.roots[int(row['root'])]; depth = (record - int(row['start_record'])) // 2
            branch = 0 if row['reward_black'] == (1 if root_state.controller == 1 else -1) else 1
            unknown_counts[ids[(root_state.horizon, root_state.controller, depth, -1 if depth == 0 else branch), record % 2 + 1]] += 1
        table = state['backend']['average']['probabilities'].numpy()
        known = unknown_counts == 0; present = mean_counts > 0
        expected = np.full_like(mean_totals, 1 / adapter.actions)
        expected[present] = mean_totals[present] / mean_counts[present, None]
        close(table[known], expected[known], 'published exact tabular reservoir means differ', atol=1e-10)
        residual = table * mean_counts[:, None] - mean_totals
        require((residual >= -1e-7).all() and (residual <= unknown_counts[:, None] + 1e-7).all(), 'published means exceed unpublished-tail uncertainty')
        close(residual.sum(-1), unknown_counts, 'published means have inconsistent unresolved label mass', atol=1e-7)
        for token in np.flatnonzero(unknown_counts):
            key, role = token_keys[int(token)]; horizon, controller, depth, branch = key
            legal_count = (2 if role == controller else 1) if depth == 0 else adapter.game.equivalent_actions
            require(np.abs(residual[token, legal_count:]).max(initial=0) <= 1e-7, 'unresolved historical target has illegal mass')
        exact_mean_states = int(known.sum()); ambiguous_mean_states = int((~known).sum())
        root_tokens = [ids[(r.horizon, r.controller, 0, -1), role] for r in adapter.roots for role in (1, 2)]
        root_mean_states = int(known[root_tokens].sum())
    # Recompute every scheduled evaluation from the permanent model weights.
    expected_steps = list(range(config.eval_every, n + 1, config.eval_every))
    if not expected_steps or expected_steps[-1] != n:expected_steps.append(n)
    require([e['records'] for e in state['evaluations']] == expected_steps, 'scheduled archive/evaluation inventory differs')
    event_records = np.array([event[0] for event in supervised_events], dtype=np.int64)
    oracle_evaluations = []
    for evaluation in state['evaluations']:
        step = evaluation['records']; archive = root / 'weights' / f'{step:012d}'; sealed = store.verify(archive)
        require(sealed['step'] == step and sealed['metadata']['source_sha256'] == source_digest(), 'archive source/clock differs')
        weights = torch.load(archive / 'models.pt', weights_only=True, map_location='cpu')
        require(weights['config'] == config.plan(), 'archive configuration differs')
        ready = bool(np.searchsorted(event_records, step, side='right'))
        require(evaluation['average_ready'] == ready, 'evaluation average availability differs')
        epsilon = .01 + .09 * max(0., 1 - step / config.epsilon_decay)
        close(evaluation['epsilon'], epsilon, 'evaluation epsilon differs', atol=0)
        q, avg, target, greedy = predict(adapter, weights, config, epsilon)
        oracle = analytical_roots(adapter, q, avg, target, greedy, step, ready)
        compare_tree(evaluation['roots'], oracle, f'evaluation/{step}')
        oracle_evaluations.append({'records': step, 'average_ready': ready, 'max_average_gap': max(max(r['average_gaps']) for r in oracle),
            'max_greedy_gap': max(max(r['greedy_q_gaps']) for r in oracle), 'max_early_error': max(r['max_early_error'] for r in oracle)})
    final = oracle_evaluations[-5:]; enough = len(final) == 5; ready = enough and all(e['average_ready'] for e in final)
    average_gap = max(e['max_average_gap'] for e in final); greedy_gap = max(e['max_greedy_gap'] for e in final); error = max(e['max_early_error'] for e in final)
    credit = enough and greedy_gap <= .05 and error <= .1
    band = 'eligible_for_limited_pilot' if ready and average_gap <= .1 and credit else ('averaging_lag' if ready and average_gap <= .25 and credit else 'not_ready')
    passed = ready and band == 'eligible_for_limited_pilot' and config.formal_protocol()
    result = json.loads((root / 'result.json').read_text())
    require(result['status'] == 'complete' and result['records'] == n and result['formal_case_passed'] == passed and result['q_credit_passed'] == credit and result['average_readiness'] == band, 'reported completed-run decisions differ')
    close([result['final_five_max_average_gap'], result['final_five_max_greedy_gap'], result['final_five_max_early_error']], [average_gap, greedy_gap, error], 'reported final-five maxima differ')
    return {'format_version': 1, 'verified': True, 'task': config.task, 'learner': config.learner, 'seed': config.seed, 'records': n,
        'source_sha256': source_digest(), 'auditor_sha256': digest(Path(__file__)), 'training_manifest_sha256': digest(root / 'run.json'), 'checkpoint_manifest_sha256': digest(bundle / 'manifest.json'),
        'episode_audit_sha256': digest(path), 'physical_complete_games': len(completed), 'active_episode_steps': 0 if not state['active'] else state['active']['episode']['steps'],
        'eligible_labels': seen, 'accepted_reservoir_changes': changes, 'ordinary_records': ordinary, 'q_updates': counts['q_updates'],
        'supervised_updates': len(supervised_events), 'target_refreshes': counts['target_refreshes'], 'retained_executed_records_checked': retained,
        'retained_label_records_checked': reservoir.size, 'published_tabular_mean_checked': mean_check and ambiguous_mean_states == 0,
        'published_tabular_exact_mean_states': exact_mean_states, 'published_tabular_ambiguous_mean_states': ambiguous_mean_states,
        'published_tabular_exact_root_mean_states': root_mean_states, 'scheduled_evaluations': oracle_evaluations,
        'formal_protocol': config.formal_protocol(), 'q_credit_passed': credit, 'average_readiness': band, 'formal_case_passed': passed,
        'last_supervised_fresh_record': None if not supervised_events else supervised_events[-1][0], 'last_accepted_label_record': last_accepted,
        'scope': 'independent closed-form archived policy/value grades, exact episode/mode/coverage/Algorithm-R/RNG clocks and retained transitions; completed full action trajectories and every historical greedy target are not reconstructed from header-only logs'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir'); parser.add_argument('--output', required=True); parser.add_argument('--budget-profile')
    parser.add_argument('--allow-diagnostic', action='store_true')
    args = parser.parse_args()
    result = audit(args.run_dir, require_formal=not args.allow_diagnostic, budget_profile=args.budget_profile)
    write_report(args.output, result)
    print(json.dumps({k: result[k] for k in ('verified', 'learner', 'records', 'q_credit_passed', 'average_readiness', 'formal_case_passed')}, sort_keys=True))

"""Read-only Q best-response diagnostics for the actual mixed branching task.

The opponent component is hidden and chosen once per episode. Actions are
optimized after marginalizing that component, never with an oracle mode tag.
Legacy Nash-gap/value/average grades and original files remain unchanged.
"""
import argparse
from contextlib import ExitStack
import fcntl
from importlib import metadata
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from graygo.engine import BLACK, WHITE
from graygo.manifest import source_digest, write_report
from graygo.storage import CheckpointStore
from graygo.task_adapter import TaskAdapter
from graygo.task_training import NFSPTaskConfig
from graygo.task_evaluation import predict, evaluate_task, native_opponent_weights
from tools.audit_nfsp_delayed import compare_tree
from tools.nfsp_resources import require
from tools.task_actor_boundary import file_sha

VERSION = 'graygo/mixed-branch-native-response-diagnostic/1'


def load(path):return json.loads(Path(path).read_text())


def probabilities(adapter, table, state, role):
    legal = tuple(adapter.game.actions(state, role))
    values = np.asarray(table[adapter.index[state], role - 1, list(legal)], dtype=np.float64)
    require(values.shape == (len(legal),) and np.isfinite(values).all() and (values >= 0).all()
            and abs(float(values.sum()) - 1) <= 1e-10, 'complete legal policy distribution required')
    return legal, values


def native_value(adapter, root, role, components, weights, own_policy=None):
    """Finite belief-state recursion; None selects an observable best response."""
    require(adapter.config.task == 'branching', 'this diagnostic is scoped to the actual two-round mixed branching game')
    game = adapter.game; other = 3 - role; nodes = 0
    belief = np.asarray(weights, dtype=np.float64)
    require(belief.shape == (len(components),) and np.isfinite(belief).all() and (belief >= 0).all()
            and abs(float(belief.sum()) - 1) <= 1e-12, 'fixed episode-mixture weights required')
    def visit(state, posterior):
        nonlocal nodes
        nodes += 1; require(nodes <= adapter.config.max_nodes, 'finite native-response node budget exhausted')
        if game.terminal(state):return float(game.utility(state)) * (1 if role == BLACK else -1)
        own = tuple(game.actions(state, role)); opposing = tuple(game.actions(state, other))
        table = np.stack([probabilities(adapter, c, state, other)[1] for c in components])
        joint = posterior[:, None] * table; marginal = joint.sum(0); action_values = []
        for action in own:
            value = 0.
            for index, move in enumerate(opposing):
                if marginal[index] == 0:continue
                black, white = (action, move) if role == BLACK else (move, action)
                value += float(marginal[index]) * visit(game.transition(state, black, white), joint[:, index] / marginal[index])
            action_values.append(value)
        if own_policy is None:return max(action_values)
        return float(probabilities(adapter, own_policy, state, role)[1] @ np.asarray(action_values))
    return float(visit(root, belief))


def enumerated_response(adapter, root, role, components, weights, own_policy):
    """Independent complete pure-strategy enumeration, mixing before maximizing."""
    game = adapter.game; other = 3 - role; states = set()
    def collect(state):
        if game.terminal(state) or state in states:return
        states.add(state)
        for black, white in itertools.product(game.actions(state, BLACK), game.actions(state, WHITE)):
            collect(game.transition(state, black, white))
    collect(root)
    decisions = sorted((state for state in states if len(game.actions(state, role)) > 1), key=lambda s: str(s))
    def component_value(state, opponent, choices):
        if game.terminal(state):return float(game.utility(state)) * (1 if role == BLACK else -1)
        own, policy = probabilities(adapter, own_policy, state, role)
        if choices is not None:
            selected = choices.get(state, own[0]); policy = np.array([float(a == selected) for a in own])
        opposing, opponent_policy = probabilities(adapter, opponent, state, other)
        total = 0.
        for a, p in zip(own, policy):
            for b, q in zip(opposing, opponent_policy):
                black, white = (a, b) if role == BLACK else (b, a)
                total += float(p * q) * component_value(game.transition(state, black, white), opponent, choices)
        return total
    candidates = []
    for actions in itertools.product(*(game.actions(state, role) for state in decisions)):
        choices = dict(zip(decisions, actions))
        candidates.append(sum(float(w) * component_value(root, c, choices) for w, c in zip(weights, components)))
    actual = sum(float(w) * component_value(root, c, None) for w, c in zip(weights, components))
    return float(max(candidates)), float(actual), len(candidates)


def diagnose(adapter, components, weights, greedy):
    roots = []
    for tag, root in zip(adapter.root_tags, adapter.roots):
        rows = []
        for role in (BLACK, WHITE):
            best = native_value(adapter, root, role, components, weights)
            actual = native_value(adapter, root, role, components, weights, greedy)
            independent_best, independent_actual, count = enumerated_response(adapter, root, role, components, weights, greedy)
            require(abs(best - independent_best) <= 1e-12 and abs(actual - independent_actual) <= 1e-12,
                    'belief recursion differs from independent complete-strategy enumeration')
            gap = best - actual; require(-1e-12 <= gap <= 2 + 1e-12, 'invalid native best-response gap')
            rows.append({'player': role, 'native_best_response_value': best, 'greedy_value_against_same_native_mixture': actual,
                         'greedy_native_best_response_gap': max(0., gap), 'pure_complete_strategies_checked': count})
        roots.append({'root_tag': tag, 'native_episode_mixture_weights': list(map(float, weights)), 'roles': rows})
    return roots


def report(repo, specification, expected_sha, output):
    repo = Path(repo).resolve(); specification = Path(specification).resolve(); output = Path(output).resolve()
    require(file_sha(specification) == expected_sha and not output.exists(), 'frozen diagnostic spec and new output required')
    spec = load(specification)
    require(spec['version'] == VERSION and spec['seeds'] == [0, 1, 2]
            and spec['task'] == 'branching' and spec['cohort_selection'] is False, 'complete original three-seed branching diagnostic required')
    require(file_sha(Path(__file__)) == spec['producer_sha256'] and source_digest() == spec['learner_source_sha256'], 'exact diagnostic/source package required')
    torch.set_num_threads(1); cases = {}; inputs = {str(specification): file_sha(specification)}
    with ExitStack() as locks:
        for seed in spec['seeds']:
            binding = spec['cases'][str(seed)]; root = repo / binding['root']
            require(not output.is_relative_to(root), 'diagnostic output cannot modify an original learner')
            fcntl.flock(locks.enter_context((root / '.trainer.lock').open('rb')), fcntl.LOCK_SH | fcntl.LOCK_NB)
            require(file_sha(root / 'run.json') == binding['manifest_sha256'] and file_sha(root / 'result.json') == binding['result_sha256'], 'original completed branch case changed')
            manifest = load(root / 'run.json'); result = load(root / 'result.json'); config = NFSPTaskConfig(**manifest['config'])
            require(config.formal_protocol() and config.task == 'branching' and config.learner == 'neural' and config.seed == seed
                    and config.device == 'cpu' and manifest['source_sha256'] == spec['learner_source_sha256']
                    and result['status'] == 'complete' and result['records'] == 15000000, 'full original CPU branching case required')
            require(all(metadata.version(name) == value for name, value in manifest['dependencies'].items() if value is not None), 'original numerical dependency versions changed')
            adapter = TaskAdapter(config); rows = []
            require([row['records'] for row in result['evaluations']] == list(range(500000, 15000001, 500000)), 'all thirty original reads required')
            for recorded in result['evaluations']:
                step = recorded['records']; archive = root / 'weights' / f'{step:012d}'; seal = CheckpointStore.verify(archive)
                require(seal['step'] == step and seal['metadata']['source_sha256'] == spec['learner_source_sha256'], 'actual original public archive differs')
                payload = torch.load(archive / 'models.pt', weights_only=True, map_location='cpu')
                require(payload['config'] == config.plan(), 'archive original config differs')
                with torch.random.fork_rng(devices=[]), torch.device('cpu'):
                    q, average = adapter.models()
                q.load_state_dict(payload['q']); average.load_state_dict(payload['average']); q.eval(); average.eval()
                ready = recorded['counters']['supervised_updates'] > 0
                backend = SimpleNamespace(q=q, average=average, device=torch.device('cpu'), counters=SimpleNamespace(fresh_records=step), average_ready=ready)
                epsilon = config.schedule().epsilon(step)
                legacy = evaluate_task(adapter, backend, epsilon, batch_size=config.eval_batch)
                compare_tree(recorded['roots'], legacy, path=f'seed{seed}/record{step}/original-root-metrics')
                _, average_table, target, greedy = predict(adapter, backend, epsilon, config.eval_batch)
                masks = np.stack([o.legal for o in adapter.observations]).reshape(target.shape)
                uniform = masks.astype(float) / masks.sum(-1, keepdims=True)
                native = diagnose(adapter, (target, average_table, uniform), native_opponent_weights(adapter, backend), greedy)
                maximum = max(role['greedy_native_best_response_gap'] for r in native for role in r['roles'])
                rows.append({'records': step, 'greedy_native_best_response_gap': maximum,
                    'max_average_equilibrium_gap_unchanged': max(max(r['average_gaps']) for r in legacy),
                    'max_native_value_error_unchanged': max(r['max_early_error'] for r in legacy),
                    'legacy_greedy_equilibrium_gap_unchanged': max(max(r['greedy_q_gaps']) for r in legacy), 'native_response_roots': native})
                for path in (archive / 'manifest.json', archive / 'models.pt'):inputs[str(path)] = file_sha(path)
            cases[str(seed)] = {'original_root': str(root), 'original_manifest_sha256': binding['manifest_sha256'],
                'original_result_sha256': binding['result_sha256'], 'evaluations': rows,
                'final_five_maxima': {key: max(row[key] for row in rows[-5:]) for key in
                    ('greedy_native_best_response_gap', 'max_average_equilibrium_gap_unchanged', 'max_native_value_error_unchanged', 'legacy_greedy_equilibrium_gap_unchanged')}}
            inputs[str(root / 'run.json')] = file_sha(root / 'run.json'); inputs[str(root / 'result.json')] = file_sha(root / 'result.json')
        require(all(file_sha(path) == value for path, value in inputs.items()), 'original diagnostic inputs changed during read')
    result = {'version': VERSION, 'status': 'all_original_branch_archives_native_response_diagnosed',
        'source_sha256': source_digest(), 'producer_sha256': file_sha(Path(__file__)), 'cases': cases, 'input_files_sha256': inputs,
        'all_thirty_legacy_root_metrics_reproduced_per_seed': True, 'independent_enumeration_matches_belief_recursion': True,
        'original_data_grades_or_thresholds_modified': False, 'corrected_gate_protocol_selected': False,
        'new_training_records': 0, 'new_scored_evaluation_games': 0, 'a2_complete': False, 'execution_authorized': False,
        'scope': 'Exploratory read-only native-opponent best-response diagnostic for every original archive; latent opponent type is marginalized before choosing own actions. Source-correct legacy Nash/value/average metrics are reproduced unchanged. No case pass, evaluator replacement, protocol/corrective selection, resampling or pilot follows.'}
    write_report(output, result); print(json.dumps({'status': result['status'], 'final_five': {k: v['final_five_maxima'] for k, v in cases.items()}}))


def main():
    parser = argparse.ArgumentParser()
    for name in ('repo', 'specification', 'expected-specification-sha', 'output'):parser.add_argument('--' + name, required=True)
    args = parser.parse_args(); report(args.repo, args.specification, args.expected_specification_sha, args.output)


if __name__ == '__main__':main()

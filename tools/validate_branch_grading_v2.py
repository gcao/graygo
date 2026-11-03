"""Compare the v2 production evaluator with every original branching archive.

Independent frozen v1 evidence keeps its original source and grades. This
checks a new evaluator only; it is not a source-admission or A2 exit bridge.
"""
import argparse
from contextlib import ExitStack
import fcntl
from importlib import metadata
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from graygo.manifest import source_digest, write_report
from graygo.storage import CheckpointStore
from graygo.task_adapter import TaskAdapter
from graygo.task_evaluation import evaluate_task, EVALUATION_VERSION
from graygo.task_training import NFSPTaskConfig
from tools.audit_nfsp_delayed import compare_tree
from tools.nfsp_resources import require
from tools.task_actor_boundary import file_sha

VERSION = 'graygo/branch-grading-v2-production-validation/1'


def load(path):
    return json.loads(Path(path).read_text())


def report(repo, specification, expected_sha, output):
    repo = Path(repo).resolve()
    specification = Path(specification).resolve()
    output = Path(output).resolve()
    require(not output.exists() and file_sha(specification) == expected_sha,
            'new output and exact frozen validation specification required')
    spec = load(specification)
    require(spec['version'] == VERSION and spec['evaluation_protocol_version'] == EVALUATION_VERSION,
            'explicit version-two production evaluator required')
    require(source_digest() == spec['new_learner_source_sha256']
            and file_sha(Path(__file__)) == spec['validator_sha256'],
            'exact new production source and validator required')
    original_spec_path = repo / spec['original_diagnostic_specification']
    original_spec = load(original_spec_path)
    diagnostic_path = repo / spec['independent_diagnostic']
    diagnostic = load(diagnostic_path)
    require(file_sha(original_spec_path) == spec['original_diagnostic_specification_sha256']
            and file_sha(diagnostic_path) == spec['independent_diagnostic_sha256'],
            'independent original evidence differs')
    require(original_spec['seeds'] == [0, 1, 2]
            and original_spec['archive_records'] == list(range(500000, 15000001, 500000))
            and diagnostic['independent_enumeration_matches_belief_recursion']
            and diagnostic['all_thirty_legacy_root_metrics_reproduced_per_seed']
            and not diagnostic['original_data_grades_or_thresholds_modified'],
            'complete independent legacy-metric and response-enumeration evidence required')
    require(diagnostic['source_sha256'] == original_spec['learner_source_sha256']
            and diagnostic['producer_sha256'] == original_spec['producer_sha256'],
            'independent diagnostic original source differs')
    torch.set_num_threads(1)
    inputs = {str(specification): file_sha(specification),
              str(original_spec_path): file_sha(original_spec_path),
              str(diagnostic_path): file_sha(diagnostic_path)}
    cases = {}
    with ExitStack() as locks:
        for seed in (0, 1, 2):
            binding = original_spec['cases'][str(seed)]
            root = repo / binding['root']
            require(not output.is_relative_to(root), 'output cannot modify an original case')
            fcntl.flock(locks.enter_context((root / '.trainer.lock').open('rb')), fcntl.LOCK_SH | fcntl.LOCK_NB)
            manifest = load(root / 'run.json')
            result = load(root / 'result.json')
            require(file_sha(root / 'run.json') == binding['manifest_sha256']
                    and file_sha(root / 'result.json') == binding['result_sha256']
                    and manifest['config'] == binding['original_config']
                    and manifest['source_sha256'] == binding['original_source_sha256'],
                    'original source/config/result identity differs')
            require(all(metadata.version(name) == version for name, version in manifest['dependencies'].items()
                        if version is not None), 'original numerical dependencies differ')
            config = NFSPTaskConfig(**manifest['config'])
            require(config.formal_protocol() and config.task == 'branching' and config.seed == seed
                    and result['status'] == 'complete' and result['records'] == 15000000,
                    'full original branching run required')
            require([r['records'] for r in result['evaluations']] == original_spec['archive_records'],
                    'all thirty original archives required')
            adapter = TaskAdapter(config)
            reference = diagnostic['cases'][str(seed)]
            require(reference['original_manifest_sha256'] == binding['manifest_sha256']
                    and reference['original_result_sha256'] == binding['result_sha256']
                    and Path(reference['original_root']) == root,
                    'independent reference is for another original case')
            require([r['records'] for r in reference['evaluations']] == original_spec['archive_records'],
                    'all thirty independent reference reads required')
            rows = []
            for recorded, independent in zip(result['evaluations'], reference['evaluations']):
                step = recorded['records']
                require(independent['records'] == step, 'independent evaluation clock differs')
                archive = root / 'weights' / f'{step:012d}'
                seal = CheckpointStore.verify(archive)
                require(seal['step'] == step and seal['metadata']['source_sha256'] == binding['original_source_sha256'],
                        'public archive original source/step differs')
                payload = torch.load(archive / 'models.pt', weights_only=True, map_location='cpu')
                require(payload['config'] == manifest['config'], 'original archive configuration differs')
                with torch.random.fork_rng(devices=[]), torch.device('cpu'):
                    q, average = adapter.models()
                q.load_state_dict(payload['q']); average.load_state_dict(payload['average'])
                q.eval(); average.eval()
                backend = SimpleNamespace(q=q, average=average, device=torch.device('cpu'),
                    counters=SimpleNamespace(fresh_records=step),
                    average_ready=recorded['counters']['supervised_updates'] > 0)
                current = evaluate_task(adapter, backend, config.schedule().epsilon(step), batch_size=config.eval_batch)
                legacy = [{k: v for k, v in r.items() if k != 'native_response'} for r in current]
                compare_tree(recorded['roots'], legacy, path=f'seed{seed}/{step}/unchanged-legacy-metrics')
                require(len(current) == len(independent['native_response_roots']) == 2,
                        'both role-swapped roots required')
                for actual, expected in zip(current, independent['native_response_roots']):
                    require(actual['root_tag'] == expected['root_tag'], 'independent root differs')
                    compare_tree(actual['native_opponent_weights'], expected['native_episode_mixture_weights'])
                    require([r['player'] for r in actual['native_response']] == [1, 2],
                            'both responding roles required')
                    for a, e in zip(actual['native_response'], expected['roles']):
                        compare_tree([a['player'], a['best_response_value'], a['greedy_value'], a['gap']],
                                     [e['player'], e['native_best_response_value'],
                                      e['greedy_value_against_same_native_mixture'], e['greedy_native_best_response_gap']])
                rows.append({'records': step,
                    'max_native_response_gap': max(r['gap'] for root_row in current for r in root_row['native_response']),
                    'max_average_gap': max(max(r['average_gaps']) for r in current),
                    'max_early_error': max(r['max_early_error'] for r in current),
                    'max_legacy_greedy_equilibrium_gap': max(max(r['greedy_q_gaps']) for r in current)})
                for path in (archive / 'manifest.json', archive / 'models.pt'):
                    inputs[str(path)] = file_sha(path)
            maxima = {key: max(row[key] for row in rows[-5:]) for key in rows[0] if key != 'records'}
            cases[str(seed)] = {'original_root': str(root), 'evaluations': rows, 'final_five_maxima': maxima,
                'prospective_v2_numeric_passed': all(maxima[key] <= limit for key, limit in
                    [('max_native_response_gap', .05), ('max_average_gap', .05), ('max_early_error', .10)]),
                'original_legacy_pass_unchanged': result['formal_case_passed']}
            inputs[str(root / 'run.json')] = binding['manifest_sha256']
            inputs[str(root / 'result.json')] = binding['result_sha256']
        require(all(file_sha(path) == digest for path, digest in inputs.items()), 'input changed during read')
    output_report = {'version': VERSION, 'status': 'all_ninety_original_archives_match_v2_production_evaluator',
        'evaluation_protocol_version': EVALUATION_VERSION, 'new_learner_source_sha256': source_digest(),
        'cases': cases, 'input_files_sha256': inputs, 'all_original_legacy_metrics_reproduced': True,
        'all_native_response_values_match_independent_enumeration': True,
        'original_grades_or_files_modified': False, 'source_admission_qualified': False,
        'new_protocol_training_selected': False, 'a2_complete': False, 'execution_authorized': False,
        'new_training_records': 0, 'new_scored_evaluation_games': 0,
        'scope': 'Read-only compatibility of the explicit v2 production evaluator with frozen independent evidence. No old plan conversion, new training grant, source admission, control waiver or pilot.'}
    write_report(output, output_report)
    return output_report


def main():
    parser = argparse.ArgumentParser()
    for name in ('repo', 'specification', 'expected-specification-sha', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = report(args.repo, args.specification, args.expected_specification_sha, args.output)
    print(json.dumps({'status': result['status'],
                      'prospective_v2_passes': {s: c['prospective_v2_numeric_passed'] for s, c in result['cases'].items()}}))


if __name__ == '__main__':
    main()

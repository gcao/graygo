"""Compare actual source-pinned diagnostic task pipelines without resuming them."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tools.task_actor_boundary import fingerprint, file_sha, require

VERSION = 'graygo/task-pipeline-parity/1'
MANIFEST_ENV = ('created_at', 'run_id', 'pid', 'source_sha256', 'git_commit', 'disk_free_bytes', 'host_memory')


def typed(value):
    """Retain optimizer integer keys rather than coerce them to parameter names."""
    if isinstance(value, dict):
        require(all(type(key) in (str, int) for key in value), 'unexpected state dictionary key type')
        return {('str:' + key if isinstance(key, str) else 'int:' + str(key)): typed(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(typed(item) for item in value)
    return value


def state_sha(value):
    return fingerprint(typed(value))


def omit_root(value, names):
    require(isinstance(value, dict) and all(name in value for name in names), 'expected explicit timing/provenance fields missing')
    return {key: item for key, item in value.items() if key not in names}


def npz_sha(path):
    with np.load(path, allow_pickle=False) as data:
        return state_sha({name: data[name] for name in sorted(data.files)})


def snapshot(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        require(not path.is_symlink(), 'pipeline evidence cannot contain symlinks')
        if path.is_file():
            result[str(path.relative_to(root))] = file_sha(path)
    return result


def compare_case(reference, candidate, *, reference_source, candidate_source, config):
    from graygo.storage import CheckpointStore
    old_root, new_root = Path(reference).resolve(), Path(candidate).resolve()
    before = {str(root): snapshot(root) for root in (old_root, new_root)}
    require(config['purpose'] == 'profile' and config['task'] == 'delayed' and config['records'] == 65536
            and config['eval_every'] == 32768 and config['q_reuse'] == 4 and config['device'] == 'cpu', 'fixed diagnostic comparison recipe required')
    values = []
    for root, source in ((old_root, reference_source), (new_root, candidate_source)):
        manifest = json.loads((root / 'run.json').read_text())
        require(manifest['source_sha256'] == source and manifest['git_dirty'] is False and manifest['config'] == config,
                'pipeline source/recipe/clean manifest differs')
        result = json.loads((root / 'result.json').read_text())
        require(result['source_sha256'] == source and result['status'] == 'complete' and result['records'] == 65536
                and result['formal_protocol'] is False and result['a2_complete'] is False, 'incomplete or rebranded pipeline diagnostic')
        require([row['records'] for row in result['evaluations']] == [32768, 65536], 'diagnostic scheduled evaluations differ')
        counters = result['counters']
        require(counters['q_updates'] > 0 and counters['target_refreshes'] >= 2, 'missing actual optimizer/target work')
        if config['reservoir_open'] == 0:
            require(counters['supervised_updates'] > 0 and result['acting_policy_counts']['average'] > 0, 'post-opening pipeline did not use learned averaging')
        else:
            require(config['reservoir_open'] == 5000000 and counters['supervised_updates'] == 0
                    and result['acting_policy_counts']['average'] == 0, 'pre-opening pipeline changed')
        weights = sorted(path for path in (root / 'weights').iterdir() if path.is_dir())
        resumes = sorted(path for path in (root / 'resume').iterdir() if path.is_dir())
        require([int(path.name) for path in weights] == [32768, 65536] and [int(path.name) for path in resumes] == [65536], 'public archive or final heavy-state inventory differs')
        public = {}
        for path in weights:
            seal = CheckpointStore.verify(path)
            require(seal['metadata']['source_sha256'] == source, 'public archive source differs')
            payload = torch.load(path / 'models.pt', map_location='cpu', weights_only=True)
            require(payload['config'] == config, 'public archive configuration differs')
            public[path.name] = state_sha(payload)
        resume = resumes[0]
        seal = CheckpointStore.verify(resume)
        require(seal['metadata']['source_sha256'] == source, 'resume source differs')
        trainer = torch.load(resume / 'trainer.pt', map_location='cpu', weights_only=True)
        require(trainer['config'] == config, 'resume configuration differs')
        require(trainer['backend']['counters'] == counters, 'resume/final learner clocks differ')
        values.append({'manifest': state_sha(omit_root(manifest, MANIFEST_ENV)),
                       'result': state_sha(omit_root(result, ('elapsed_seconds', 'timings', 'source_sha256'))),
                       'public_weights': public,
                       'trainer_state': state_sha(omit_root(trainer, ('elapsed', 'timings'))),
                       'replay': npz_sha(resume / 'replay.npz'), 'reservoir': npz_sha(resume / 'reservoir.npz'),
                       'metrics': npz_sha(resume / 'metrics.npz'), 'episode_bytes': file_sha(root / 'episodes.bin'),
                       'episode_schema': state_sha(json.loads((root / 'episodes.schema.json').read_text()))})
    for key in values[0]:
        require(values[0][key] == values[1][key], 'exact pipeline comparison differs: ' + key)
    require(before == {str(root): snapshot(root) for root in (old_root, new_root)}, 'pipeline evidence changed during comparison')
    return {'status': 'finite_complete_pipeline_parity_verified', 'config': config, 'equal_partitions': values[0],
            'input_snapshots': before, 'records_per_source': 65536,
            'reference_source_sha256': reference_source, 'candidate_source_sha256': candidate_source,
            'timing_and_environment_exclusions': {'trainer_root': ['elapsed', 'timings'],
                'result_root': ['elapsed_seconds', 'timings', 'source_sha256'], 'manifest_root': list(MANIFEST_ENV)},
            'all_other_nested_state_compared': True, 'original_learning_runs_modified': False,
            'full_15M_source_equivalence_verified': False, 'a2_evidence_reusable': False, 'execution_authorized': False}


def compare_profiles(reference, candidate, allocation):
    result = {}
    for case in allocation['cases']:
        config = case['config']
        name = f"{config['learner']}-R4-{case['phase']}"
        require(name not in result, 'duplicate pipeline case')
        result[name] = compare_case(Path(reference) / name, Path(candidate) / name,
                                    reference_source=allocation['reference_source_sha256'], candidate_source=allocation['candidate_source_sha256'], config=config)
    require(set(result) == {f'{learner}-R4-{phase}' for learner in ('neural','tabular') for phase in ('pre','post')}, 'complete four-case comparison required')
    return {'version': VERSION, 'status': 'four_finite_complete_pipelines_verified', 'comparisons': result,
            'aggregate_new_diagnostic_records': 8 * 65536, 'formal_A2_learning_records': 0,
            'full_15M_source_equivalence_verified': False, 'source_compatibility_qualified': False,
            'a2_evidence_reusable': False, 'execution_authorized': False,
            'scope': 'Actual matched pre/post-opening production trainer prefixes, complete model/optimizer/active-actor/private-RNG/replay/reservoir/metric states and scheduled archives. Only named root timing/environment/source fields differ. This is finite source behavior verification, not completed A2 learning or a full-history replay.'}


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('reference', 'candidate', 'allocation', 'expected-allocation-sha', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    try:
        output = Path(args.output).resolve()
        require(not output.exists() and all(not output.is_relative_to(Path(root).resolve()) for root in (args.reference, args.candidate)), 'new output outside both compared pipelines required')
        require(file_sha(args.allocation) == args.expected_allocation_sha, 'pinned pipeline allocation differs')
        result = compare_profiles(args.reference, args.candidate, json.loads(Path(args.allocation).read_text()))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
        print(json.dumps({key: result[key] for key in ('status', 'aggregate_new_diagnostic_records', 'a2_evidence_reusable')}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())

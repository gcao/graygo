"""Qualify only full CPU neural R1 delayed comparisons without filling primary R4 slots."""
import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import time
import torch
from graygo.task_training import NFSPTaskConfig
from tools.task_pipeline_parity import (require,snapshot,state_sha,npz_sha,omit_root,omit_recorded_timings,MANIFEST_ENV,file_sha)
from tools.a2_source_cohort import read_registry
from tools.nfsp_delayed_cpu_prices import read_qualified as read_prices,check_case as check_price_case

VERSION = "graygo/delayed-r1-comparison-source/1"

def compare_case(reference, candidate, *, reference_source, candidate_source, config):
    from graygo.storage import CheckpointStore
    old_root, new_root = Path(reference).resolve(), Path(candidate).resolve()
    before = {str(root): snapshot(root) for root in (old_root, new_root)}
    require(config['purpose'] == 'profile' and config['task'] == 'delayed' and config['records'] == 65536
            and config['eval_every'] == 32768 and config['q_reuse'] == 1 and config['learner'] == 'neural' and config['device'] == 'cpu', 'fixed diagnostic comparison recipe required')
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
                       'result': state_sha(omit_recorded_timings(result, ('elapsed_seconds', 'timings', 'source_sha256'))),
                       'public_weights': public,
                       'trainer_state': state_sha(omit_recorded_timings(trainer, ('elapsed', 'timings'))),
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
                'result_root': ['elapsed_seconds', 'timings', 'source_sha256'], 'manifest_root': list(MANIFEST_ENV),
                'trainer_evaluation_rows': ['elapsed_seconds'], 'result_evaluation_rows': ['elapsed_seconds']},
            'all_other_nested_state_compared': True, 'original_learning_runs_modified': False,
            'full_15M_source_equivalence_verified': False, 'a2_evidence_reusable': False, 'execution_authorized': False}


def load(path):
    return json.loads(Path(path).read_text())


def qualify(repo, specification):
    started = time.monotonic(); repo = Path(repo).resolve(); specification = Path(specification).resolve()
    spec = load(specification)
    require(spec['cohort_role'] == 'one-step-reuse-comparison' and spec['q_reuse'] == 1
            and spec['learner'] == 'neural' and spec['task'] == 'delayed', 'only delayed neural R1 comparison admission required')
    paths = {key: repo / spec[key] for key in ('cpu_price_receipt', 'source_registry', 'own_source_plan')}
    require(all(file_sha(path) == spec[key + '_sha256'] for key, path in paths.items()), 'declared price or original source registration changed')
    prices = read_prices(paths['cpu_price_receipt'])
    require(prices['target_runtime_source_sha256'] == spec['candidate_source_sha256']
            and prices['reference_source_sha256'] == spec['reference_source_sha256'], 'R1 price source pair differs')
    _, _, plans, edges = read_registry(paths['source_registry'], repo=repo, current=True)
    own = [p for p in plans.values() if p.learner_source_sha256 == spec['candidate_source_sha256']]
    require(len(own) == 1 and own[0].to_dict() == load(paths['own_source_plan']) and own[0].multiround_reuse == 4,
            'preserve the original primary R4 source plan; comparisons cannot replace it')
    require(any(e['left'] == spec['reference_source_sha256'] and e['right'] == spec['candidate_source_sha256']
                and e['proof']['source_compatibility_qualified'] is True for e in edges), 'original full default-source bridge absent')
    require(spec['comparison_core_sha256'] == file_sha(repo / spec['comparison_core']), 'original full pipeline comparison core changed')
    inputs = dict(prices['original_inputs_sha256'])
    inputs.update({str(specification): file_sha(specification), **{str(p): file_sha(p) for p in paths.values()},
                   str(repo / spec['comparison_core']): spec['comparison_core_sha256']})
    comparisons = {}
    with ExitStack() as stack:
        for path in sorted({repo / c[key] / '.trainer.lock' for c in spec['cases'] for key in ('reference_root', 'candidate_root')}):
            fcntl.flock(stack.enter_context(path.open('rb')), fcntl.LOCK_SH | fcntl.LOCK_NB)
        for case in spec['cases']:
            phase = case['phase']; config = case['config']
            require(phase in ('pre', 'post') and phase not in comparisons, 'distinct complete original R1 profile pair required')
            expected = NFSPTaskConfig(learner='neural', q_reuse=1, records=65536, eval_every=32768,
                time_cap_seconds=config['time_cap_seconds'], purpose='profile', reservoir_open=5000000 if phase == 'pre' else 0,
                handoff_end=7000000 if phase == 'pre' else 1000)
            require(config == expected.plan(), 'original full-model delayed R1 diagnostic recipe differs')
            for key in ('reference_root', 'candidate_root'):
                path = repo / case[key]
                require(file_sha(path / 'run.json') == case[key + '_manifest_sha256']
                        and file_sha(path / 'result.json') == case[key + '_result_sha256'], 'original R1 profile manifest/result changed')
            comparisons[phase] = compare_case(repo / case['reference_root'], repo / case['candidate_root'],
                reference_source=spec['reference_source_sha256'], candidate_source=spec['candidate_source_sha256'], config=config)
        require(set(comparisons) == {'pre', 'post'}, 'both complete R1 pipeline comparisons required')
        require(all(file_sha(path) == expected for path, expected in inputs.items()), 'R1 source/price inputs changed during qualification')
    return {'version': VERSION, 'status': 'qualified_cpu_delayed_R1_comparison_source',
            'reference_source_sha256': spec['reference_source_sha256'], 'candidate_source_sha256': spec['candidate_source_sha256'],
            'cpu_prices': prices, 'comparisons': comparisons, 'input_files_sha256': inputs,
            'verifier_sha256': file_sha(__file__), 'source_registry_sha256': file_sha(paths['source_registry']),
            'own_source_plan_sha256': file_sha(paths['own_source_plan']),
            'original_primary_plan_unchanged': True, 'primary_A2_slots_fillable': False,
            'new_training_records': 0, 'new_accelerator_jobs': 0, 'execution_authorized': False, 'a2_complete': False,
            'qualification_seconds': time.monotonic() - started,
            'evidence': {'repo': str(repo), 'specification': str(specification)},
            'scope': 'CPU/features neural delayed R1 comparison only, original seeds 0/1/2, unchanged default '
                     'source semantics plus exact actual complete pre/post state/optimizer/RNG/buffer/trajectory '
                     'comparison and fresh source-bound C1 pricing. No 15M replay equivalence claim, primary '
                     'R4 case replacement, cross-source resume, table/GPU allocation, extension or project entry.'}


def read_qualified(path):
    proof = load(path)
    require(proof['version'] == VERSION and proof['verifier_sha256'] == file_sha(__file__), 'receipt-bound R1 comparison qualifier required')
    e = proof['evidence']; fresh = qualify(e['repo'], e['specification'])
    left, right = dict(proof), dict(fresh)
    left.pop('qualification_seconds'); right.pop('qualification_seconds')
    require(left == right, 'fresh R1 comparison source/price qualification differs')
    return proof


def check_case(manifest, proof):
    require(proof.get('version') == VERSION and proof.get('status') == 'qualified_cpu_delayed_R1_comparison_source'
            and proof.get('primary_A2_slots_fillable') is False and proof.get('execution_authorized') is False,
            'scoped non-primary R1 comparison receipt required')
    require(manifest['source_sha256'] == proof['candidate_source_sha256'], 'R1 comparison must retain its actual candidate source')
    result = check_price_case(manifest, proof['cpu_prices'], 'C1')
    return result | {'cohort_role': 'one-step-reuse-comparison', 'primary_A2_slots_fillable': False,
                     'original_primary_plan_unchanged': True}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--repo', required=True)
    parser.add_argument('--specification', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); require(not Path(args.output).exists(), 'preserve original R1 comparison qualification')
    proof = qualify(args.repo, args.specification); Path(args.output).write_text(json.dumps(proof, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': proof['status'], 'complete_pipelines': len(proof['comparisons']), 'new_training_records': 0}))


if __name__ == '__main__':
    main()

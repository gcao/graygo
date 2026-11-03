"""Exact full-state comparison of actual CPU raw-cache task prefixes.

Only the supported diagnostic recipe differs from the preserved delayed-task
comparison. All original source identities, payloads and complete partitions
remain independently checked; no records or manifests are relabeled.
"""
import argparse
import json
from pathlib import Path
import torch
from tools.task_pipeline_parity import (require,snapshot,state_sha,npz_sha,omit_root,omit_recorded_timings,MANIFEST_ENV,file_sha)

VERSION='graygo/cpu-raw-cache-pipeline/1'


def compare_case(reference, candidate, *, reference_source, candidate_source, config):
    from graygo.storage import CheckpointStore
    old_root, new_root = Path(reference).resolve(), Path(candidate).resolve()
    before = {str(root): snapshot(root) for root in (old_root, new_root)}
    require(config['purpose'] == 'profile' and config['task'] in ('gray-pure','gray-mixed') and config['learner']=='neural' and config['records'] == 32768
            and config['eval_every'] == 16384 and config['q_reuse'] == 4 and config['device'] == 'cpu', 'fixed diagnostic comparison recipe required')
    values = []
    for root, source in ((old_root, reference_source), (new_root, candidate_source)):
        manifest = json.loads((root / 'run.json').read_text())
        require(manifest['source_sha256'] == source and manifest['git_dirty'] is False and manifest['config'] == config,
                'pipeline source/recipe/clean manifest differs')
        result = json.loads((root / 'result.json').read_text())
        require(result['source_sha256'] == source and result['status'] == 'complete' and result['records'] == 32768
                and result['formal_protocol'] is False and result['a2_complete'] is False, 'incomplete or rebranded pipeline diagnostic')
        require([row['records'] for row in result['evaluations']] == [16384, 32768], 'diagnostic scheduled evaluations differ')
        counters = result['counters']
        require(counters['q_updates'] > 0 and counters['target_refreshes'] >= 2, 'missing actual optimizer/target work')
        if config['reservoir_open'] == 0:
            require(counters['supervised_updates'] > 0 and result['acting_policy_counts']['average'] > 0, 'post-opening pipeline did not use learned averaging')
        else:
            require(config['reservoir_open'] == 5000000 and counters['supervised_updates'] == 0
                    and result['acting_policy_counts']['average'] == 0, 'pre-opening pipeline changed')
        weights = sorted(path for path in (root / 'weights').iterdir() if path.is_dir())
        resumes = sorted(path for path in (root / 'resume').iterdir() if path.is_dir())
        require([int(path.name) for path in weights] == [16384, 32768] and [int(path.name) for path in resumes] == [32768], 'public archive or final heavy-state inventory differs')
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
            'input_snapshots': before, 'records_per_source': 32768,
            'reference_source_sha256': reference_source, 'candidate_source_sha256': candidate_source,
            'timing_and_environment_exclusions': {'trainer_root': ['elapsed', 'timings'],
                'result_root': ['elapsed_seconds', 'timings', 'source_sha256'], 'manifest_root': list(MANIFEST_ENV),
                'trainer_evaluation_rows': ['elapsed_seconds'], 'result_evaluation_rows': ['elapsed_seconds']},
            'all_other_nested_state_compared': True, 'original_learning_runs_modified': False,
            'full_15M_source_equivalence_verified': False, 'a2_evidence_reusable': False, 'execution_authorized': False}


def compare_profiles(reference,candidate,allocation):
    result={}
    for case in allocation['cases']:
        config=case['config'];task=config['task'];phase=case['phase'];key=task+'/'+phase
        require(key not in result,'duplicate raw-cache case')
        result[key]=compare_case(Path(reference)/task/'neural'/('neural-R4-'+phase),Path(candidate)/task/('neural-R4-'+phase),reference_source=allocation['reference_source_sha256'],candidate_source=allocation['candidate_source_sha256'],config=config)
    require(set(result)=={t+'/'+p for t in ('gray-pure','gray-mixed') for p in ('pre','post')},'all four full-model pipelines required')
    return {'version':VERSION,'status':'four_actual_raw_cache_pipelines_verified','comparisons':result,'aggregate_records_compared':262144,'new_candidate_diagnostic_records':131072,'new_formal_A2_records':0,'source_compatibility_qualified':False,'execution_authorized':False,'scope':'Full model/target/average/optimizer/private RNG/replay/reservoir/metrics/archive/episode equality over original matched production prefixes, excluding only the preserved named timing/environment fields. No full 15M learning or source admission is claimed.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference',required=True);parser.add_argument('--candidate',required=True);parser.add_argument('--allocation',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();require(not Path(args.output).exists(),'preserve prior comparison output')
    result=compare_profiles(args.reference,args.candidate,json.loads(Path(args.allocation).read_text()));Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':result['status'],'cases':len(result['comparisons'])}))


if __name__=='__main__':main()

"""Compare actual scalar/batched matrix pipelines and finite timings.

Production-sized replay layout and small-capacity wrap stress are distinct.
Neither workload establishes a full-run price, A2 case or deployment policy.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from tools.task_pipeline_parity import state_sha,npz_sha,snapshot,omit_recorded_timings,omit_root,MANIFEST_ENV
from tools.task_actor_boundary import file_sha,require

VERSION='graygo/terminal-batch-profile/1'


def cases():
    from graygo.nfsp_training import NFSPMatrixConfig
    rows=[]
    for layout,records,capacity,phases in [('production-layout',100000,200000,('pre','post')),('wrap-stress',65536,4096,('post',))]:
        for task in ('asymmetric','rps'):
            for learner in ('tabular','neural'):
                for phase in phases:
                    cfg=NFSPMatrixConfig(task=task,learner=learner,records=records,eval_every=records//2,capacity=capacity,
                        time_cap_seconds=65,reservoir_open=5000000 if phase=='pre' else 0,handoff_end=7000000 if phase=='pre' else 1000)
                    rows.append({'label':f'{layout}/{task}/{learner}/{phase}','config':asdict(cfg)})
    return rows


def semantic_run(root):
    from graygo.storage import CheckpointStore
    import torch
    root=Path(root);manifest=json.loads((root/'run.json').read_text());cfg=manifest['config'];n=cfg['records']
    result=json.loads((root/'result.json').read_text());require(result['status']=='complete' and result['records']==n and not result['formal_protocol'],'incomplete or misbranded pipeline timing')
    require(result['source_sha256']==manifest['source_sha256'],'profile source provenance differs')
    require([row['records'] for row in result['evaluations']]==[n//2,n],'actual two-read profile inventory differs')
    bundle=root/'resume'/f'{n:012d}';seal=CheckpointStore.verify(bundle);require(seal['metadata']['source_sha256']==manifest['source_sha256'],'checkpoint source differs');state=torch.load(bundle/'trainer.pt',map_location='cpu',weights_only=True)
    require(state['collection_implementation']==result['collection_implementation']==manifest['collection_implementation'],'execution metadata differs')
    def meaningful(value,root_fields):return omit_root(omit_recorded_timings(value,root_fields),('collection_implementation',))
    weights={}
    for path in sorted((root/'weights').iterdir()):
        if path.is_dir():CheckpointStore.verify(path);weights[path.name]=state_sha(torch.load(path/'models.pt',map_location='cpu',weights_only=True))
    require(set(weights)=={f'{n//2:012d}',f'{n:012d}'},'scheduled public archive inventory differs')
    return {'source_sha256':manifest['source_sha256'],'manifest':state_sha(omit_root(manifest,(*MANIFEST_ENV,'collection_implementation'))),
        'result':state_sha(meaningful(result,('elapsed_seconds','timings'))),'trainer_state':state_sha(meaningful(state,('elapsed','timings'))),
        'public_weights':weights,'replay':npz_sha(bundle/'replay.npz'),'reservoir':npz_sha(bundle/'reservoir.npz'),
        'episode_bytes':file_sha(root/'episodes.bin'),'episode_schema':state_sha(json.loads((root/'episodes.schema.json').read_text()))}


def compare_pair(reference,batched):
    reference,batched=Path(reference),Path(batched);before={str(p):snapshot(p) for p in (reference,batched)}
    require(json.loads((reference/'run.json').read_text())['collection_implementation']=='scalar-reference'
            and json.loads((batched/'run.json').read_text())['collection_implementation']=='terminal-batch','two distinct declared collector implementations required')
    left,right=semantic_run(reference),semantic_run(batched)
    for key in left:require(left[key]==right[key],'exact terminal-batch pipeline differs: '+key)
    require(before=={str(p):snapshot(p) for p in (reference,batched)},'pipeline artifacts changed during comparison')
    return {'equal_partitions':left,'input_files_sha256':before,
        'exclusions':{'manifest_root':[*MANIFEST_ENV,'collection_implementation'],
                      'result_root':['elapsed_seconds','timings','collection_implementation'],
                      'trainer_root':['elapsed','timings','collection_implementation'],
                      'scheduled_evaluation_rows':['elapsed_seconds']}}


def run(root,*,expected_source,time_cap_seconds):
    from graygo.manifest import source_digest,write_report
    from graygo.nfsp_training import NFSPMatrixConfig,NFSPMatrixTrainer
    started=time.monotonic();root=Path(root).resolve();require(not root.exists(),'preserve original timing run; choose a new root')
    require(source_digest()==expected_source,'selected terminal-batch package differs');root.mkdir(parents=True)
    plan={'version':VERSION,'source_sha256':expected_source,'producer_sha256':file_sha(__file__),'cases':cases(),
          'scope':'actual complete diagnostic matrix pipeline pairs only; source/capacity/records/mode bound; no full training price or A2 case'}
    write_report(root/'plan.json',plan);results=[];pairs={}
    for index,case in enumerate(plan['cases']):
        modes=('scalar-reference','terminal-batch') if index%2==0 else ('terminal-batch','scalar-reference')
        paths={};timings={}
        for mode in modes:
            require(time.monotonic()-started<time_cap_seconds,'terminal-batch profile time ceiling exhausted')
            directory=root/case['label']/mode;cfg=NFSPMatrixConfig(**case['config'])
            write_report(root/'progress.json',{'status':'running','current_case':case['label'],'current_mode':mode,'completed_pairs':len(pairs)})
            tick=time.monotonic();trainer=NFSPMatrixTrainer(cfg,directory,collection=mode);setup=time.monotonic()-tick
            result=trainer.run(verbose=False);require(result['status']=='complete' and result['records']==cfg.records,'diagnostic pipeline reached a cap before its endpoint')
            row={'label':case['label'],'mode':mode,'config':case['config'],'setup_seconds':setup,'loop_seconds':result['elapsed_seconds'],
                'seconds_including_setup':setup+result['elapsed_seconds'],'timings':result['timings'],'records':result['records'],
                'counters':result['counters'],'reservoir_size':result['reservoir_size'],'replay_size':trainer.replay.size,
                'model_and_optimizer_work_not_mocked':True};results.append(row);paths[mode]=directory;timings[mode]=row
            write_report(root/'results.json',{'plan':plan,'results':results});del trainer
        compared=compare_pair(paths['scalar-reference'],paths['terminal-batch']);cfg=case['config']
        if case['label'].startswith('wrap-stress'):
            require(timings['scalar-reference']['replay_size']==cfg['capacity'] and timings['scalar-reference']['reservoir_size']==cfg['capacity'],'stress case did not wrap both storage classes')
        compared['records_per_mode']=cfg['records'];compared['capacity']=cfg['capacity']
        compared['collection_speed_ratio']=timings['scalar-reference']['timings']['collection']/timings['terminal-batch']['timings']['collection']
        compared['loop_speed_ratio']=timings['scalar-reference']['loop_seconds']/timings['terminal-batch']['loop_seconds'];pairs[case['label']]=compared
        write_report(root/'comparisons.json',{'version':VERSION,'completed_pairs':len(pairs),'pairs':pairs})
    require(len(pairs)==12 and len(results)==24 and source_digest()==expected_source,'profile work/source inventory differs')
    report={'version':VERSION,'status':'complete_exact_pipeline_pairs','plan':plan,'pairs':pairs,'results':results,
            'source_unchanged':True,'diagnostic_records':sum(row['records'] for row in results),'elapsed_seconds':time.monotonic()-started,
            'formal_A2_records':0,'full_run_price_qualified':False,'execution_authorized':False,
            'scope':'finite actual production-learner comparisons, with partial production replay and separate both-buffer wrap stress; mode/source/capacity bound, no universal speedup or learning readiness claim'}
    write_report(root/'report.json',report)
    write_report(root/'progress.json',{'status':'complete','completed_pairs':len(pairs),'current_case':None,'current_mode':None})
    print(json.dumps({'status':report['status'],'diagnostic_records':report['diagnostic_records'],'elapsed_seconds':report['elapsed_seconds'],
        'ratios':{label:{key:pair[key] for key in ('collection_speed_ratio','loop_speed_ratio')} for label,pair in pairs.items()}},indent=2));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--expected-source-sha',required=True);parser.add_argument('--time-cap-seconds',type=int,default=2400);args=parser.parse_args()
    run(args.run_dir,expected_source=args.expected_source_sha,time_cap_seconds=args.time_cap_seconds)

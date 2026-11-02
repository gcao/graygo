"""Real CLI registration/report integration with zero training/evaluation records."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from graygo.board_runtime import BoardNFSPRuntime,BoardRuntimeConfig
from graygo.board_journal import BoardMilestoneJournal,BoardJournalPlan
from graygo.manifest import source_digest,make_manifest,write_report
from graygo.pilot_gates import DEVELOPMENT_SEEDS,PRE,POST
from graygo.storage import CheckpointStore,digest


def tree(root):
    return {'files':{str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()},
            'directories':sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_dir())}


def check(root):
    root=Path(root).resolve()
    if root.exists():raise FileExistsError('diagnostic exists; inspect its original process and evidence')
    source=source_digest();manifest=make_manifest(seed=0,command={'check_cohort_evidence':'zero-record real CLI registration/report'},rules={'size':9,'max_rounds':200,'collision':'gray'})
    if manifest['git_dirty']:raise ValueError('commit the measured source before collection')
    root.mkdir(parents=True);write_report(root/'run.json',manifest)
    write_report(root/'plan.json',{'purpose':'diagnostic','source_sha256':source,'aggregate_fresh_record_cap':0,
        'scored_game_cap':0,'seeds':list(DEVELOPMENT_SEEDS),'formal_scope_approved':False,
        'scope':'initial public/resume archives plus real metadata/report CLI; no learning, A2 or strength result'})
    started=time.monotonic();runs={};plan=BoardJournalPlan('development',20000000,(0,*PRE,*POST,20000000))
    for seed in DEVELOPMENT_SEEDS:
        config=BoardRuntimeConfig(seed=seed);runtime=BoardNFSPRuntime(config)
        journal=BoardMilestoneJournal(root/str(seed),runtime,plan);journal.checkpoint()
        assert runtime.records==runtime.replay.insertions==runtime.reservoir.seen==0
        runs[seed]=root/str(seed)
        del journal,runtime
        # Pure archive loading/reporting must not recreate this optional save
        # staging directory. A future authorized writer can create it normally.
        (runs[seed]/'staging').rmdir()
    registry=root/'registry'
    command=[sys.executable,'-m','graygo','cohort-register','--run-dir',str(registry),'--name','zero-record-cli-diagnostic']
    for seed,path in runs.items():command+=['--training-run',f'{seed}={path}']
    with (root/'registration-command.json').open('w') as output:subprocess.run(command,stdout=output,check=True)
    before={str(p):tree(p) for p in [registry,*runs.values()]};results={}
    for stage,count in (('pilot',54),('screening',60)):
        path=root/(stage+'-command.json')
        with path.open('w') as output:
            subprocess.run([sys.executable,'-m','graygo','cohort-report','--run-dir',str(registry),'--stage',stage],stdout=output,check=True)
        report=json.loads(path.read_text())['result'];body=report['body']
        assert body['status']=='incomplete' and len(body['missing'])==count and not body['bindings']
        assert not body['formal_scope_approved'] and not body['a2_complete'] and not body['training_trajectory_independently_verified']
        results[stage]={'status':body['status'],'missing':len(body['missing']),'report_sha256':report['sha256']}
    assert {str(p):tree(p) for p in [registry,*runs.values()]}==before
    checkpoints={}
    for seed,path in runs.items():
        bundle=path/'resume'/'000000000000';seal=CheckpointStore.verify(bundle)
        state=torch.load(bundle/'runtime.pt',weights_only=True,map_location='cpu')
        assert seal['step']==state['collector']['records']==0
        assert state['collector']['metrics']['per_role_records']==[0,0]
        assert not any(state['backend']['counters'].values())
        checkpoints[str(seed)]={'run_sha256':digest(path/'run.json'),'checkpoint_manifest_sha256':digest(bundle/'manifest.json')}
    assert source_digest()==source
    result={'verified':True,'purpose':'diagnostic','source_sha256':source,'driver_sha256':digest(Path(__file__)),
        'elapsed_seconds':time.monotonic()-started,'aggregate_fresh_records':0,'scored_games':0,'reports':results,
        'input_files_and_directories_unchanged':True,'original_zero_record_checkpoints':checkpoints,
        'formal_scope_approved':False,'a2_complete':False,
        'scope':'actual full-shape seeded initialization, immutable CLI registry and read-only incomplete reports; no learned performance or formal gate pass'}
    write_report(root/'report.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_dir');args=parser.parse_args()
    print(json.dumps(check(args.run_dir),sort_keys=True))

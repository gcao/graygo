"""Profile complete one-step task training; outputs declared per-learner budgets.

Pre/post timing recipes deliberately differ from formal opening schedules. They
cannot establish convergence and do not price 9x9 training or another target.
"""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import graygo
from graygo.manifest import source_digest,write_report
from graygo.native_decode import file_sha256
from graygo.task_training import NFSPTaskConfig,NFSPTaskTrainer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True)
    parser.add_argument('--expected-source-sha',required=True);parser.add_argument('--records',type=int,default=50000)
    parser.add_argument('--device',default='cpu');parser.add_argument('--case-minutes',type=float,default=15)
    parser.add_argument('--native-library')
    parser.add_argument('--storage',choices=('features','rule-state'),default='features')
    parser.add_argument('--task',choices=('delayed','branching','gray-pure','gray-mixed'),default='delayed')
    parser.add_argument('--learners',nargs='+',choices=('neural','tabular'),default=['neural','tabular'])
    parser.add_argument('--reuses',nargs='+',type=int,choices=(1,4),default=[1,4]);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.records<25600 or args.records%4:raise ValueError('at least two refreshes and balanced midpoint/final profiles required')
    root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    source=Path(graygo.__file__).resolve().parents[2];os.chdir(source)
    if source_digest()!=args.expected_source_sha:raise ValueError('profile loaded a different source revision')
    cases=[]
    for learner in args.learners:
        for reuse in args.reuses:
            for phase in ('pre','post'):
                c=NFSPTaskConfig(time_cap_seconds=args.case_minutes*60,task=args.task,learner=learner,records=args.records,
                    eval_every=args.records//2,q_reuse=reuse,device=args.device,purpose='profile',storage=args.storage,native_library=args.native_library,
                    reservoir_open=5_000_000 if phase=='pre' else 0,handoff_end=7_000_000 if phase=='pre' else 1000)
                cases.append({'phase':phase,'config':c.plan()})
    plan={'format_version':1,'source_commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
        'source_sha256':source_digest(),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases':cases,'scope':'complete one-step solved-task pipeline; no 9x9/other-target pricing',
        'aggregate_profile_ceiling_seconds':sum(c['config']['time_cap_seconds'] for c in cases)}
    with (root/'.driver.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'plan.json'
        if args.resume:
            if json.loads(path.read_text())!=plan:raise ValueError('profile resume plan/source differs')
        elif path.exists():raise FileExistsError('profile exists; inspect its process before resuming')
        else:write_report(path,plan)
        stopped=[]
        for name in (signal.SIGINT,signal.SIGTERM):signal.signal(name,lambda s,f:stopped.append(s))
        started=time.monotonic();rows=[];status='complete'
        for case in cases:
            if stopped:status='interrupted';break
            config=NFSPTaskConfig(**case['config']);phase=case['phase'];name=f'{config.learner}-R{config.q_reuse}-{phase}'
            write_report(root/'state.json',{'status':'running','pid':os.getpid(),'current_case':name,'completed_cases':len(rows),'source_sha256':source_digest()})
            directory=root/name
            trainer=NFSPTaskTrainer(config,directory,resume=(directory/'run.json').exists())
            result=trainer.run(stop_requested=lambda:bool(stopped),verbose=False)
            measured=result['elapsed_seconds']+result['timings']['setup']
            row={'case':name,'phase':phase,'learner':config.learner,'q_reuse':config.q_reuse,'status':result['status'],
                'records':result['records'],'seconds_including_setup':measured,'fresh_records_per_second':result['records']/measured,
                'timings':result['timings'],'counters':result['counters'],'acting_policy_counts':result['acting_policy_counts'],
                'reservoir_size':result['reservoir_size'],'replay_size':trainer.replay.size,
                'logical_replay_bytes':trainer.replay.allocated_bytes,'logical_reservoir_bytes':trainer.reservoir.allocated_bytes,
                'coverage':result['coverage'],'formal_protocol':result['formal_protocol']}
            rows.append(row);write_report(root/'profile.json',{'plan':plan,'results':rows})
            print(json.dumps({k:row[k] for k in ('case','status','records','seconds_including_setup','fresh_records_per_second')}),flush=True)
            if result['status']!='complete':status=result['status'];break
        if status=='complete' and len(rows)==len(cases):
            budgets=root/'budgets';budgets.mkdir(exist_ok=True)
            for learner in args.learners:
                for reuse in args.reuses:
                    pair=[r for r in rows if r['learner']==learner and r['q_reuse']==reuse]
                    if len(pair)!=2:raise ValueError('missing pre/post profile pair')
                    pre=next(r for r in pair if r['phase']=='pre');post=next(r for r in pair if r['phase']=='post')
                    complete=(all(r['counters']['q_updates']>0 and r['counters']['target_refreshes']>=2 for r in pair)
                        and post['counters']['supervised_updates']>0 and post['acting_policy_counts']['average']>0)
                    forecast=5_000_000/pre['records']*pre['seconds_including_setup']+10_000_000/post['records']*post['seconds_including_setup']
                    budget={'format_version':1,'source_sha256':source_digest(),'task':args.task,'learner':learner,'q_reuse':reuse,
                        'target':'one-step','device':args.device,'storage':args.storage,'native_library_sha256':None if args.native_library is None else file_sha256(args.native_library),'phases':['pre','post'],'complete_pipeline':complete,
                        'profile_plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'profile_cases':pair,
                        'phase_weighted_15M_seconds':forecast,'proposed_time_ceiling_seconds':max(900,math.ceil(2*forecast/900)*900),
                        'margin':2.,'limits':'partial buffers; scaled startup/evaluation/checkpoint overhead; no 9x9 or Retrace forecast'}
                    write_report(budgets/f'{learner}-R{reuse}.json',budget)
        write_report(root/'state.json',{'status':status,'pid':os.getpid(),'current_case':None,'completed_cases':len(rows),
            'source_sha256':source_digest(),'elapsed_this_process_seconds':time.monotonic()-started})


if __name__=='__main__':main()

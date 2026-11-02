"""Pinned-source matrix cost profiles followed by a conditional first screen.

The formal trial recipe and time ceilings are fixed before profiling. Profiles
must exercise both modes and fit a twofold margin before formal work starts.
"""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import graygo
from graygo.manifest import source_digest,write_report
from graygo.nfsp_training import NFSPMatrixConfig,NFSPMatrixTrainer


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--task',choices=('asymmetric','rps'),required=True)
    p.add_argument('--expected-source-sha',required=True);p.add_argument('--trial-hours',type=float,default=1.5);p.add_argument('--resume',action='store_true')
    args=p.parse_args();root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    source=Path(graygo.__file__).resolve().parents[2];os.chdir(source)
    if source_digest()!=args.expected_source_sha:raise ValueError('wrong pinned matrix source')
    profiles=[{'phase':phase,'config':asdict(NFSPMatrixConfig(task=args.task,learner=kind,records=100000,eval_every=50000,
                 reservoir_open=5_000_000 if phase=='pre' else 0,handoff_end=7_000_000 if phase=='pre' else 1000,time_cap_seconds=1800))}
              for phase in ('pre','post') for kind in ('tabular','neural')]
    trials=[asdict(NFSPMatrixConfig(task=args.task,learner=kind,time_cap_seconds=args.trial_hours*3600)) for kind in ('tabular','neural')]
    plan={'format_version':1,'scope':args.task+' seed-0 first screen; other families and seeds remain required',
          'source_sha256':source_digest(),'source_commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
          'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'profiles':profiles,'conditional_trials':trials,
          'profile_gate':'both pre/post complete; Q and supervised work; learned average acts; trial cap >= twice phase-weighted 15M forecast',
          'aggregate_profile_ceiling_seconds':7200,'aggregate_trial_ceiling_seconds':sum(c['time_cap_seconds'] for c in trials)}
    with (root/'.driver.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);path=root/'plan.json'
        if args.resume:
            if json.loads(path.read_text())!=plan:raise ValueError('matrix stage resume plan differs')
        elif path.exists():raise FileExistsError('stage exists; inspect its original process before resuming')
        else:write_report(path,plan)
        stopped=[]
        for name in (signal.SIGINT,signal.SIGTERM):signal.signal(name,lambda s,f:stopped.append(s))
        started=time.monotonic();cost=[];results=[];status='complete'
        for case in profiles:
            if stopped:status='interrupted';break
            c=NFSPMatrixConfig(**case['config']);name=case['phase']+'-'+c.learner
            write_report(root/'state.json',{'status':'running','phase':'profile','current_case':name,'pid':os.getpid(),'completed_profiles':len(cost),'completed_trials':0})
            directory=root/'profiles'/name;t=NFSPMatrixTrainer(c,directory,resume=(directory/'run.json').exists());r=t.run(verbose=False,stop_requested=lambda:bool(stopped))
            row={'phase':case['phase'],'learner':c.learner,'status':r['status'],'records':r['records'],'elapsed_seconds':r['elapsed_seconds'],
                 'counters':r['counters'],'acting_policy_counts':r['acting_policy_counts'],'source_sha256':r['source_sha256']}
            cost.append(row);write_report(root/'profile.json',{'plan':plan,'results':cost});print(json.dumps({'event':'matrix-profile',**row}),flush=True)
            if r['status']!='complete':status=r['status'];break
        funding=[]
        if status=='complete' and len(cost)==4:
            for config in trials:
                pre=next(r for r in cost if r['learner']==config['learner'] and r['phase']=='pre')
                post=next(r for r in cost if r['learner']==config['learner'] and r['phase']=='post')
                estimate=50*pre['elapsed_seconds']+100*post['elapsed_seconds']
                complete=(pre['counters']['q_updates']>0 and post['counters']['q_updates']>0 and post['counters']['supervised_updates']>0 and post['acting_policy_counts']['average']>0)
                passed=complete and 2*estimate<=config['time_cap_seconds']
                funding.append({'learner':config['learner'],'phase_weighted_15M_seconds':estimate,'time_ceiling_seconds':config['time_cap_seconds'],'twofold_margin_passed':passed})
            write_report(root/'budget.json',{'source_sha256':source_digest(),'task':args.task,'funding':funding,'scope':'terminal matrix only; no delayed/9x9 price'})
            if not all(r['twofold_margin_passed'] for r in funding):status='budget_limited'
        if status=='complete' and len(funding)==2:
            for config in trials:
                if stopped:status='interrupted';break
                c=NFSPMatrixConfig(**config);name=c.learner
                write_report(root/'state.json',{'status':'running','phase':'formal','current_case':name,'pid':os.getpid(),'completed_profiles':4,'completed_trials':len(results)})
                directory=root/'trials'/name;t=NFSPMatrixTrainer(c,directory,resume=(directory/'run.json').exists())
                r=t.run(stop_requested=lambda:bool(stopped));results.append(r)
                write_report(root/'summary.json',{'results':results,'a2_complete':False,'remaining':'remaining task/seed gates, cost ledger and deployment stages'})
                if r['status']!='complete':status=r['status'];break
        write_report(root/'state.json',{'status':status,'phase':None,'current_case':None,'pid':os.getpid(),'completed_profiles':len(cost),
             'completed_trials':sum(r['status']=='complete' for r in results),'elapsed_this_process_seconds':time.monotonic()-started,'source_sha256':source_digest()})


if __name__=='__main__':main()

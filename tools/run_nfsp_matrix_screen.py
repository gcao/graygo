"""Run a declared one-seed asymmetric control/neural screen from pinned source.

This is a first screen, not the complete A2 gate. Repeat with --resume only after
checking that the previous process is terminal; a live writer lock fails closed.
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
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True);parser.add_argument('--profile',required=True)
    parser.add_argument('--expected-source-sha',required=True);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--time-cap-hours',type=float,default=1);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    profile=Path(args.profile).resolve();source_root=Path(graygo.__file__).resolve().parents[2]
    os.chdir(source_root)
    if source_digest()!=args.expected_source_sha:raise ValueError('loaded source differs from declared pinned profile')
    settings=[NFSPMatrixConfig(task='asymmetric',learner=kind,seed=args.seed,time_cap_seconds=args.time_cap_hours*3600) for kind in ('tabular','neural')]
    if not all(c.formal_protocol() for c in settings):raise ValueError('first screen requires the full primary recipe')
    plan={'format_version':1,'scope':'asymmetric matrix first screen; other tasks/seeds remain required',
          'source_sha256':source_digest(),'source_commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
          'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'profile_sha256':hashlib.sha256(profile.read_bytes()).hexdigest(),'profile_path':str(profile),
          'trial_settings':[asdict(c) for c in settings],'aggregate_time_ceiling_seconds':sum(c.time_cap_seconds for c in settings)}
    with (root/'.driver.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan_path=root/'plan.json'
        if args.resume:
            if json.loads(plan_path.read_text())!=plan:raise ValueError('driver resume plan/source/profile differs')
        elif plan_path.exists():raise FileExistsError('screen already exists; inspect its live process before resuming')
        else:write_report(plan_path,plan)
        stopped=[]
        for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,lambda s,f:stopped.append(s))
        started=time.monotonic();results=[];status='complete'
        for config in settings:
            if stopped:status='interrupted';break
            write_report(root/'state.json',{'status':'running','pid':os.getpid(),'current_trial':config.learner,
                'completed_trials':len(results),'source_sha256':plan['source_sha256'],'elapsed_this_process_seconds':time.monotonic()-started})
            case=root/config.learner
            trainer=NFSPMatrixTrainer(config,case,resume=(case/'run.json').exists())
            result=trainer.run(stop_requested=lambda:bool(stopped))
            results.append(result)
            summary={'status':result['status'],'results':results,'a2_complete':False,
                'remaining':'RPS, bounded Gray Go, delayed/branching controls, additional seeds, profiles and deployment gates'}
            write_report(root/'summary.json',summary)
            if result['status']!='complete':status=result['status'];break
        write_report(root/'state.json',{'status':status,'pid':os.getpid(),'current_trial':None,'completed_trials':sum(r['status']=='complete' for r in results),
                    'source_sha256':plan['source_sha256'],'elapsed_this_process_seconds':time.monotonic()-started})


if __name__=='__main__':main()

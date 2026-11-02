"""Budgeted seed-0 one-step R=1 delayed screen, control before neural learner.

Full 15M trials require completed matching pre/post profiles. This driver never
selects R=4, moves opening, filters seeds or declares the complete A2 gate done.
"""
import argparse
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
from graygo.task_training import NFSPTaskConfig,NFSPTaskTrainer


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--profile-dir',required=True)
    p.add_argument('--expected-source-sha',required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
    root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=True);profile=Path(args.profile_dir).resolve()
    source=Path(graygo.__file__).resolve().parents[2];os.chdir(source)
    if source_digest()!=args.expected_source_sha:raise ValueError('wrong pinned delayed source')
    state=json.loads((profile/'state.json').read_text())
    if state['status']!='complete':raise ValueError('complete matching profile evidence is required')
    settings=[];budgets=[]
    for learner in ('tabular','neural'):
        raw=(profile/'budgets'/f'{learner}-R1.json').read_bytes();budget=json.loads(raw)
        if budget['source_sha256']!=source_digest() or budget['task']!='delayed' or budget['learner']!=learner or budget['q_reuse']!=1 or budget['target']!='one-step' or not budget['complete_pipeline'] or set(budget['phases'])!={'pre','post'}:
            raise ValueError('budget does not cover this source/task/learner/reuse')
        cap=budget['proposed_time_ceiling_seconds']
        if cap<2*budget['phase_weighted_15M_seconds']:raise ValueError('profile margin is inadequate')
        c=NFSPTaskConfig(time_cap_seconds=cap,task='delayed',learner=learner,budget_profile_sha256=hashlib.sha256(raw).hexdigest(),device=budget['device'])
        if not c.formal_protocol():raise ValueError('full primary delayed protocol required')
        settings.append(c);budgets.append(budget)
    plan={'format_version':1,'scope':'seed-0 R1 delayed control/neural first screen; no R4 selection or A2 completion',
        'source_sha256':source_digest(),'source_commit':subprocess.run(['git','rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip(),
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'trial_settings':[c.plan() for c in settings],
        'budget_profiles':budgets,'aggregate_time_ceiling_seconds':sum(c.time_cap_seconds for c in settings)}
    with (root/'.driver.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);path=root/'plan.json'
        if args.resume:
            if json.loads(path.read_text())!=plan:raise ValueError('delayed screen resume plan differs')
        elif path.exists():raise FileExistsError('screen exists; inspect its original handle before resuming')
        else:
            write_report(path,plan)
            for c in settings:
                raw=(profile/'budgets'/f'{c.learner}-R1.json').read_bytes()
                (root/f'{c.learner}-budget.json').write_bytes(raw)
        stopped=[]
        for name in (signal.SIGINT,signal.SIGTERM):signal.signal(name,lambda s,f:stopped.append(s))
        started=time.monotonic();results=[];status='complete'
        for c in settings:
            if stopped:status='interrupted';break
            write_report(root/'state.json',{'status':'running','pid':os.getpid(),'current_trial':c.learner,'completed_trials':len(results),'source_sha256':source_digest()})
            directory=root/c.learner;t=NFSPTaskTrainer(c,directory,resume=(directory/'run.json').exists())
            result=t.run(stop_requested=lambda:bool(stopped));results.append(result)
            write_report(root/'summary.json',{'results':results,'a2_complete':False,'remaining':'other task/seed gates, bounded profiles and full-study ledger'})
            if result['status']!='complete':status=result['status'];break
        write_report(root/'state.json',{'status':status,'pid':os.getpid(),'current_trial':None,'completed_trials':sum(r['status']=='complete' for r in results),
            'elapsed_this_process_seconds':time.monotonic()-started,'source_sha256':source_digest()})


if __name__=='__main__':main()

"""Pinned-source matrix cost profiles followed by a conditional first screen.

The formal trial recipe and time ceilings are fixed before profiling. Profiles
must exercise both modes and fit a twofold margin before formal work starts.
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
from graygo.nfsp_training import NFSPMatrixConfig,NFSPMatrixTrainer


def stage_settings(task,*,seed=0,trial_hours=1.5,tabular_trial_hours=None,neural_trial_hours=None,profile_seconds=1800):
    ceilings={'tabular':trial_hours if tabular_trial_hours is None else tabular_trial_hours,
              'neural':trial_hours if neural_trial_hours is None else neural_trial_hours}
    for value in (profile_seconds,*ceilings.values()):
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
            raise ValueError('explicit finite positive matrix profile/trial limits required')
    profiles=[{'phase':phase,'config':asdict(NFSPMatrixConfig(task=task,learner=kind,seed=seed,records=100000,eval_every=50000,
                  reservoir_open=5000000 if phase=='pre' else 0,handoff_end=7000000 if phase=='pre' else 1000,
                  time_cap_seconds=profile_seconds))}
              for phase in ('pre','post') for kind in ('tabular','neural')]
    trials=[asdict(NFSPMatrixConfig(task=task,learner=kind,seed=seed,time_cap_seconds=ceilings[kind]*3600)) for kind in ('tabular','neural')]
    return profiles,trials


def review_funding(cost,trials):
    expected={(phase,kind) for phase in ('pre','post') for kind in ('tabular','neural')}
    if len(cost)!=4 or {(row['phase'],row['learner']) for row in cost}!=expected:
        raise ValueError('complete distinct pre/post matrix profile inventory required')
    funding=[]
    for config in trials:
        pair=[row for row in cost if row['learner']==config['learner']]
        pre=next(row for row in pair if row['phase']=='pre');post=next(row for row in pair if row['phase']=='post')
        for row in pair:
            seconds=row['seconds_including_setup']
            if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0:
                raise ValueError('positive complete measured matrix timing required')
        estimate=50*pre['seconds_including_setup']+100*post['seconds_including_setup']
        complete=(all(row['status']=='complete' and row['records']==100000 and row['counters']['q_updates']>0
                      and row['counters']['target_refreshes']>=2 for row in pair)
            and pre['counters']['supervised_updates']==0 and pre['acting_policy_counts']['average']==0
            and post['counters']['supervised_updates']>0 and post['acting_policy_counts']['average']>0)
        funding.append({'learner':config['learner'],'phase_weighted_15M_seconds':estimate,
            'time_ceiling_seconds':config['time_cap_seconds'],'twofold_margin_passed':complete and 2*estimate<=config['time_cap_seconds'],
            'measured_setup_included':True,'complete_pre_post_work':complete})
    return funding


def measured_setup(previous,current,*,resuming):
    if type(current) not in (int,float) or not math.isfinite(current) or current<0:
        raise ValueError('finite nonnegative constructor timing required')
    if resuming:
        if type(previous) not in (int,float) or not math.isfinite(previous) or previous<0:
            raise ValueError('original measured setup is missing; cannot silently reset profile/trial overhead')
        return previous+current
    return current


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--task',choices=('asymmetric','rps'),required=True)
    p.add_argument('--expected-source-sha',required=True);p.add_argument('--trial-hours',type=float,default=1.5);p.add_argument('--resume',action='store_true')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--tabular-trial-hours',type=float);p.add_argument('--neural-trial-hours',type=float)
    p.add_argument('--profile-seconds',type=float,default=1800)
    args=p.parse_args();root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    source=Path(graygo.__file__).resolve().parents[2];os.chdir(source)
    if source_digest()!=args.expected_source_sha:raise ValueError('wrong pinned matrix source')
    profiles,trials=stage_settings(args.task,seed=args.seed,trial_hours=args.trial_hours,tabular_trial_hours=args.tabular_trial_hours,
                                  neural_trial_hours=args.neural_trial_hours,profile_seconds=args.profile_seconds)
    plan={'format_version':1,'scope':args.task+f' original seed {args.seed}; other required families and seeds remain explicit',
          'source_sha256':source_digest(),'source_commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
          'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'profiles':profiles,'conditional_trials':trials,
          'profile_gate':'both pre/post complete; Q and fresh-record target work; supervised work and learned-average acting post opening; trial cap >= twice phase-weighted 15M forecast including measured setup',
          'aggregate_profile_ceiling_seconds':sum(c['config']['time_cap_seconds'] for c in profiles),'aggregate_trial_ceiling_seconds':sum(c['time_cap_seconds'] for c in trials)}
    with (root/'.driver.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);path=root/'plan.json'
        if args.resume:
            if json.loads(path.read_text())!=plan:raise ValueError('matrix stage resume plan differs')
        elif path.exists():raise FileExistsError('stage exists; inspect its original process before resuming')
        else:write_report(path,plan)
        stopped=[]
        for name in (signal.SIGINT,signal.SIGTERM):signal.signal(name,lambda s,f:stopped.append(s))
        previous_cost=json.loads((root/'profile.json').read_text())['results'] if (root/'profile.json').exists() else []
        started=time.monotonic();cost=[];results=[];status='complete'
        for case in profiles:
            if stopped:status='interrupted';break
            c=NFSPMatrixConfig(**case['config']);name=case['phase']+'-'+c.learner
            write_report(root/'state.json',{'status':'running','phase':'profile','current_case':name,'pid':os.getpid(),'completed_profiles':len(cost),'completed_trials':0})
            directory=root/'profiles'/name;resuming=(directory/'run.json').exists()
            old=next((r for r in previous_cost if r['phase']==case['phase'] and r['learner']==c.learner),None)
            if resuming and old is None:raise ValueError('original profile setup measurement missing; preserve this incomplete run')
            setup=time.monotonic();t=NFSPMatrixTrainer(c,directory,resume=resuming);setup=measured_setup(None if old is None else old.get('setup_seconds'),time.monotonic()-setup,resuming=resuming)
            r=t.run(verbose=False,stop_requested=lambda:bool(stopped))
            row={'phase':case['phase'],'learner':c.learner,'status':r['status'],'records':r['records'],'elapsed_seconds':r['elapsed_seconds'],
                 'setup_seconds':setup,'seconds_including_setup':r['elapsed_seconds']+setup,
                 'counters':r['counters'],'acting_policy_counts':r['acting_policy_counts'],'source_sha256':r['source_sha256']}
            cost.append(row);write_report(root/'profile.json',{'plan':plan,'results':cost});print(json.dumps({'event':'matrix-profile',**row}),flush=True)
            if r['status']!='complete':status=r['status'];break
        funding=[]
        if status=='complete' and len(cost)==4:
            funding=review_funding(cost,trials)
            write_report(root/'budget.json',{'source_sha256':source_digest(),'task':args.task,'funding':funding,'scope':'terminal matrix only; no delayed/9x9 price'})
            if not all(r['twofold_margin_passed'] for r in funding):status='budget_limited'
        if status=='complete' and len(funding)==2:
            for config in trials:
                if stopped:status='interrupted';break
                c=NFSPMatrixConfig(**config);name=c.learner
                write_report(root/'state.json',{'status':'running','phase':'formal','current_case':name,'pid':os.getpid(),'completed_profiles':4,'completed_trials':len(results)})
                directory=root/'trials'/name;resuming=(directory/'run.json').exists();portion=root/(name+'-process-portion.json')
                previous=json.loads(portion.read_text()).get('setup_seconds') if portion.exists() else None
                if resuming and previous is None:raise ValueError('original trial setup measurement missing; preserve this incomplete run')
                setup=time.monotonic();t=NFSPMatrixTrainer(c,directory,resume=resuming);setup=measured_setup(previous,time.monotonic()-setup,resuming=resuming)
                r=t.run(stop_requested=lambda:bool(stopped));results.append(r)
                write_report(root/(name+'-process-portion.json'),{'setup_seconds':setup,'loop_seconds':r['elapsed_seconds'],
                    'seconds_including_setup':setup+r['elapsed_seconds'],'scope':'known constructor and trainer-loop intervals; excludes outer import/driver and final-report/return overhead'})
                write_report(root/'summary.json',{'results':results,'a2_complete':False,'remaining':'remaining task/seed gates, cost ledger and deployment stages'})
                if r['status']!='complete':status=r['status'];break
        write_report(root/'state.json',{'status':status,'phase':None,'current_case':None,'pid':os.getpid(),'completed_profiles':len(cost),
             'completed_trials':sum(r['status']=='complete' for r in results),'elapsed_this_process_seconds':time.monotonic()-started,'source_sha256':source_digest()})


if __name__=='__main__':main()

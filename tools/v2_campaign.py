"""Unlimited v2 learning with checkpointed, asynchronous frozen CPU evaluations.

One owner holds the learner lock. A single independently locked evaluation job
may run concurrently; its immutable candidate/incumbent/attempt survive restarts.
V1 data and its champion pointer are never modified by this controller.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
from graygo.board_runtime import BoardRuntimeConfig,BoardNFSPRuntime
from graygo.confidence import canonical_sha
from graygo.engine import Rules
from graygo.manifest import source_digest,write_report
from graygo.native_decode import build_native
from graygo.storage import CheckpointStore,DEFAULT_RESERVE,digest
from tools.parallel_gate import ParallelGate
from tools.v2_gate import V2GatePlan
from tools.v2_models import describe,export,archive_runtime,publish

VERSION='graygo/v2-campaign/1'


def save(path,body):write_report(path,{'body':body,'sha256':canonical_sha(body)})
def read(path):
    wrapper=json.loads(Path(path).read_text())
    if wrapper['sha256']!=canonical_sha(wrapper['body']):raise ValueError('campaign metadata checksum differs')
    return wrapper['body']


def implementation():
    return {name:digest(Path(__file__).with_name(name)) for name in ('v2_campaign.py','v2_models.py','v2_gate.py','parallel_gate.py','policy_selection.py')}


def prepare(root,archive,champion,historical,*,device='cuda',envs=64,validation=False,project=None,calibration=None):
    calibration_data=None
    if not validation:
        if calibration is None:raise ValueError('pre-launch gate calibration required')
        calibration_data=json.loads(Path(calibration).read_text())
        if (calibration_data.get('source_sha256')!=source_digest() or calibration_data.get('gate_plan_sha256')!=canonical_sha(V2GatePlan().identity())
                or not calibration_data.get('power_check_passed') or not calibration_data.get('null_check_passed') or calibration_data.get('trials_per_cell',0)<10000):
            raise ValueError('gate calibration code, power, or null checks differ')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    native=build_native(root/'native')['library']
    parent=CheckpointStore.verify(archive)['step']
    cfg=BoardRuntimeConfig(rules=Rules(ko='full-history'),device=device,envs=envs,q_reuse=4,
        native_library=native,history_features=True,initial_weights_archive=str(archive),
        reservoir_open=0,handoff_end=2,epsilon_offset_records=parent)
    # Bind separate source data and all transfer identities before learning.
    base=export(describe(champion,'v2-bootstrap-from-'+str(CheckpointStore.verify(champion)['step'])),root/'releases')
    history=[export(describe(path,'v2-historical-'+str(CheckpointStore.verify(path)['step'])),root/'releases') for path in historical]
    if not history:raise ValueError('historical learned opponents required')
    plan={'version':VERSION,'source_sha256':source_digest(),'implementation':implementation(),'config':cfg.plan(),
          'gate_calibration':calibration_data,'parent_records':parent,'initial_champion':base,'initial_history':history,'checkpoint_every':262144,
          'archive_every':262144,'screen_games':100,'gate_looks':[500,2000,4000],'gate_workers':8,
          'gate_total_alpha':.05,'screen_interval':1800.,'force_gate_interval':7200.,'validation_only':validation,
          'project':str(Path(project or Path(__file__).resolve().parents[1]).resolve()),
          'publication':None if validation else str(Path(project or Path(__file__).resolve().parents[1]).resolve().parent/'graygo-arena/.runtime/models/v2-champion.json'),
          'learning_selection':'raw_average_control','reservoir':'uniform_v2_only','auxiliary_margin_head':False}
    save(root/'plan.json',plan)
    write_report(root/'run.json',{'format_version':1,'rule_version':cfg.rules.rule_version,'encoding_version':'graygo/encoding/ko-hints/2','source_sha256':source_digest(),'rules':cfg.rules.to_dict(),'config':cfg.plan(),
                                'initialization':{'parent_records':parent,'weights_only':True,'replay_imported':False}})
    state={'version':VERSION,'campaign':str(root),'plan_sha256':canonical_sha(plan),'champion':base,'history':history[-3:],
           'promotions':0,'gate_attempts':0,'job_serial':0,'job':None,'next_screen':0.,'last_gate_at':0.,
           'last_screen':None,'last_gate':None,'checkpoints':0,'records':0,'status':'prepared','started_at':None}
    save(root/'state.json',state)
    return plan


def process_alive(pid,root):
    try:
        p=Path('/proc')/str(pid)
        return p.exists() and (p/'stat').read_text().split()[2]!='Z' and str(root).encode() in (p/'cmdline').read_bytes().split(b'\0')
    except (OSError,ValueError):return False


def spawn_job(root,plan,state,candidate,mode):
    state['job_serial']+=1
    if mode=='gate':state['gate_attempts']+=1;state['last_gate_at']=time.time()
    job_root=root/'evaluations'/f"{state['job_serial']:06d}"
    job_root.mkdir(parents=True,exist_ok=False)
    body={'version':VERSION,'source_sha256':source_digest(),'implementation':implementation(),'mode':mode,
          'candidate':candidate,'champion':state['champion'],'pool':{'historical-'+str(i):d for i,d in enumerate(state['history'])},
          'attempt':state['gate_attempts'] if mode=='gate' else None,'seed':11073+state['job_serial']*101,
          'looks':plan['gate_looks'] if mode=='gate' else [plan['screen_games']],'workers':plan['gate_workers'],
          'total_alpha':plan['gate_total_alpha'],'records':candidate['records']}
    save(job_root/'job.json',body)
    state['job']={'root':str(job_root),'mode':mode,'candidate':candidate,'champion_sha256':state['champion']['policy_sha256'],
                  'attempt':body['attempt'],'pid':None,'started_at':time.time()}
    # Commit attempt allocation before process creation; a restart reuses it.
    save(root/'state.json',state)
    launch_job(root,state)


def launch_job(root,state):
    job=state['job'];folder=Path(job['root']);env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (folder/'worker.log').open('ab') as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--job',str(folder)],cwd=Path(__file__).resolve().parents[1],
            env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    job['pid']=process.pid;save(root/'state.json',state)


def run_job(root):
    root=Path(root).resolve();lock=(root/'.job.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=read(root/'job.json')
    if job['source_sha256']!=source_digest() or job['implementation']!=implementation():raise ValueError('frozen job code changed')
    plan=V2GatePlan(looks=tuple(job['looks']),seed=job['seed'],alpha=job['total_alpha'])
    gate=ParallelGate(plan,workers=job['workers'])
    result=gate.evaluate_descriptors(job['candidate'],job['champion'],job['pool'],root,
        round_index=job['attempt']-1 if job['attempt'] else 0,time_cap_seconds=3600)
    save(root/'result.json',{'job_sha256':canonical_sha(job),'mode':job['mode'],'attempt':job['attempt'],
                           'candidate':job['candidate']['policy_sha256'],'champion':job['champion']['policy_sha256'],'decision':result})


def finish_job(root,plan,state):
    job=state.get('job')
    if job is None:return
    folder=Path(job['root']);path=folder/'result.json'
    if not path.exists():
        if not process_alive(job['pid'],folder):
            # Bounded retry with a visible error; preserve the exact frozen job.
            job['restarts']=job.get('restarts',0)+1
            if job['restarts']>3:raise RuntimeError('evaluation job repeatedly exited; inspect '+str(folder/'worker.log'))
            launch_job(root,state)
        return
    result=read(path);spec=read(folder/'job.json')
    if result['job_sha256']!=canonical_sha(spec) or result['mode']!=job['mode'] or result['candidate']!=job['candidate']['policy_sha256'] or result['champion']!=job['champion_sha256']:
        raise ValueError('evaluation job result identity differs')
    decision=result['decision']
    if decision['status']!='complete':
        path.rename(folder/('partial-'+str(time.time_ns())+'.json'));launch_job(root,state);return
    if state['champion']['policy_sha256']!=result['champion']:raise ValueError('refuse stale-incumbent release')
    state['job']=None;now=time.time();score=decision['head_win_score']['mean']
    if result['mode']=='screen':
        state['last_screen']={'records':spec['records'],'score':score,'root':str(folder),'measured_at':now}
        interval=900. if score>=.60 else 1800. if score>=.55 else 3600.
        state['next_screen']=now+interval
        if score>=.57 or now-state['last_gate_at']>=plan['force_gate_interval']:
            spawn_job(root,plan,state,job['candidate'],'gate')
    else:
        if result['attempt']!=job['attempt'] or decision['actual_attempt']!=job['attempt']:raise ValueError('gate attempt allocation changed')
        state['last_gate']={'records':spec['records'],'root':str(folder),'decision':decision,'measured_at':now}
        if decision['numeric_pass']:
            state['history']=(state['history']+[state['champion']])[-3:]
            state['champion']=job['candidate'];state['promotions']+=1
            # Persist the release decision before touching the atomic pointer.
            save(root/'state.json',state)
            if plan['publication']:publish(plan['publication'],state['champion'],state)
        state['next_screen']=min(state['next_screen'],now+1800.)
    save(root/'state.json',state)


def status(root,state,runtime,phase):
    metrics=runtime.collector.metrics;hist=metrics['episode_lengths'];games=sum(hist)
    loss=runtime.loss;q=loss['q_updates'];sl=loss['supervised_updates']
    body={'version':VERSION,'status':state['status'],'phase':phase,'pid':os.getpid(),'records':runtime.records,
          'parent_records':runtime.initial_weights and runtime.config.epsilon_offset_records,'rule_version':'graygo/2',
          'encoding_version':runtime.spec.version,'native_decoder':runtime.spec.native_library is not None,'envs':runtime.config.envs,
          'counters':asdict(runtime.backend.counters),'mean_rounds':sum(i*n for i,n in enumerate(hist))/games if games else None,
          'repetition_rejections':metrics['repetition_rejections'],'physical_rounds':metrics['physical_rounds'],
          'q_loss':loss['q_loss_sum']/q if q else None,'average_loss':loss['supervised_loss_sum']/sl if sl else None,
          'champion':state['champion']['label'],'promotions':state['promotions'],'gate_attempts':state['gate_attempts'],
          'evaluation':state['job'],'last_screen':state['last_screen'],'last_gate':state['last_gate'],'timings':runtime.timings,
          'unlimited':True,'time_limit_seconds':None,'updated_at':time.time()}
    write_report(root/'status.json',body)


def run(root,*,until=None,publication_enabled=True):
    if not publication_enabled and (until is None or until>65536):raise ValueError('suppressed publication requires a bounded validation endpoint')
    root=Path(root).resolve();lock=(root/'.learner.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(root/'plan.json');state=read(root/'state.json')
    if plan['source_sha256']!=source_digest() or plan['implementation']!=implementation() or state['plan_sha256']!=canonical_sha(plan):raise ValueError('campaign plan/code changed')
    if until is not None and (type(until) is not int or until<2 or until%2):raise ValueError('even operational endpoint required')
    data=dict(plan['config']);data['rules']=Rules(**data['rules']);runtime=BoardNFSPRuntime(BoardRuntimeConfig(**data))
    store=CheckpointStore(root)
    try:latest=store.latest()
    except FileNotFoundError:latest=None
    if latest is not None:runtime.restore(latest)
    if state['status'] in ('stopped','operational_boundary') and state.get('job'):state['job']['restarts']=0
    state['status']='running';state['started_at']=state['started_at'] or time.time();state['records']=runtime.records;save(root/'state.json',state)
    stopped=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.append(True))
    if not plan['validation_only'] and publication_enabled:
        project=Path(plan['project'])
        write_report(project/'runs/latest-training-loop.json',{'version':VERSION,'run_dir':str(root),'supervisor_pid':os.getpid(),'unlimited':True})
        publish(plan['publication'],state['champion'],state)
    last_checkpoint=runtime.records if latest is not None else -1;next_checkpoint=((runtime.records//plan['checkpoint_every'])+1)*plan['checkpoint_every'];heartbeat=0.
    def checkpoint():
        nonlocal last_checkpoint
        if runtime.records!=last_checkpoint:
            runtime.save(root,controller_state={'campaign':str(root),'plan_sha256':state['plan_sha256']});last_checkpoint=runtime.records
            state['checkpoints']+=1;state['records']=runtime.records;save(root/'state.json',state)
    try:
        if latest is None:checkpoint();archive_runtime(runtime,root,DEFAULT_RESERVE)
        while not stopped and (until is None or runtime.records<until):
            if time.monotonic()-heartbeat>=30:
                finish_job(root,plan,state);status(root,state,runtime,'training');heartbeat=time.monotonic()
            count=min(2*runtime.config.envs,runtime.backend.counters.until_q_boundary(runtime.backend.config),next_checkpoint-runtime.records)
            if until is not None:count=min(count,until-runtime.records)
            for boundary in (runtime.config.reservoir_open,runtime.config.handoff_end):
                if runtime.records<boundary:count=min(count,boundary-runtime.records)
            runtime.advance(count)
            if runtime.records==next_checkpoint:
                checkpoint();archive=archive_runtime(runtime,root,DEFAULT_RESERVE)
                finish_job(root,plan,state)
                if state['job'] is None and time.time()>=state['next_screen']:
                    candidate=export(describe(archive,'v2-candidate-'+str(runtime.records)),root/'releases')
                    spawn_job(root,plan,state,candidate,'screen')
                next_checkpoint+=plan['checkpoint_every']
        checkpoint();state['status']='stopped' if stopped else 'operational_boundary';save(root/'state.json',state);status(root,state,runtime,'idle')
    except BaseException:
        state['status']='error';save(root/'state.json',state);status(root,state,runtime,'error');raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path);p.add_argument('--job',type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--archive',type=Path)
    p.add_argument('--champion',type=Path);p.add_argument('--historical',type=Path,action='append',default=[])
    p.add_argument('--calibration',type=Path);p.add_argument('--project',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--envs',type=int,default=64);p.add_argument('--validation',action='store_true');p.add_argument('--until-records',type=int);p.add_argument('--no-publication',action='store_true')
    a=p.parse_args()
    if a.job:run_job(a.job);return
    if a.prepare:prepare(a.run_dir,a.archive,a.champion,a.historical,device=a.device,envs=a.envs,validation=a.validation,project=a.project,calibration=a.calibration)
    if a.run:run(a.run_dir,until=a.until_records,publication_enabled=not a.no_publication)

if __name__=='__main__':main()

"""Prepare or explicitly run a bounded comparison of two fresh training stacks.

Both arms get the same wall-clock ceiling, including initialization/checkpoints.
The legacy arm keeps its native training rules; common evaluation uses the new
engine. This compares practical stacks, not an isolated causal algorithm change.
"""
import argparse
from dataclasses import asdict,dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import torch
from graygo.confidence import canonical_sha
from graygo.engine import Rules
from graygo.evaluation import BaselinePolicy,EvaluationPlan,EvaluationRun,FrozenNeuralPolicy,seed_for
from graygo.manifest import source_digest,write_report
from graygo.models import ModelConfig,PolicyNet,QNet
from graygo.storage import CheckpointStore,DEFAULT_RESERVE,digest
from tools.champion_screen import summarize
from tools.legacy_policy import LegacyPolicy


@dataclass(frozen=True)
class ComparisonPlan:
    total_seconds:float
    seed:int=17
    evaluation_fraction:float=.1
    games:int=20
    visits:int=900
    device:str='cuda'

    def __post_init__(self):
        if type(self.total_seconds) not in (int,float) or not math.isfinite(self.total_seconds) or not 60<=self.total_seconds<=3600:
            raise ValueError('comparison needs an explicit total budget between sixty seconds and one hour')
        if self.evaluation_fraction!=.1 or type(self.games) is not int or self.games<4 or self.games%2:
            raise ValueError('fixed ten-percent evaluation share and even fixed sample count required')
        if type(self.seed) is not int or self.seed<0 or type(self.visits) is not int or self.visits<1:
            raise ValueError('nonnegative seed and positive search visits required')
        if self.device not in ('cpu','cuda'):raise ValueError('explicit CPU/CUDA execution required')

    @property
    def arm_seconds(self):return self.total_seconds*(1-self.evaluation_fraction)/2


def run_limited(command,*,cwd,log,seconds):
    """The whole process group gets a hard deadline, even inside native search."""
    if not math.isfinite(seconds) or seconds<=0:raise ValueError('positive process budget required')
    started=time.monotonic();grace=min(10.,seconds*.1);terminated=killed=False
    environment=dict(os.environ,PYTHONPATH=str(Path(cwd)/'src')+os.pathsep+str(cwd),PYTHONUNBUFFERED='1')
    with Path(log).open('wb') as stream:
        child=subprocess.Popen(command,cwd=cwd,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            try:child.wait(timeout=max(.01,seconds-grace))
            except subprocess.TimeoutExpired:
                terminated=True;os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=max(.01,seconds-(time.monotonic()-started)))
                except subprocess.TimeoutExpired:
                    killed=True;os.killpg(child.pid,signal.SIGKILL);child.wait()
        finally:
            if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
    return {'returncode':child.returncode,'wall_seconds':time.monotonic()-started,
            'ceiling_seconds':seconds,'soft_stop':terminated,'hard_stop':killed}


def prepare(root,plan,repo,reference):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    snapshot=root/'legacy-source';snapshot.mkdir()
    repo=Path(repo).resolve();reference=Path(reference).resolve()
    # Snapshot actual Python/native artifacts, including local changes, without
    # changing graygo-old or copying its launcher, secrets, data or checkpoints.
    needed=('model.py','train.py','selfplay.py','selfplay_cpp.py','export_model.py','engine.py')
    native=('graygo_engine.','mcts_engine.','mcts_engine_callback.')
    for path in repo.iterdir():
        if path.name in needed or (path.suffix=='.so' and path.name.startswith(native)):
            shutil.copy2(path,snapshot/path.name)
    if not all((snapshot/name).exists() for name in needed):raise ValueError('legacy training libraries missing')
    shutil.copy2(reference,root/'reference.pt')
    body={'version':'graygo/training-comparison/1','plan':asdict(plan),'arm_seconds':plan.arm_seconds,
          'source_sha256':source_digest(),'new_checkout':str(Path(__file__).resolve().parents[1]),
          'tool_sha256':digest(Path(__file__)),'legacy_worker_sha256':digest(Path(__file__).with_name('legacy_training_arm.py')),
          'new_tools':{name:digest(Path(__file__).with_name(name)) for name in
                       ('champion_loop.py','champion_gate.py','champion_screen.py','legacy_policy.py','nfsp_resources.py','task_pipeline_parity.py')},
          'legacy_files':{p.name:digest(p) for p in sorted(snapshot.iterdir())},'reference_sha256':digest(root/'reference.pt'),
          'initialization':'both learners fresh; reference checkpoint is evaluation-only',
          'averaging_schedule':{'reservoir_open':5000000,'handoff_end':7000000},
          'evaluation_rules':asdict(Rules()),'promotion_permitted':False,
          'limitations':['native legacy training rules and larger architecture differ',
                        'short prefix may finish before average-policy initialization',
                        'one seed and small descriptive evaluation cannot establish algorithm superiority']}
    write_report(root/'plan.json',{'body':body,'sha256':canonical_sha(body)})
    write_report(root/'status.json',{'status':'prepared','training_started':False})
    return body


def resume_policies(root):
    directory=CheckpointStore(root).latest();CheckpointStore.verify(directory)
    data=torch.load(directory/'runtime.pt',map_location='cpu',weights_only=True)
    cfg=data['config'];shape=ModelConfig(**{k:cfg[k] for k in ('width','blocks','norm_groups')})
    ready=data['backend']['counters']['supervised_updates']>0
    policies={}
    for head,cls,key in [('greedy_q',QNet,'q'),('average',PolicyNet,'average')]:
        if head=='average' and not ready:continue
        with torch.random.fork_rng(devices=[]):model=cls(shape)
        model.load_state_dict(data['backend'][key],strict=True)
        policies['new-'+head]=FrozenNeuralPolicy(model,representation=head,label='new-'+head,
            origin={'rules':cfg['rules'],'records':data['backend']['counters']['fresh_records'],
                    'resume_manifest_sha256':digest(directory/'manifest.json')})
    return policies,{'checkpoint':str(directory),'average_ready':ready,'counters':data['backend']['counters']}


def execute(root):
    root=Path(root).resolve()
    torch.set_num_threads(1)
    lock=(root/'.comparison.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        # Let the outer watchdog's soft stop unwind wait()/checkpoint loading
        # and kill an active training child's separate process group in finally.
        def terminate(*unused):raise SystemExit('comparison watchdog stopped the worker')
        for kind in (signal.SIGTERM,signal.SIGINT):signal.signal(kind,terminate)
        wrapper=json.loads((root/'plan.json').read_text());body=wrapper['body']
        if wrapper['sha256']!=canonical_sha(body):raise ValueError('comparison plan changed')
        if json.loads((root/'status.json').read_text())['status']!='prepared':raise ValueError('comparison already started; cannot buy a new budget by rerunning')
        if (body['source_sha256']!=source_digest() or body['tool_sha256']!=digest(Path(__file__))
            or body['legacy_worker_sha256']!=digest(Path(__file__).with_name('legacy_training_arm.py'))):raise ValueError('prepared implementation changed')
        for name,sha in body['new_tools'].items():
            if digest(Path(__file__).with_name(name))!=sha:raise ValueError('prepared controller changed')
        for name,sha in body['legacy_files'].items():
            if digest(root/'legacy-source'/name)!=sha:raise ValueError('legacy snapshot changed')
        if digest(root/'reference.pt')!=body['reference_sha256']:raise ValueError('reference checkpoint changed')
        plan=ComparisonPlan(**body['plan']);checkout=Path(body['new_checkout']);started=time.monotonic()
        report={'status':'running','plan_sha256':wrapper['sha256'],'arms':{},'promotion_permitted':False}
        write_report(root/'status.json',report)
        commands={
            'new':[sys.executable,'-u','-m','tools.champion_loop','--practice','--run-dir',str(root/'new'),
                   '--records','5000000','--block-records','262144','--time-cap-seconds',str(plan.arm_seconds),
                   '--evaluation-fraction','0','--device',plan.device,'--seed',str(plan.seed),
                   '--old-repo',str(root/'legacy-source'),'--old-checkpoint',str(root/'reference.pt')],
            'legacy':[sys.executable,'-u','-m','tools.legacy_training_arm','--repo',str(root/'legacy-source'),
                      '--root',str(root/'legacy'),'--seconds',str(plan.arm_seconds),'--device',plan.device,
                      '--seed',str(plan.seed),'--visits',str(plan.visits)]}
        for name,command in commands.items():
            remaining=plan.total_seconds-(time.monotonic()-started)
            if remaining<=0:break
            report['active_arm']=name;write_report(root/'status.json',report)
            report['arms'][name]=run_limited(command,cwd=checkout,log=root/(name+'.log'),seconds=min(plan.arm_seconds,remaining))
            report.pop('active_arm',None)
            write_report(root/'status.json',report)
            if report['arms'][name]['returncode']>0:raise RuntimeError(name+' training failed; inspect its log')
        # Setup, checkpoint loads and model conversion are inside the global cap.
        evaluation_started=time.monotonic()
        policies,report['new_checkpoint']=resume_policies(root/'new')
        if (root/'legacy/latest.pt').exists():
            policies['legacy-fresh']=LegacyPolicy(root/'legacy-source',root/'legacy/latest.pt',label='legacy-fresh')
            legacy=torch.load(root/'legacy/latest.pt',map_location='cpu',weights_only=True)
            report['legacy_checkpoint']=legacy['metadata'];del legacy
        policies['reference']=LegacyPolicy(root/'legacy-source',root/'reference.pt',label='reference')
        jobs=[(name,p,BaselinePolicy(opponent),opponent) for name,p in policies.items() for opponent in ('random','tactical')]
        if 'legacy-fresh' in policies:jobs.append(('new-greedy_q',policies['new-greedy_q'],policies['legacy-fresh'],'legacy-fresh'))
        report['evaluations']={}
        for index,(name,subject,opponent,label) in enumerate(jobs):
            remaining=min(plan.total_seconds-(time.monotonic()-started),
                          plan.total_seconds*plan.evaluation_fraction-(time.monotonic()-evaluation_started))
            if remaining<=0:break
            eval_plan=EvaluationPlan('training-comparison',name,label,Rules(),seed_for(plan.seed,'common-evaluation',label),
                                     (plan.games,),wrapper['sha256'])
            run=EvaluationRun(eval_plan,root/'evaluation'/name/label,subject,opponent)
            progress=run.run(plan.games,time_cap_seconds=remaining/(len(jobs)-index))
            report['evaluations'][name+'/'+label]=dict(summarize(run.records()),status=progress['status'])
            write_report(root/'status.json',report)
        report.update(status='finished',wall_seconds=time.monotonic()-started,
                      scope='one-seed bounded practical-stack comparison; no promotion or formal study conclusion')
        write_report(root/'status.json',report);return report
    except BaseException as error:
        # A failed/aborted invocation is never silently restarted with a fresh cap.
        if 'report' in locals():
            report.update(status='failed',error=repr(error));write_report(root/'status.json',report)
        raise
    finally:lock.close()


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    mode.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--total-seconds',type=float)
    parser.add_argument('--old-repo');parser.add_argument('--reference-checkpoint');parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.prepare:
        if None in (args.total_seconds,args.old_repo,args.reference_checkpoint):parser.error('preparation needs budget, old repository and reference checkpoint')
        result=prepare(args.run_dir,ComparisonPlan(args.total_seconds,device=args.device),args.old_repo,args.reference_checkpoint)
    elif args.worker:result=execute(args.run_dir)
    else:
        root=Path(args.run_dir).resolve();wrapper=json.loads((root/'plan.json').read_text())
        if wrapper.get('sha256')!=canonical_sha(wrapper.get('body')):raise ValueError('comparison plan changed')
        plan=ComparisonPlan(**wrapper['body']['plan'])
        # Exclusive launch record prevents even an initialization crash from
        # turning a later --run into another unrecorded compute allocation.
        with (root/'launch.json').open('x') as stream:json.dump({'started_at':time.time(),'ceiling_seconds':plan.total_seconds},stream)
        def terminate(*unused):raise SystemExit('comparison launcher stopped')
        for kind in (signal.SIGTERM,signal.SIGINT):signal.signal(kind,terminate)
        supervision=run_limited([sys.executable,'-u','-m','tools.training_comparison','--worker','--run-dir',str(root)],
            cwd=wrapper['body']['new_checkout'],log=root/'comparison.log',seconds=plan.total_seconds)
        write_report(root/'supervision.json',supervision)
        result=json.loads((root/'status.json').read_text())
        if result['status']!='finished' and supervision['soft_stop']:
            result.update(status='budget_finished',supervision=supervision);write_report(root/'status.json',result)
        elif result['status'] not in ('finished','failed'):
            result.update(status='failed',supervision=supervision);write_report(root/'status.json',result)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()

"""Explicit, state-preserving handoff into an experimental champion loop.

The parent remains immutable. Numerical implementation, configuration, complete
learner state and cumulative exploration clocks are preserved across the handoff.
"""
import argparse
import ast
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import torch
from graygo.board_execution import BoardExecutor,BoardExecutionPlan
from graygo.board_journal import BoardJournalPlan
from graygo.board_runtime import BoardRuntimeConfig
from graygo.confidence import canonical_sha
from graygo.engine import Rules
from graygo.manifest import source_digest,write_report
from graygo.storage import CheckpointStore,DEFAULT_RESERVE,digest
from tools.parallel_gate import ParallelGate
from tools.champion_gate import GatePlan,IndependentGate,HEAD_TO_HEAD_HISTORY,LEGACY_POOL_GAIN
from tools.champion_loop import ChampionLoop,LoopPlan
from tools.champion_screen import IndependentScreen,ScreenPlan
from tools.training_comparison import run_limited


def source_files(checkout):
    root=Path(checkout)/'src/graygo'
    return {p.relative_to(root).as_posix():p for p in sorted([*root.glob('*.py'),*(root/'native').glob('*.cpp')])}


def tree_digest(checkout):
    h=hashlib.sha256()
    for name,path in source_files(checkout).items():h.update(name.encode()+b'\0'+path.read_bytes()+b'\0')
    return h.hexdigest()


def compatible_sources(parent,current):
    before=source_files(parent);after=source_files(current)
    if set(before)!=set(after):raise ValueError('learner source inventory changed')
    for name in before:
        if name in ('board_execution.py','board_journal.py'):continue # Controller/scope changes are explicit.
        if before[name].read_bytes()==after[name].read_bytes():continue
        if name!='board_runtime.py':raise ValueError('numerical learner source changed: '+name)
        trees=[ast.parse(p.read_text()) for p in (before[name],after[name])]
        restores=[]
        for tree in trees:
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BoardNFSPRuntime')
            snapshot=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_snapshot')
            recovery=ast.parse('''if self.snapshot is None:
    self.snapshot=next((actor.snapshot for episode in self.collector.active if episode is not None
                       for actor in episode.actors.values() if actor.snapshot.version==version),None)
''').body[0]
            # Admit this exact cache-identity repair, never another model update
            # or a changed action distribution under the same version.
            snapshot.body=[node for node in snapshot.body if ast.dump(node)!=ast.dump(recovery)]
            method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='restore')
            # The only loader change may precede the unchanged state loading.
            start=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign)
                       and any(isinstance(t,ast.Name) and t.id=='state' for t in n.targets))
            restores.append(ast.dump(ast.Module(body=method.body[start:],type_ignores=[])))
            cls.body[cls.body.index(method)]=ast.Pass()
        if restores[0]!=restores[1] or ast.dump(trees[0])!=ast.dump(trees[1]):
            raise ValueError('runtime training or state restoration changed')
    return {'parent_source_sha256':tree_digest(parent),'current_source_sha256':tree_digest(current),
            'rule':'identical numerical code; controller/source-admission and exact restored-snapshot canonicalization may differ'}


def parent_lock(root):
    lock=(Path(root)/'.execution.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return lock


def prepare(root,parent,parent_checkout,publication,*,total_seconds=3600,records=20000000,block=262144,
            reserve_bytes=DEFAULT_RESERVE,gate_alpha=.05,gate_seed=73):
    unlimited=total_seconds is None
    if not unlimited and (type(total_seconds) not in (int,float) or not math.isfinite(total_seconds) or not 60<=total_seconds<=3600):
        raise ValueError('explicit total allocation of at most one hour required')
    if type(block) is not int or block<2 or block%2:raise ValueError('paired loop block required')
    root,parent=Path(root).resolve(),Path(parent).resolve();checkout=Path(__file__).resolve().parents[1]
    with parent_lock(parent):
        clock_wrapper=json.loads((parent/'execution.json').read_text());clock_body=clock_wrapper['body'];clock=clock_body['clock']
        if clock_wrapper['sha256']!=canonical_sha(clock_body) or clock['inflight_since'] is not None:
            raise ValueError('parent must be stopped with a verified closed execution clock')
        if (parent/'handoff.json').exists():raise ValueError('parent allocation has already been handed off')
        compatible=compatible_sources(parent_checkout,checkout)
        source=compatible['parent_source_sha256']
        checkpoint=parent/'resume'/f'{clock["last_checkpoint"]["records"]:012d}'
        seal=CheckpointStore.verify(checkpoint)
        if digest(checkpoint/'manifest.json')!=clock['last_checkpoint']['manifest_sha256'] or seal['metadata']['source_sha256']!=source:
            raise ValueError('parent checkpoint/source binding differs')
        data=torch.load(checkpoint/'runtime.pt',map_location='cpu',weights_only=True)
        if source!=source_digest() and data['backend']['counters']['supervised_updates']:
            raise ValueError('cross-version handoff is limited to pre-average learning; old deployed archives must remain valid')
        parent_manifest=json.loads((parent/'run.json').read_text())
        parent_init=parent_manifest.get('initialization',{})
        prior=0. if parent_init.get('clock_includes_parent') else parent_init.get('prior_used_seconds',0.)
        prior=math.ceil(prior+clock['observed_seconds']+clock['recovery_charge_seconds']+10.)
        remaining=None if unlimited else total_seconds-prior
        if remaining is not None and remaining<30:raise ValueError('insufficient unspent allocation for a new loop stage')
        if type(records) is not int or records%2 or not seal['step']<records or (not unlimited and records>20000000):
            raise ValueError('loop archive segment must extend the parent record clock')
        GatePlan(1,alpha=gate_alpha,seed=gate_seed)
        published=json.loads(Path(publication).read_text())
        if published['sha256']!=canonical_sha(published['body']):raise ValueError('published champion checksum differs')
        old=json.loads((parent/'league/state.json').read_text())
        if old['sha256']!=canonical_sha(old['body']):raise ValueError('parent league checksum differs')
        if old['body']['champion']['policy_sha256']!=published['body']['champion']['policy_sha256']:
            raise ValueError('arena champion changed; explicitly reconcile the release baseline first')
        initialization={'kind':'verified-learner-handoff','parent_run':str(parent),'parent_checkout':str(Path(parent_checkout).resolve()),
            'parent_checkpoint':str(checkpoint),'parent_manifest_sha256':digest(checkpoint/'manifest.json'),
            'parent_clock_sha256':clock_wrapper['sha256'],'records':seal['step'],'compatibility':compatible,
            'prior_used_seconds':prior,'total_allocation_seconds':total_seconds}
        if unlimited:initialization['clock_includes_parent']=True
        body={'version':'graygo/learning-loop-handoff/1','initialization':initialization,'runtime':data['config'],
            'remaining_seconds':remaining,'records':records,'block':block,'publication':str(Path(publication).resolve()),
            'initial_champion':published['body']['champion'],'history':old['body']['history'],
            'promotion_policy':published['body'].get('promotion_policy',HEAD_TO_HEAD_HISTORY),
            'release_selection':published['body']['champion'].get('selection'),
            'gate_workers':old['body'].get('contract',{}).get('gate_workers',1),
            'gate_looks':old['body'].get('contract',{}).get('gate_looks',[500,2000]),
            'parent_evaluation_seconds':old['body']['evaluation_seconds'],
            'adaptive_gate_schedule':unlimited,
            'parent_adaptive_scheduler':deepcopy(old['body'].get('adaptive_scheduler')),
            'prior_promotions':old['body'].get('prior_promotions',0)+sum(r['promoted'] for r in old['body']['rounds']),
            'gate_alpha':gate_alpha,'gate_seed':gate_seed,
            'checkout':str(checkout),'source_sha256':source_digest(),
            'tools':{name:digest(Path(__file__).with_name(name)) for name in
                     ('learning_loop.py','champion_loop.py','champion_gate.py','champion_screen.py','legacy_policy.py','training_comparison.py','adaptive_gate_schedule.py','policy_selection.py','parallel_gate.py')},
            'formal_training_or_promotion_grant':False,'experimental':True}
        root.mkdir(parents=True,exist_ok=False)
        write_report(root/'plan.json',{'body':body,'sha256':canonical_sha(body)},reserve_bytes=reserve_bytes)
        write_report(root/'status.json',{'status':'prepared','parent_records':seal['step'],'remaining_seconds':remaining},reserve_bytes=reserve_bytes)
        # A single parent can transfer its remaining allocation only once.
        with (parent/'handoff.json').open('x') as stream:
            json.dump({'destination':str(root),'plan_sha256':canonical_sha(body),'remaining_seconds':remaining},stream)
            stream.flush();os.fsync(stream.fileno())
        return body


def initialize(executor,loop,body):
    init=body['initialization'];checkpoint=Path(init['parent_checkpoint'])
    if digest(checkpoint/'manifest.json')!=init['parent_manifest_sha256']:raise ValueError('parent checkpoint changed')
    if executor.runtime.records==0:
        executor.runtime.restore(checkpoint,compatible_source_sha256=init['compatibility']['parent_source_sha256'])
        if executor.runtime.records!=init['records']:raise ValueError('parent learner clock differs')
        executor._checkpoint()
    elif executor.runtime.records<init['records']:raise ValueError('child rolled back behind its imported parent')
    if loop.state['completed_records']<init['records']:
        loop.state['completed_records']=init['records'];loop.state['history']=deepcopy(body['history'])
        loop.state['evaluation_seconds']=body.get('parent_evaluation_seconds',0.)
        loop.state['prior_promotions']=body.get('prior_promotions',0)
        if loop.plan.adaptive_schedule:
            from tools import adaptive_gate_schedule as adaptive
            loop.state['adaptive_scheduler']=(deepcopy(body['parent_adaptive_scheduler'])
                if body.get('parent_adaptive_scheduler') is not None else adaptive.initial_state(loop._learning_seconds()))
        loop._save();loop._publish()


def worker(root,*,stop_state=None,reserve_bytes=DEFAULT_RESERVE):
    root=Path(root).resolve();wrapper=json.loads((root/'plan.json').read_text());body=wrapper['body']
    if wrapper['sha256']!=canonical_sha(body) or body['source_sha256']!=source_digest():raise ValueError('loop plan/source changed')
    for name,sha in body['tools'].items():
        if digest(Path(__file__).with_name(name))!=sha:raise ValueError('loop implementation changed')
    init=body['initialization'];parent=Path(init['parent_run'])
    with parent_lock(parent):
        reservation=json.loads((parent/'handoff.json').read_text())
        if reservation['destination']!=str(root) or reservation['plan_sha256']!=wrapper['sha256']:raise ValueError('parent allocation belongs to another child')
        if json.loads((parent/'execution.json').read_text())['sha256']!=init['parent_clock_sha256']:raise ValueError('parent executed after handoff')
        if compatible_sources(init['parent_checkout'],body['checkout'])!=init['compatibility']:raise ValueError('parent source compatibility changed')
        values=dict(body['runtime']);values['rules']=Rules(**values['rules']);cfg=BoardRuntimeConfig(**values)
        start=init['records'];ends=tuple(sorted({*range((start//body['block']+1)*body['block'],body['records']+1,body['block']),body['records']}))
        milestones=tuple(sorted({0,start,*ends,*[n for n in (4500000,4750000,5000000,7000000) if n<=body['records']]}))
        journal=BoardJournalPlan('loop',body['records'],milestones)
        execution=BoardExecutionPlan(body['remaining_seconds'],checkpoint_every=body['block'])
        stopped=[] if stop_state is None else stop_state
        for kind in (signal.SIGTERM,signal.SIGINT):signal.signal(kind,lambda *unused:stopped.append(True))
        resume=(root/'learner/execution.json').exists()
        write_report(root/'status.json',{'status':'running','parent_records':start,'publication':body['publication'],
                                      'remaining_allocation_seconds':body['remaining_seconds'],'phase':'learning; average starts at 5M',
                                      'unlimited':body['remaining_seconds'] is None,
                                      'loop_enabled':True,'gatekeeping_enabled':True,'automatic_publication_enabled':True},reserve_bytes=reserve_bytes)
        inherited=init['prior_used_seconds'] if body['remaining_seconds'] is None and not resume else 0.
        with BoardExecutor(root/'learner',cfg,journal,execution,resume=resume,initialization=init,
                           prior_setup_seconds=inherited,reserve_bytes=reserve_bytes) as executor:
            promotion=body.get('promotion_policy',HEAD_TO_HEAD_HISTORY)
            gate=IndependentGate(GatePlan(len(ends),looks=tuple(body.get('gate_looks',[500,2000])),alpha=body.get('gate_alpha',.05),seed=body.get('gate_seed',73),
                pool_gain_policy=promotion,improvement=.03 if promotion==LEGACY_POOL_GAIN else 0.),reserve_bytes=reserve_bytes)
            if body.get('gate_workers',1)>1:gate=ParallelGate(gate.plan,workers=body['gate_workers'],reserve_bytes=reserve_bytes)
            gate_seconds=3600. if body['remaining_seconds'] is None else min(360.,body['remaining_seconds']*.1)
            plan=LoopPlan(ends,gate_seconds=gate_seconds,continue_after_evaluation_budget=True,
                adaptive_schedule=body.get('adaptive_gate_schedule',False),
                release_temperature=(body.get('release_selection') or {}).get('temperature'),
                release_placement_only=(body.get('release_selection') or {}).get('placement_only',True))
            with ChampionLoop(executor,plan,gate,initial_champion=body['initial_champion'],resume=resume,
                publish_manifest=body['publication'],screen=IndependentScreen(ScreenPlan(),reserve_bytes=reserve_bytes)) as loop:
                initialize(executor,loop,body)
                result=loop.run(stop_requested=lambda:bool(stopped))
                result.update(parent_records=start,total_records=executor.runtime.records,
                    counters=vars(executor.runtime.backend.counters),loop_enabled=True)
                write_report(root/'status.json',result,reserve_bytes=reserve_bytes)
                return result


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    mode.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--parent-run');parser.add_argument('--parent-checkout')
    parser.add_argument('--publish-manifest');parser.add_argument('--total-seconds',type=float,default=3600)
    parser.add_argument('--unlimited',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        if None in (args.parent_run,args.parent_checkout,args.publish_manifest):parser.error('parent run, frozen source and publication are required')
        result=prepare(args.run_dir,args.parent_run,args.parent_checkout,args.publish_manifest,
                       total_seconds=None if args.unlimited else args.total_seconds)
    elif args.worker:result=worker(args.run_dir)
    else:
        root=Path(args.run_dir).resolve();wrapper=json.loads((root/'plan.json').read_text());body=wrapper['body']
        if wrapper['sha256']!=canonical_sha(body):raise ValueError('loop plan changed')
        with (root/'launch.json').open('x') as stream:json.dump({'started_at':time.time(),'remaining_seconds':body['remaining_seconds']},stream)
        def terminate(*unused):raise SystemExit('loop launcher stopped')
        for kind in (signal.SIGTERM,signal.SIGINT):signal.signal(kind,terminate)
        if body['remaining_seconds'] is None:
            print(json.dumps(worker(root),sort_keys=True));return
        result=run_limited([sys.executable,'-u','-m','tools.learning_loop','--worker','--run-dir',str(root)],
                           cwd=body['checkout'],log=root/'worker.log',seconds=body['remaining_seconds'])
        write_report(root/'supervision.json',result)
        status=json.loads((root/'status.json').read_text())
        if status['status']=='running' or (result['soft_stop'] and status['status'] in ('interrupted','budget_limited')):
            status.update(status='budget_finished' if result['soft_stop'] else 'failed',supervision=result)
            write_report(root/'status.json',status)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()

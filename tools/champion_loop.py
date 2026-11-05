"""Persistent NFSP learner -> frozen candidate -> independent gate -> release.

The public command runs bounded diagnostics. Formal training still needs the
project entry verifier and existing calibrated protected-promotion authority.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
import argparse
import fcntl
import io
import json
import math
import os
from pathlib import Path
import shutil
import time
import torch
from graygo.board_execution import BoardExecutor,BoardExecutionPlan
from graygo.board_journal import BoardJournalPlan
from graygo.board_runtime import BoardRuntimeConfig
from graygo.confidence import canonical_sha
from graygo.engine import Rules
from graygo.evaluation import BaselinePolicy,FrozenNeuralPolicy
from graygo.manifest import source_digest,write_report
from graygo.models import ModelConfig,PolicyNet
from graygo.storage import CheckpointStore,DEFAULT_RESERVE,digest as _digest,DiskBudgetError
from tools.champion_gate import GatePlan,IndependentGate
from tools.champion_screen import IndependentScreen,ScreenPlan
from tools.legacy_policy import LegacyPolicy
from tools.nfsp_resources import require
from tools.task_pipeline_parity import state_sha

def digest(path):return _digest(Path(path))

VERSION='graygo/persistent-champion-loop/1'
RELEASE_VERSION='graygo/champion-release/1'


@dataclass(frozen=True)
class LoopPlan:
    round_ends:tuple
    history_size:int=3
    gate_seconds:float=600.
    evaluation_fraction:float=.1
    continue_after_evaluation_budget:bool=False

    def __post_init__(self):
        object.__setattr__(self,'round_ends',tuple(self.round_ends))
        require(self.round_ends and self.round_ends==tuple(sorted(set(self.round_ends)))
                and all(type(n) is int and n>0 and n%2==0 for n in self.round_ends),'fixed increasing even training endpoints required')
        require(type(self.history_size) is int and 0<=self.history_size<=8,'bounded historical pool required')
        require(isinstance(self.gate_seconds,(int,float)) and 0<self.gate_seconds<=3600,'explicit bounded per-round gate time required')
        require(type(self.evaluation_fraction) in (int,float) and math.isfinite(self.evaluation_fraction)
                and 0<=self.evaluation_fraction<=.1,'evaluation may reserve at most ten percent of the run ceiling')
        require(type(self.continue_after_evaluation_budget) is bool,'explicit evaluation exhaustion policy required')


def load(path):return json.loads(Path(path).read_text())


def policy(descriptor):
    kind=descriptor['kind']
    if kind=='archive':
        if 'bundle' in descriptor:require(digest(descriptor['bundle'])==descriptor['bundle_sha256'],'published average bundle changed')
        result=FrozenNeuralPolicy.from_archive(descriptor['archive'],label=descriptor['label'])
    elif kind=='legacy':
        require(digest(descriptor['checkpoint'])==descriptor['checkpoint_sha256'],'legacy snapshot changed')
        result=LegacyPolicy(descriptor['repo'],descriptor['checkpoint'],label=descriptor['label'])
    elif kind=='bundle':
        require(digest(descriptor['bundle'])==descriptor['bundle_sha256'],'imported bundle changed')
        data=torch.load(descriptor['bundle'],weights_only=True,map_location='cpu')
        require(data['format_version']=='graygo/arena-nfsp-average/1' and data['head']=='average','supported average-policy bundle required')
        with torch.random.fork_rng(devices=[]),torch.device('cpu'):model=PolicyNet(ModelConfig(**data['model_config']))
        model.load_state_dict(data['state_dict'],strict=True)
        result=FrozenNeuralPolicy(model,representation='average',label=descriptor['label'],
            origin={'rules':descriptor['rules'],'records':data['training_records'],'imported_bundle_sha256':descriptor['bundle_sha256']})
    else:raise ValueError('unknown frozen champion representation')
    if 'policy_sha256' in descriptor:require(result.sha256==descriptor['policy_sha256'],'frozen policy identity changed')
    return result


def legacy_descriptor(repo,checkpoint,rules):
    result={'kind':'legacy','repo':str(Path(repo).resolve()),'checkpoint':str(Path(checkpoint).resolve()),
            'checkpoint_sha256':digest(checkpoint),'rules':asdict(rules),'label':'old-champion'}
    frozen=policy(result);require(frozen.size==rules.size,'legacy board shape differs')
    return dict(result,policy_sha256=frozen.sha256,weights_sha256=frozen.identity['weights_sha256'])


def bundle_descriptor(bundle,rules):
    data=torch.load(bundle,weights_only=True,map_location='cpu')
    require(data['board_size']==rules.size,'bundle board shape differs')
    # This legacy demo bundle includes the exact original private checkpoint.
    original=Path(data['source_checkpoint'])
    require(digest(original/'manifest.json')==data['source_checkpoint_manifest_sha256'],'imported model origin changed')
    CheckpointStore.verify(original)
    runtime=torch.load(original/'runtime.pt',weights_only=True,map_location='cpu')
    require(runtime['config']['rules']==asdict(rules),'bundle training rules differ')
    result={'kind':'bundle','bundle':str(Path(bundle).resolve()),'bundle_sha256':digest(bundle),
            'rules':asdict(rules),'label':'imported-average','records':data['training_records']}
    frozen=policy(result)
    return dict(result,policy_sha256=frozen.sha256,weights_sha256=frozen.identity['weights_sha256'])


class ChampionLoop:
    def __init__(self,executor,plan,gate,*,initial_champion=None,resume=False,entry_verifier=None,publish_manifest=None,screen=None):
        require(executor.active,'hold the persistent learner executor lock')
        require(plan.round_ends[-1]==executor.journal_plan.records and set(plan.round_ends)<=set(executor.journal_plan.milestones),'each fixed loop endpoint must be permanently archived')
        require(gate.plan.rounds==len(plan.round_ends),'loop and confidence-round inventory differ')
        self.executor=executor;self.plan=plan;self.gate=gate;self.screen=screen;self.root=executor.root/'league'
        self.root.mkdir(exist_ok=True);self.reserve=executor.reserve
        self.publication=Path(publish_manifest).resolve() if publish_manifest else self.root/'champion.json'
        self.publication_lock=None
        self.diagnostic=executor.journal_plan.scope in ('diagnostic','practice','loop')
        self.contract={'version':VERSION,'source_sha256':source_digest(),'implementation_sha256':digest(__file__),
            'training_run_sha256':executor.journal.run_sha256,'execution_contract_sha256':canonical_sha(executor.contract),
            'loop_plan':asdict(plan),'gate_sha256':gate.sha256,'screen_sha256':None if screen is None else screen.sha256,
            'scope':executor.journal_plan.scope,'publication':str(self.publication)}
        self.contract=json.loads(json.dumps(self.contract))
        self.lock=(self.root/'.loop.lock').open('a+b');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if self.publication!=self.root/'champion.json':
                self.publication.parent.mkdir(parents=True,exist_ok=True)
                self.publication_lock=self.publication.with_name('.'+self.publication.name+'.publisher.lock').open('a+b')
                fcntl.flock(self.publication_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not self.diagnostic:
                authority=None if entry_verifier is None else entry_verifier(executor,self.contract)
                require(isinstance(authority,dict) and authority.get('training_authorized') is True
                    and authority.get('execution_contract_sha256')==self.contract['execution_contract_sha256']
                    and gate.protected_authorizer is not None,'formal loop needs verified project entry and calibrated protected promotion')
            path=self.root/'state.json'
            if resume:
                wrapper=load(path);require(wrapper['sha256']==canonical_sha(wrapper['body']),'loop state checksum differs')
                self.state=wrapper['body'];require(self.state['contract']==self.contract,'loop source/plans changed on resume')
                require(executor.runtime.records>=self.state['completed_records'],'learner rolled back behind committed loop state')
                if initial_champion is not None:require(initial_champion['policy_sha256']==self.state['bootstrap']['policy_sha256'],'bootstrap changed on resume')
                if self.publication.exists():
                    published=load(self.publication)
                    require(published['sha256']==canonical_sha(published['body']),'publication checksum differs')
                    require(published['body'].get('loop_contract_sha256')==canonical_sha(self.contract),'another loop owns the current publication; stale resume is rejected')
            else:
                require(not path.exists() and executor.runtime.records==0,'new loop requires a fresh learner; use explicit resume')
                if initial_champion is None:
                    archive=executor.journal.capture()
                    initial_champion=self._candidate(archive,'initial')
                initial_champion=self._export(initial_champion)
                require(initial_champion['rules']==asdict(executor.config.rules),'champion and learner game rules differ')
                policy(initial_champion).assert_frozen()
                if self.publication.exists():
                    old=load(self.publication)
                    require(old['sha256']==canonical_sha(old['body']),'existing publication checksum differs')
                    require(old['body']['champion']['policy_sha256']==initial_champion['policy_sha256'],'new run may not replace a different current champion')
                    require(not self.diagnostic or old['body']['experimental'],'diagnostic loop cannot overwrite a formal release')
                self.state={'contract':self.contract,'bootstrap':deepcopy(initial_champion),'completed_records':0,'round':0,'champion':initial_champion,
                    'history':[],'rounds':[],'pending':None,'evaluation_seconds':0.,'evaluation_inflight':None}
                self._save()
            # Charge an interrupted evaluation conservatively against the same
            # quota, using the executor's durable recovery clock, never a reset.
            if self.state['evaluation_inflight'] is not None:self._finish_evaluation()
            policy(self.state['champion']).assert_frozen()
            self._publish()
        except BaseException:
            if self.publication_lock is not None:self.publication_lock.close()
            self.lock.close();raise

    def __enter__(self):return self
    def __exit__(self,*args):
        if self.publication_lock is not None:self.publication_lock.close()
        self.lock.close()

    def _save(self):
        write_report(self.root/'state.json',{'body':self.state,'sha256':canonical_sha(self.state)},reserve_bytes=self.reserve)

    def _publish(self):
        body={'version':RELEASE_VERSION,'champion':self.state['champion'],'completed_round':self.state['round'],
              'loop_state_sha256':canonical_sha(self.state),'loop_contract_sha256':canonical_sha(self.contract),'experimental':self.diagnostic,'formal_training_or_promotion_grant':False}
        write_report(self.root/'champion.json',{'body':body,'sha256':canonical_sha(body)},reserve_bytes=self.reserve)
        if self.publication!=self.root/'champion.json':write_report(self.publication,{'body':body,'sha256':canonical_sha(body)},reserve_bytes=self.reserve)

    def _candidate(self,archive,label):
        frozen=FrozenNeuralPolicy.from_archive(archive,label=label)
        origin=frozen.identity['origin']
        return {'kind':'archive','archive':str(Path(archive).resolve()),'label':label,'rules':origin['rules'],
                'records':origin['records'],'policy_sha256':frozen.sha256,'weights_sha256':frozen.identity['weights_sha256']}

    def _export(self,descriptor):
        if descriptor['kind']!='archive':return descriptor
        folder=self.root/'releases'/descriptor['policy_sha256'];folder.mkdir(parents=True,exist_ok=True)
        target=folder/'policy.pt'
        source=Path(descriptor['archive']);CheckpointStore.verify(source)
        data=torch.load(source/'models.pt',weights_only=True,map_location='cpu');cfg=data['config']
        blob={'format_version':'graygo/arena-nfsp-average/1','head':'average','encoding_version':'graygo/encoding/1',
            'board_size':descriptor['rules']['size'],'rules':descriptor['rules'],'training_records':descriptor['records'],
            'model_config':{k:cfg[k] for k in ('width','blocks','norm_groups')},'state_dict':data['average'],
            'strength_validated':False,'source_archive_sha256':digest(source/'manifest.json')}
        if target.exists():
            stored=torch.load(target,weights_only=True,map_location='cpu')
            require(state_sha(stored)==state_sha(blob),'existing immutable release payload differs')
        else:
            memory=io.BytesIO();torch.save(blob,memory);raw=memory.getvalue()
            if shutil.disk_usage(folder).free-len(raw)-65536<self.reserve:raise DiskBudgetError('release export would violate reserve')
            temporary=folder/'.policy.pt.tmp'
            with temporary.open('wb') as stream:
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
            temporary.replace(target)
        return dict(descriptor,bundle=str(target),bundle_sha256=digest(target))

    def _pool(self,history):
        result={'random':BaselinePolicy('random'),'tactical':BaselinePolicy('tactical')}
        for i,descriptor in enumerate(history[-self.plan.history_size:] if self.plan.history_size else []):
            result['historical-'+str(i)]=policy(descriptor)
        return result

    def _evaluation_remaining(self):
        spent=self.state['evaluation_seconds']
        if self.state['evaluation_inflight'] is not None:
            spent+=max(0.,self.executor._spent()-self.state['evaluation_inflight'])
        return max(0.,min(self.executor.plan.time_ceiling_seconds-self.executor._spent(),
            self.plan.evaluation_fraction*self.executor.plan.time_ceiling_seconds-spent))

    def _finish_evaluation(self):
        self.state['evaluation_seconds']+=max(0.,self.executor._spent()-self.state['evaluation_inflight'])
        self.state['evaluation_inflight']=None;self._save()

    def _evaluate(self,archive,candidate,pending,stop_requested):
        rejected={'status':'complete','numeric_pass':False,'promotion_permitted':False,
                  'experimental_promotion_permitted':False}
        if self._evaluation_remaining()<=0:return dict(rejected,reason='evaluation_budget_exhausted')
        self.state['evaluation_inflight']=self.executor._spent();self._save()
        try:
            average_ready=self.executor.runtime.backend.average_ready
            if self.screen is not None:
                pending['status']='screening';self._save()
                pending['screen']=self.screen.evaluate(archive,average_ready=average_ready,
                    root=self.root/'screens'/f'{pending["index"]:04d}',time_cap_seconds=self._evaluation_remaining(),
                    stop_requested=lambda:stop_requested() or self._evaluation_remaining()<=0)
                self._save()
                if stop_requested():return dict(rejected,status='interrupted')
                if not average_ready:return dict(rejected,reason='average_not_ready')
                if not pending['screen']['average_eligible']:return dict(rejected,reason='screen_not_cleared')
            elif not average_ready:return dict(rejected,reason='average_not_ready')
            if candidate['weights_sha256']==pending['champion']['weights_sha256']:
                return dict(rejected,reason='unchanged_average_weights')
            remaining=self._evaluation_remaining()
            if remaining<=0:return dict(rejected,reason='evaluation_budget_exhausted')
            pending['status']='evaluating';pending['candidate']=candidate;self._save()
            decision=self.gate.evaluate(policy(candidate),policy(pending['champion']),self._pool(pending['history']),
                self.root/'evaluations'/f'{pending["index"]:04d}',round_index=pending['index'],
                time_cap_seconds=min(remaining,self.plan.gate_seconds),
                stop_requested=lambda:stop_requested() or self._evaluation_remaining()<=0)
            if (decision['status'] in ('budget_limited','interrupted') and not stop_requested()
                    and self.plan.continue_after_evaluation_budget):
                return dict(rejected,reason='evaluation_budget_exhausted',evaluation_status=decision['status'])
            return decision
        finally:self._finish_evaluation()

    def run(self,*,until_round=None,stop_requested=lambda:False):
        limit=len(self.plan.round_ends) if until_round is None else until_round
        require(type(limit) is int and self.state['round']<=limit<=len(self.plan.round_ends),'original bounded round endpoint required')
        status='complete' if limit==len(self.plan.round_ends) else 'round_boundary'
        while self.state['round']<limit:
            if stop_requested():status='interrupted';break
            index=self.state['round'];target=self.plan.round_ends[index]
            require(self.executor.runtime.records<=target,'learner advanced outside its fixed loop round')
            if self.state['pending'] is None:
                self.state['pending']={'index':index,'target':target,'champion':deepcopy(self.state['champion']),
                    'history':deepcopy(self.state['history']),'status':'training'}
                self._save()
            pending=self.state['pending'];require(pending['target']==target and pending['index']==index,'pending round changed')
            trained=self.executor.run(target,stop_requested=stop_requested)
            if (trained['records']!=target or trained['status'] not in ('complete','checkpoint_boundary')
                    or trained['last_durable_checkpoint']['records']!=target):
                status=trained['status'];break
            archive=self.executor.root/'weights'/f'{target:012d}'
            candidate=self._export(self._candidate(archive,'candidate-'+str(target)))
            decision=self._evaluate(archive,candidate,pending,stop_requested)
            if decision['status']!='complete':status=decision['status'];break
            promoted=bool(decision.get('experimental_promotion_permitted') if self.diagnostic else decision.get('promotion_permitted'))
            if promoted:
                self.state['history'].append(pending['champion']);self.state['champion']=candidate
            self.state['rounds'].append({'index':index,'records':target,'candidate':candidate,'incumbent':pending['champion'],
                'decision':decision,'screen':pending.get('screen'),'promoted':promoted,'learner_reinitialized':False,'buffers_flushed':False})
            self.state.update(round=index+1,completed_records=target,pending=None)
            self._save();self._publish()
        return {'version':VERSION,'status':status,'rounds':self.state['round'],'learner_records':self.executor.runtime.records,
                'evaluation_seconds':self.state['evaluation_seconds'],
                'evaluation_ceiling_seconds':self.plan.evaluation_fraction*self.executor.plan.time_ceiling_seconds,
                'champion':deepcopy(self.state['champion']),'experimental':self.diagnostic,
                'champion_manifest':str(self.publication),'training_state_preserved':True,
                'a2_complete':False,'formal_scope_approved':False}


def parse_args(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--diagnostic',action='store_true');mode.add_argument('--practice',action='store_true');mode.add_argument('--loop',action='store_true')
    parser.add_argument('--records',type=int);parser.add_argument('--block-records',type=int,default=65536)
    parser.add_argument('--time-cap-seconds',type=float,required=True);parser.add_argument('--gate-seconds',type=float,default=60)
    parser.add_argument('--screen-games',type=int,default=20);parser.add_argument('--screen-seconds',type=float,default=30)
    parser.add_argument('--evaluation-fraction',type=float,default=.1)
    parser.add_argument('--gate-looks',type=int,nargs='+',default=[500,2000]);parser.add_argument('--device',default='cpu')
    parser.add_argument('--seed',type=int,default=17);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--reservoir-open',type=int,default=5000000);parser.add_argument('--handoff-end',type=int,default=7000000)
    parser.add_argument('--early-opening-diagnostic',action='store_true')
    parser.add_argument('--incumbent-bundle');parser.add_argument('--old-repo');parser.add_argument('--old-checkpoint')
    parser.add_argument('--incumbent-release');parser.add_argument('--publish-manifest')
    args=parser.parse_args(argv)
    if args.records is None:args.records=20000000 if args.loop else 5000000 if args.practice else 32768
    cap=20000000 if args.loop else 5000000 if args.practice else 65536
    require(type(args.records) is int and 0<args.records<=cap and args.records%2==0,'bounded even diagnostic/practice record horizon required')
    require(math.isfinite(args.time_cap_seconds) and 0<args.time_cap_seconds<=3600,'explicit run budget of at most one hour required')
    require(args.block_records>0 and args.block_records%2==0,'fixed paired block inventory required')
    if (args.reservoir_open,args.handoff_end)!=(5000000,7000000):
        require(args.diagnostic and args.early_opening_diagnostic,'changed averaging schedule requires --diagnostic --early-opening-diagnostic')
    ScreenPlan(args.screen_games,args.screen_seconds)
    LoopPlan((args.records,),gate_seconds=args.gate_seconds,evaluation_fraction=args.evaluation_fraction)
    return args


def main():
    args=parse_args()
    cfg=BoardRuntimeConfig(seed=args.seed,device=args.device,q_reuse=4,reservoir_open=args.reservoir_open,handoff_end=args.handoff_end)
    ends=tuple(sorted({*range(args.block_records,args.records+1,args.block_records),args.records}))
    milestones=tuple(sorted({0,*ends,*[n for n in (4500000,4750000,args.reservoir_open,args.handoff_end) if 0<=n<=args.records]}))
    journal=BoardJournalPlan('loop' if args.loop else 'practice' if args.practice else 'diagnostic',args.records,milestones)
    execution=BoardExecutionPlan(args.time_cap_seconds,checkpoint_every=args.block_records)
    gate=IndependentGate(GatePlan(len(ends),tuple(args.gate_looks)),reserve_bytes=DEFAULT_RESERVE)
    initial=None
    if args.incumbent_release:
        imported=load(args.incumbent_release);require(imported['sha256']==canonical_sha(imported['body']),'imported release checksum differs')
        initial=imported['body']['champion']
    if args.incumbent_bundle:initial=bundle_descriptor(args.incumbent_bundle,cfg.rules)
    if args.old_checkpoint:
        require(args.old_repo is not None and initial is None,'one explicit bootstrap champion required')
        initial=legacy_descriptor(args.old_repo,args.old_checkpoint,cfg.rules)
    import signal
    stopped=[]
    for kind in (signal.SIGTERM,signal.SIGINT):signal.signal(kind,lambda number,frame:stopped.append(number))
    with BoardExecutor(args.run_dir,cfg,journal,execution,resume=args.resume) as executor:
        if cfg.reservoir_open==0 and not executor.journal.decision_path.exists():executor.journal.diagnostic_opening('pass')
        screen=IndependentScreen(ScreenPlan(args.screen_games,args.screen_seconds))
        plan=LoopPlan(ends,gate_seconds=args.gate_seconds,evaluation_fraction=args.evaluation_fraction,continue_after_evaluation_budget=True)
        with ChampionLoop(executor,plan,gate,initial_champion=initial,resume=args.resume,publish_manifest=args.publish_manifest,screen=screen) as loop:
            print(json.dumps(loop.run(stop_requested=lambda:bool(stopped)),sort_keys=True))


if __name__=='__main__':main()

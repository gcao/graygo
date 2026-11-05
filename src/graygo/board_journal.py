"""Permanent board milestones and recoverable fixed-opening control state.

This journal records lifecycle evidence. It neither issues A2/resource approval
nor advances a learner; the production scope/cohort controller must own that
separate authority and call before_advance before collecting any records.
"""
from dataclasses import dataclass,asdict
from contextlib import ExitStack
import io
import json
from pathlib import Path
import torch
from .confidence import canonical_sha
from .engine import Rules
from .manifest import make_manifest,source_digest,write_report
from .pilot_gates import PilotProtocol,PRE,POST,DEVELOPMENT_SEEDS,CONFIRMATION_SEEDS
from .storage import CheckpointStore,DEFAULT_RESERVE,digest

JOURNAL_VERSION='graygo/board-milestone-journal/1'
DECISION_VERSION='graygo/board-opening-decision/1'


class MilestonePending(RuntimeError):pass
class OpeningPending(RuntimeError):pass
class OpeningNoGo(RuntimeError):pass


@dataclass(frozen=True)
class BoardJournalPlan:
    scope:str
    records:int
    milestones:tuple
    evaluation_seed:int=41

    def __post_init__(self):
        object.__setattr__(self,'milestones',tuple(self.milestones))
        if self.scope not in ('diagnostic','practice','loop','development','confirmation') or type(self.records) is not int or self.records<2 or self.records%2:
            raise ValueError('declared scope and even record cap required')
        if not self.milestones or self.milestones!=tuple(sorted(set(self.milestones))) or any(type(n) is not int or n<0 or n%2 or n>self.records for n in self.milestones) or self.milestones[-1]!=self.records:
            raise ValueError('fixed increasing paired milestones including the endpoint required')
        if type(self.evaluation_seed) is not int or self.evaluation_seed<0:raise ValueError('independent evaluation seed required')
        if self.scope=='diagnostic':
            if self.records>65536:raise ValueError('diagnostic journal cannot allocate a production campaign')
        elif self.scope=='practice':
            if self.records>5000000:raise ValueError('practice is a bounded Q-learning prefix through the fixed reservoir opening')
        elif self.scope=='loop':
            pass # Sustained runs advance through successive fixed archive segments.
        else:
            cap=20000000 if self.scope=='development' else 80000000
            if self.records!=cap or not set((*PRE,*POST,cap)).issubset(self.milestones):raise ValueError('fixed development/confirmation archive windows and record horizon required')
    def to_dict(self):return {'version':JOURNAL_VERSION,'scope':self.scope,'records':self.records,'milestones':list(self.milestones),'evaluation_seed':self.evaluation_seed}
    @property
    def sha256(self):return canonical_sha(self.to_dict())


def _equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
    return a==b


class BoardMilestoneJournal:
    def __init__(self,root,runtime,plan,*,resume=False,reserve_bytes=DEFAULT_RESERVE,initialization=None):
        self.root=Path(root).resolve();self.runtime=runtime;self.plan=plan;self.reserve=reserve_bytes
        if type(reserve_bytes) is not int or reserve_bytes<0:raise ValueError('nonnegative journal reserve required')
        c=runtime.config
        if c.reservoir_open<=plan.records and c.reservoir_open not in plan.milestones:raise ValueError('the fixed opening must itself be an archived boundary')
        if plan.scope in ('practice','loop') and (c.reservoir_open,c.handoff_end)!=(5000000,7000000):
            raise ValueError('practice/loop preserves the primary averaging schedule')
        if plan.scope in ('development','confirmation'):
            cohort=DEVELOPMENT_SEEDS if plan.scope=='development' else CONFIRMATION_SEEDS
            if c.seed not in cohort or c.rules.size!=9 or c.rules.max_rounds!=200 or (plan.scope=='development' and c.rules.collision!='gray') or (c.width,c.blocks,c.norm_groups)!=(64,8,8) or (c.batch,c.learning_start,c.capacity,c.target_every)!=(512,4096,500000,12800):raise ValueError('formal journal requires the declared board/cohort/learner configuration')
            if (c.learning_rate,c.max_grad_norm)!=(3e-4,.5):raise ValueError('formal optimizer protocol differs')
            starts=runtime.collector.starts.identity
            if plan.scope=='development' and starts.get('kind')!='official-empty':raise ValueError('primary development starts must be official empty')
            if plan.scope=='confirmation' and starts.get('kind') not in ('official-empty','independent-gray-exposure'):raise ValueError('declared confirmation start distribution required')
            if starts.get('kind')=='independent-gray-exposure' and (starts.get('split')!='train' or starts.get('mode')!='mixed' or starts.get('gray_probability')!=.5):raise ValueError('confirmation exposure training must use the frozen mixed training split')
            if c.reservoir_open!=5000000 or c.handoff_end!=7000000 or c.uniform_comparison or not c.augmentation or not c.deterministic:raise ValueError('formal primary schedule and deterministic augmented runtime required')
            if reserve_bytes<DEFAULT_RESERVE:raise ValueError('formal journal cannot reduce the twenty-GiB reserve')
        self.store=CheckpointStore(self.root,reserve_bytes=reserve_bytes);path=self.root/'run.json'
        config=c.plan()|{'scope':plan.scope,'records':plan.records,'journal_plan':plan.to_dict()}
        if resume:
            self.manifest=json.loads(path.read_text())
            if self.manifest['config']!=config or self.manifest['source_sha256']!=source_digest() or self.manifest['training_starts']!=runtime.collector.starts.identity:raise ValueError('journal source/config/start distribution changed')
            if self.manifest.get('initialization')!=initialization:raise ValueError('journal parent initialization changed')
        else:
            if path.exists():raise FileExistsError('board run manifest exists; use its original controller/resume')
            if runtime.records or runtime.replay.insertions or runtime.reservoir.seen:raise ValueError('new run manifest requires a fresh initialized runtime')
            self.manifest=make_manifest(seed=c.seed,command={'board_journal':plan.to_dict(),'runtime':c.plan()},rules=asdict(c.rules))
            self.manifest.update(config=config,training_starts=runtime.collector.starts.identity,
                scope_approval_verified=False,approval_scope='journal declaration only; project A2/resource controller must authorize actual training')
            if initialization is not None:self.manifest['initialization']=initialization
            write_report(path,self.manifest,reserve_bytes=reserve_bytes)
        self.run_sha256=digest(path);self.archived={};self.opening_sha256=None

    @property
    def records(self):return self.runtime.records
    @property
    def opening(self):return self.runtime.config.reservoir_open
    @property
    def decision_path(self):return self.root/'opening-decision.json'

    def _archive_path(self,record):return self.root/'weights'/f'{record:012d}'

    def _verify_current_archive(self,record):
        if self.records!=record:raise ValueError('current parameters cannot be substituted for another milestone')
        path=self._archive_path(record);sealed=self.store.verify(path)
        if sealed['step']!=record or sealed['metadata'].get('training_run_sha256')!=self.run_sha256 or sealed['metadata'].get('journal_plan_sha256')!=self.plan.sha256:raise ValueError('archive origin/clock/plan differs')
        weights=torch.load(path/'models.pt',weights_only=True,map_location='cpu')
        if weights['config']!=self.manifest['config'] or not _equal(weights['q'],self.runtime.backend.q.state_dict()) or not _equal(weights['average'],self.runtime.backend.average.state_dict()):raise ValueError('replayed milestone weights differ from the permanent archive')
        return digest(path/'manifest.json')

    def capture(self):
        record=self.records
        if record not in self.plan.milestones:raise ValueError('only declared exact milestones may be archived')
        path=self._archive_path(record)
        if path.exists():checksum=self._verify_current_archive(record)
        else:
            raw=io.BytesIO();torch.save({'config':self.manifest['config'],'q':self.runtime.backend.q.state_dict(),'average':self.runtime.backend.average.state_dict()},raw)
            payload=raw.getvalue()
            self.store.save(record,lambda p:(p/'models.pt').write_bytes(payload),kind='weights',
                metadata={'source_sha256':source_digest(),'training_run_sha256':self.run_sha256,'journal_plan_sha256':self.plan.sha256,
                          'native_schema':self.runtime.spec.to_dict(),'counters':asdict(self.runtime.backend.counters)},estimated_peak_bytes=len(payload)+65536)
            checksum=digest(path/'manifest.json')
        old=self.archived.get(str(record))
        if old is not None and old!=checksum:raise ValueError('permanent milestone changed')
        self.archived[str(record)]=checksum;return path

    def capture_if_due(self):
        if self.records in self.plan.milestones:return self.capture()
        return None

    def next_boundary(self):
        future=[n for n in (*self.plan.milestones,self.opening,self.runtime.config.handoff_end,self.plan.records) if self.records<n<=self.plan.records]
        return min(future) if future else self.plan.records

    def _decision(self):
        if not self.decision_path.exists():return None
        wrapper=json.loads(self.decision_path.read_text());body=wrapper.get('body')
        if wrapper.get('version')!=DECISION_VERSION or wrapper.get('sha256')!=canonical_sha(body):raise ValueError('opening decision checksum differs')
        if body['plan_sha256']!=self.plan.sha256 or body['training_run_sha256']!=self.run_sha256 or body['records']!=self.opening or body['seed']!=self.runtime.config.seed or body['status'] not in ('pass','no_go'):
            raise ValueError('opening decision source/clock/cohort differs')
        path=self._archive_path(self.opening);self.store.verify(path)
        if body['archive_manifest_sha256']!=digest(path/'manifest.json'):raise ValueError('opening decision checkpoint differs')
        if body['kind']=='diagnostic':
            if self.plan.scope!='diagnostic':raise ValueError('diagnostic opening is not formal evidence')
        elif body['kind']=='pilot':
            if self.plan.scope!='development':raise ValueError('confirmation seeds cannot be performance-filtered')
            protocol=PilotProtocol.from_dict(body['protocol']);window=protocol.window(body['reads'],body['seed'],window='pre')
            expected='pass' if window['status']=='pass' else 'no_go' if window['status']=='no_go' or body['budget_exhausted'] else 'incomplete'
            if expected!=body['status'] or window!=body['window']:raise ValueError('opening decision does not reproduce the fixed pre-window')
        else:raise ValueError('unknown opening decision kind')
        return body

    def _sync_decision(self):
        body=self._decision()
        if self.opening_sha256 is not None:
            if body is None or digest(self.decision_path)!=self.opening_sha256:raise ValueError('recorded opening event changed or disappeared')
            if body['status']=='no_go' and self.records>self.opening:raise ValueError('records advanced after an opening no-go')
        elif body is not None and self.records>=self.opening:
            if self.records!=self.opening:raise ValueError('post-opening checkpoint lacks its recorded opening event')
            self._verify_current_archive(self.opening)
            if self.runtime.reservoir.seen or self.runtime.reservoir.size or self.runtime.backend.counters.supervised_updates:raise ValueError('opening event cannot be applied to preexisting label history')
            self.opening_sha256=digest(self.decision_path)
        return body

    def label_insertion_allowed(self):
        if self.records<self.opening:return False
        if self.plan.scope in ('confirmation','loop'):return True
        body=self._sync_decision();return bool(body and body['status']=='pass')

    def before_advance(self,count):
        if type(count) is not int or count<2 or count%2:raise ValueError('paired collection count required')
        if self.records!=self.runtime.collector.records or self.records!=self.runtime.replay.insertions:raise ValueError('journal/runtime fresh clocks differ')
        if self.records in self.plan.milestones and str(self.records) not in self.archived:raise MilestonePending('archive the exact current milestone before collecting more')
        if self.records>=self.plan.records or self.records+count>self.next_boundary():raise MilestonePending('collection must stop at the next declared archive/opening boundary')
        if self.records>=self.opening and not self.label_insertion_allowed():
            body=self._decision()
            if body and body['status']=='no_go':raise OpeningNoGo('preserve the closed-reservoir opening checkpoint')
            raise OpeningPending('fixed opening is paused until its complete decision')

    def _write_decision(self,details):
        if self.records!=self.opening:raise ValueError('opening may be decided only at its original fixed boundary')
        archive=self.capture();checksum=self._verify_current_archive(self.opening)
        if self.runtime.reservoir.seen or self.runtime.reservoir.size or self.runtime.backend.counters.supervised_updates:raise ValueError('opening must preserve the original closed reservoir')
        body={'kind':details['kind'],'records':self.opening,'seed':self.runtime.config.seed,'status':details['status'],
              'plan_sha256':self.plan.sha256,'training_run_sha256':self.run_sha256,'archive_manifest_sha256':checksum,
              **{key:value for key,value in details.items() if key not in ('kind','status')}}
        if self.decision_path.exists():
            if self._decision()!=body:raise ValueError('final opening decision is immutable; no rolling window or later replacement')
        else:write_report(self.decision_path,{'version':DECISION_VERSION,'body':body,'sha256':canonical_sha(body)},reserve_bytes=self.reserve)
        self._sync_decision();return body

    def diagnostic_opening(self,status):
        if self.plan.scope!='diagnostic' or status not in ('pass','no_go'):raise ValueError('diagnostic control decision only')
        return self._write_decision({'kind':'diagnostic','status':status,'formal_evidence':False})

    def record_pilot_opening(self,protocol,evaluation_directories,*,budget_exhausted=False):
        from .cohort_evidence import read_milestone
        from .evaluation_evidence import evaluation_read_lock
        if self.plan.scope!='development' or self.records!=5000000:raise ValueError('formal pilot opening uses the original 5M development boundary')
        seed=self.runtime.config.seed
        if dict(protocol.training_runs).get(seed)!=self.run_sha256:raise ValueError('pilot registry is not this initialized training run')
        allowed={(record,opponent) for record in PRE for opponent in ('random','tactical')}
        if set(evaluation_directories)-allowed:raise ValueError('opening evidence must use only its original fixed pre-window')
        reads=[];cache={}
        # Public weight loading/evaluation must not consume any training RNG.
        devices=[self.runtime.backend.device.index or 0] if self.runtime.backend.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices),ExitStack() as locks:
            for path in evaluation_directories.values():locks.enter_context(evaluation_read_lock(path))
            for record in PRE:
                if str(record) not in self.archived:raise ValueError('pre-opening snapshot archive is missing')
                for opponent in ('random','tactical'):
                    key=(record,opponent)
                    if key not in evaluation_directories:continue
                    receipt,_=read_milestone(self.root,protocol,seed,record,'greedy_q',opponent,evaluation_directories[key],cache=cache)
                    if receipt is not None:reads.append(receipt)
            window=protocol.window(reads,seed,window='pre')
            if window['status']=='incomplete' and not budget_exhausted:
                progress={'records':self.records,'protocol_sha256':protocol.sha256,'window':window,'reads':reads,'status':'incomplete'}
                write_report(self.root/'opening-progress.json',progress,reserve_bytes=self.reserve);return progress
            status='pass' if window['status']=='pass' else 'no_go'
            # All scored inputs remain shared-locked until the immutable event
            # commits. A concurrent later look cannot race this decision.
            return self._write_decision({'kind':'pilot','status':status,'protocol':protocol.to_dict(),'reads':reads,'window':window,'budget_exhausted':bool(budget_exhausted)})

    def state_dict(self):
        self._sync_decision()
        return {'version':JOURNAL_VERSION,'plan_sha256':self.plan.sha256,'training_run_sha256':self.run_sha256,'records':self.records,
                'archived':dict(self.archived),'opening_sha256':self.opening_sha256}

    def checkpoint(self):
        self.capture_if_due();control=self.state_dict()
        try:latest=self.store.latest()
        except FileNotFoundError:latest=None
        if latest is not None and int(latest.name)==self.records:
            old=torch.load(latest/'runtime.pt',weights_only=True,map_location='cpu');current=self.runtime.state_dict()
            for state in (old,current):state.pop('timings',None)
            if not _equal(old,current):raise ValueError('same-clock runtime changed; immutable checkpoint cannot be overwritten')
            saved=json.loads((latest/'controller.json').read_text())
            before=dict(saved);after=dict(control);before.pop('opening_sha256');after.pop('opening_sha256')
            if before!=after or (saved['opening_sha256']!=control['opening_sha256'] and not (self.records==self.opening and saved['opening_sha256'] is None)):
                raise ValueError('same-clock controller state differs outside its fixed-opening overlay')
            return latest
        return self.runtime.save(self.root,reserve_bytes=self.reserve,controller_state=control)

    def restore(self,path=None):
        path=self.store.latest() if path is None else Path(path);seal=self.store.verify(path)
        state=json.loads((path/'controller.json').read_text())
        if seal['metadata'].get('controller_sha256')!=canonical_sha(state) or state.get('version')!=JOURNAL_VERSION or state['plan_sha256']!=self.plan.sha256 or state['training_run_sha256']!=self.run_sha256:
            raise ValueError('checkpoint controller/run/plan integrity differs')
        self.runtime.restore(path)
        if state['records']!=self.records:raise ValueError('checkpoint controller clock differs')
        self.archived=state['archived'];self.opening_sha256=state['opening_sha256']
        for record,checksum in self.archived.items():
            if int(record) not in self.plan.milestones or int(record)>self.records:raise ValueError('checkpoint contains future/undeclared milestone evidence')
            archive=self._archive_path(int(record));self.store.verify(archive)
            if digest(archive/'manifest.json')!=checksum:raise ValueError('permanent archived evidence changed')
        if self.plan.scope not in ('confirmation','loop') and self.records>self.opening and self.opening_sha256 is None:raise ValueError('post-opening checkpoint has no recorded opening authorization')
        self._sync_decision();return path


    def finalize(self):
        decision=self._decision()
        ended_no_go=decision is not None and decision['status']=='no_go' and self.records==self.opening
        if self.records!=self.plan.records and not ended_no_go:raise ValueError('an active/budget-limited run cannot be finalized as a completed endpoint')
        self.checkpoint();return self.store.finalize()

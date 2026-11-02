"""Resumable one-step NFSP training on solved multiround tasks and Gray Go.

Both roles contribute actual own-action records. Frozen episode actors survive
update/evaluation/chunk boundaries; budget exhaustion never terminates a game.
"""
from contextlib import contextmanager
from dataclasses import asdict,dataclass
import fcntl
import hashlib
from importlib import metadata
import io
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from .buffers import ReplayBuffer,ReservoirBuffer
from .engine import BLACK,WHITE
from .episode_audit import EpisodeAudit
from .manifest import make_manifest,source_digest,write_report
from .models import ModelConfig,epsilon_greedy_distribution
from .nfsp import NFSPBackend,NFSPConfig
from .nfsp_collection import FrozenPolicies,EpisodeActor,JointEpisode
from .schedules import LearnerSchedule
from .storage import CheckpointStore,DEFAULT_RESERVE
from .task_adapter import TaskAdapter,ADAPTER_VERSION
from .task_evaluation import evaluate_task
from .tasks import BalancedCoverage

HEADER_DTYPE=np.dtype([
    ('episode','<u8'),('start_record','<u8'),('steps','<u4'),('root','<u2'),('generation','<u8'),
    ('supplemental','?'),('forced_role','u1'),('forced_action','<u2'),('average_ready','?'),
    ('best_response','u1',(2,)),('acting','u1',(2,)),('mode_draw','<f8',(2,)),
    ('ramp_draw','<f8',(2,)),('ramp_probability','<f8',(2,)),('epsilon','<f8',(2,)),
    ('reward_black','<f8'),('termination','u1'),('trajectory_sha256','u1',(32,))])


@dataclass(frozen=True)
class NFSPTaskConfig:
    time_cap_seconds: float
    task: str='delayed'
    learner: str='neural'
    seed: int=0
    records: int=15_000_000
    eval_every: int=500_000
    batch: int=512
    learning_start: int=4096
    capacity: int=500_000
    q_reuse: int=1
    target_every: int=12800
    learning_rate: float=3e-4
    max_grad_norm: float=.5
    reservoir_open: int=5_000_000
    handoff_end: int=7_000_000
    epsilon_decay: int=7_500_000
    uniform_comparison: bool=False
    horizons: tuple=(50,100,200)
    equivalent_actions: int=40
    hidden: int=64
    width: int=64
    blocks: int=8
    norm_groups: int=8
    eval_batch: int=128
    max_nodes: int=100000
    device: str='cpu'
    purpose: str='a2'
    budget_profile_sha256: str|None=None
    storage: str='features'
    native_library: str|None=None

    def __post_init__(self):
        object.__setattr__(self,'horizons',tuple(self.horizons))
        if self.task not in ('delayed','delayed-forced','branching','gray-pure','gray-mixed') or self.learner not in ('neural','tabular'):
            raise ValueError('unknown multiround task/learner')
        if type(self.seed) is not int or self.seed<0:raise ValueError('nonnegative seed required')
        if any(type(n) is not int or n<2 or n%2 for n in (self.records,self.eval_every,self.batch)):
            raise ValueError('positive even record/evaluation/update counts required')
        if self.storage not in ('features','rule-state') or (self.storage=='rule-state' and (self.task not in ('gray-pure','gray-mixed') or self.learner!='neural')):
            raise ValueError('rule-state storage is for neural board tasks')
        if self.native_library is not None:
            if self.storage!='rule-state':raise ValueError('native decoding requires rule-state storage')
            object.__setattr__(self,'native_library',str(Path(self.native_library).expanduser().resolve()))
        self.backend_config();self.schedule();ModelConfig(self.width,self.blocks,self.norm_groups)
        if type(self.capacity) is not int or self.capacity<self.learning_start:raise ValueError('replay must cover warm-up')
        if any(type(n) is not int or n<1 for n in (self.hidden,self.eval_batch,self.max_nodes,self.equivalent_actions)):
            raise ValueError('positive adapter/evaluation settings required')
        if not self.horizons or len(set(self.horizons))!=len(self.horizons) or any(type(h) is not int or h<2 for h in self.horizons):
            raise ValueError('distinct horizons of at least two rounds required')
        if not math.isfinite(self.time_cap_seconds) or self.time_cap_seconds<=0:raise ValueError('explicit finite time ceiling required')
        if self.purpose not in ('a2','profile'):raise ValueError('a2 or profile purpose required')
        if self.purpose=='a2' and self.records>=15_000_000 and not self.budget_profile_sha256:
            raise ValueError('full A2 task runs require a declared complete-learner budget profile')
        if self.budget_profile_sha256 is not None and (len(self.budget_profile_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.budget_profile_sha256)):
            raise ValueError('invalid budget-profile SHA-256')
        if self.reservoir_open%2 or self.handoff_end%2:raise ValueError('opening/handoff must preserve paired record boundaries')

    def backend_config(self):return NFSPConfig(self.batch,self.learning_start,self.q_reuse,self.target_every,self.learning_rate,self.max_grad_norm)
    def schedule(self):return LearnerSchedule(self.epsilon_decay,self.reservoir_open,self.handoff_end)
    def plan(self):return json.loads(json.dumps(asdict(self)))
    def formal_protocol(self):
        return (self.purpose=='a2' and bool(self.budget_profile_sha256) and self.task!='delayed-forced' and self.records==15_000_000
            and self.eval_every==500_000 and self.batch==512 and self.learning_start==4096 and self.capacity==500_000
            and self.target_every==12800 and self.learning_rate==3e-4 and self.max_grad_norm==.5
            and self.reservoir_open==5_000_000 and self.handoff_end==7_000_000 and self.epsilon_decay==7_500_000
            and not self.uniform_comparison and self.horizons==(50,100,200) and self.equivalent_actions==40
            and self.hidden==64 and (self.width,self.blocks,self.norm_groups)==(64,8,8))


class NFSPTaskTrainer:
    def __init__(self,config,run_dir,*,resume=False,reserve_bytes=DEFAULT_RESERVE):
        started=time.monotonic();self.config=config;self.root=Path(run_dir);self.root.mkdir(parents=True,exist_ok=True)
        self.store=CheckpointStore(self.root,reserve_bytes=reserve_bytes);self.device=torch.device(config.device)
        torch.set_num_threads(1);torch.manual_seed(config.seed)
        self.adapter=TaskAdapter(config);self.schedule=config.schedule()
        q,average=self.adapter.models();self.backend=NFSPBackend(q,average,config.backend_config(),device=config.device)
        self.replay=ReplayBuffer(config.capacity,self.adapter.spec);self.reservoir=ReservoirBuffer(config.capacity,self.adapter.spec,opening_record=config.reservoir_open)
        offsets={'mode_black':101,'mode_white':102,'action_black':103,'action_white':104,'reservoir':105,'q_replay':106,'supervised':107}
        self.rng={name:np.random.default_rng(config.seed+offset) for name,offset in offsets.items()}
        self.coverage=BalancedCoverage(config.horizons if self.adapter.delayed else (2,),seed=config.seed+108) if not self.adapter.board else None
        self.active=None;self.active_root=None;self.active_start=None;self.cached=None
        self.episodes=self.completed_rounds=0;self.next_eval=config.eval_every;self.evaluations=[];self.last_saved=-1
        self.per_role=[0,0];self.per_role_q=[0,0];self.best_response_visits=[0,0];self.acting_counts={'q':0,'average':0,'uniform':0}
        self.termination_counts={'horizon':0,'mutual_pass':0,'round_limit':0};self.outcomes=[0,0,0]
        self.elapsed=0.;self._started=None;self.timings={k:0. for k in ('setup','collection','learning','evaluation','checkpoint')}
        shape=(len(self.adapter.states),2,self.adapter.actions,2)
        self.metrics={'visits':np.zeros(shape,dtype=np.uint64),'q_eligible_visits':np.zeros(shape,dtype=np.uint64),'replay_starts':np.zeros(shape,dtype=np.uint64),
            'overlap_counts':np.zeros((2,2,3),dtype=np.uint64),'overlap_sums':np.zeros((2,2,3,4),dtype=np.float64),
            'root_episodes':np.zeros((len(self.adapter.roots),2),dtype=np.uint64)}
        self.loss={'q_updates':0,'q_loss_sum':0.,'supervised_updates':0,'supervised_loss_sum':0.,'projected_targets':0,'raw_min':None,'raw_max':None}
        with self._lock():
            path=self.root/'run.json'
            if resume:
                self.manifest=json.loads(path.read_text())
                if self.manifest['config']!=config.plan() or self.manifest['source_sha256']!=source_digest():raise ValueError('resume configuration/source differs')
                for package,version in self.manifest['dependencies'].items():
                    if version is not None and metadata.version(package)!=version:raise ValueError('resume dependency versions differ')
                self.audit=EpisodeAudit(self.root,dtype=HEADER_DTYPE);self._restore()
            else:
                if path.exists():raise FileExistsError('run already exists; use explicit resume')
                self.manifest=make_manifest(seed=config.seed,command={'operation':'nfsp-task','config':config.plan()})
                self.manifest.update({'config':config.plan(),'task_schema':ADAPTER_VERSION,'adapter':self.adapter.manifest(),
                    'encoding_version':self.adapter.spec.version,'rules':None if not self.adapter.board else self.adapter.roots[0].to_dict()['rules'],
                    'rule_version':None if not self.adapter.board else self.adapter.roots[0].to_dict()['rule_version'],'baseline_version':None,
                    'shared_parameters_across_roles':True,'counter_unit':'two own-side fresh records per simultaneous round; supplemental records included',
                    'eligibility_version':2,'q_eligibility':'exclude only the passive pre-forcing root; keep own forced action and normal suffixes',
                    'record_clocks':'aggregate fresh records drive budget/epsilon/opening/ramp; eligible Q records drive Q reuse and target refresh',
                    'backend_settings':asdict(config.backend_config()),'schedule':asdict(self.schedule),
                    'model':{'q_class':type(self.backend.q).__name__,'average_class':type(self.backend.average).__name__,'public_inputs_only':True},
                    'q_optimizer':{'class':'Adam','defaults':dict(self.backend.q_optimizer.defaults)},
                    'supervised_optimizer':None if self.backend.average_optimizer is None else {'class':'Adam','defaults':dict(self.backend.average_optimizer.defaults)},
                    'average_method':'exact reservoir conditional label mean' if self.backend.exact_average else 'masked frozen-label cross-entropy',
                    'q_loss':'mean squared error against projected expected one-step targets',
                    'runtime':{'workers':1,'torch_threads':torch.get_num_threads(),'device':str(self.device),
                        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_benchmark':torch.backends.cudnn.benchmark,
                        'cudnn_deterministic':torch.backends.cudnn.deterministic},
                    'rng_stream_seeds':{name:config.seed+offset for name,offset in offsets.items()}|{'coverage':config.seed+108},
                    'replay_age_bins_fresh_records':[12800,128000],'overlap_sum_fields':['min(1,pi/mu)','target_probability','behavior_probability','fresh_record_age'],
                    'episode_audit':{'termination_codes':{'horizon':1,'mutual_pass':2,'round_limit':3},'unforced_action_sentinel':65535},
                    'target_refresh_order':'after optimizer work at exact fresh-Q-record boundary','pid':os.getpid()})
                write_report(path,self.manifest,reserve_bytes=reserve_bytes)
                self.audit=EpisodeAudit(self.root,create=True,dtype=HEADER_DTYPE)
                self.timings['setup']=time.monotonic()-started;self.save()
        if resume:self.timings['setup']+=time.monotonic()-started

    @contextmanager
    def _lock(self):
        with (self.root/'.trainer.lock').open('a+b') as file:
            fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:yield
            finally:fcntl.flock(file,fcntl.LOCK_UN)

    @property
    def records(self):return self.backend.counters.fresh_records
    def elapsed_now(self):return self.elapsed+(time.monotonic()-self._started if self._started is not None else 0.)

    def _start_episode(self):
        root_index,forced=self.adapter.new_root(self.coverage)
        version=self.backend.counters.q_updates+self.backend.counters.supervised_updates
        if self.cached is None or self.cached.version!=version:
            self.cached=FrozenPolicies(self.backend.q,self.backend.average,version=version,average_ready=self.backend.average_ready)
        actors=[]
        for mode,action in (('mode_black','action_black'),('mode_white','action_white')):
            choice=self.schedule.choose_episode(self.records,self.rng[mode],average_ready=self.cached.average_ready,uniform_comparison=self.config.uniform_comparison)
            actors.append(EpisodeActor(self.cached,choice,self.rng[action]))
        self.active=JointEpisode(self.adapter.game,self.adapter.roots[root_index],*actors,episode_id=self.episodes,supplemental=forced is not None,forced_root=forced)
        self.active_root=root_index;self.active_start=self.records
        self.metrics['root_episodes'][root_index,int(forced is not None)]+=1

    def _finish_episode(self,event):
        episode=self.active;actors=[episode.actors[p] for p in (BLACK,WHITE)];choices=[actor.choice for actor in actors]
        reason=episode.state.terminal_reason if self.adapter.board else 'horizon'
        digest=hashlib.sha256(json.dumps(episode.log,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).digest()
        row=np.empty(1,dtype=HEADER_DTYPE)
        row[0]=(self.episodes,self.active_start,episode.steps,self.active_root,actors[0].snapshot.version,episode.supplemental,
            0 if episode.forced_root is None else episode.forced_root[0],65535 if episode.forced_root is None else episode.forced_root[1],actors[0].snapshot.average_ready,
            [c.best_response_mode for c in choices],[{'q':0,'average':1,'uniform':2}[c.acting_policy] for c in choices],
            [c.mode_draw for c in choices],[c.ramp_draw for c in choices],[c.ramp_probability for c in choices],[c.epsilon for c in choices],
            event['reward_black'],{'horizon':1,'mutual_pass':2,'round_limit':3}[reason],np.frombuffer(digest,dtype=np.uint8))
        self.audit.append(row);self.episodes+=1;self.completed_rounds+=episode.steps;self.termination_counts[reason]+=1
        self.outcomes[0 if event['reward_black']>0 else 1 if event['reward_black']<0 else 2]+=1
        self.active=None;self.active_root=self.active_start=None

    def collect_round(self):
        started=time.monotonic();learning=0.
        if self.active is None:self._start_episode()
        episode=self.active;state=episode.state;index=self.adapter.index[state]
        callback=self.backend.average.observe_reservoir_change if self.backend.exact_average else None
        def learned_record(*,role,record,ordinary,q_eligible):
            nonlocal learning
            if self.records!=record:raise ValueError('chronological own-side record callback differs from aggregate clock')
            at=time.monotonic()
            stats=self.backend.learn_after_fresh(1,ordinary,self.replay,self.reservoir,self.rng['q_replay'],
                epsilon=self.schedule.epsilon(record+1),q_eligible=q_eligible,supervised_rng=self.rng['supervised'],q_observer=self._observe_q)
            learning+=time.monotonic()-at;self.per_role_q[role-1]+=q_eligible
            for q in stats['q']:
                self.loss['q_updates']+=1;self.loss['q_loss_sum']+=q['loss'];self.loss['projected_targets']+=q['projected_targets']
                self.loss['raw_min']=q['raw_target_min'] if self.loss['raw_min'] is None else min(self.loss['raw_min'],q['raw_target_min'])
                self.loss['raw_max']=q['raw_target_max'] if self.loss['raw_max'] is None else max(self.loss['raw_max'],q['raw_target_max'])
            if stats['supervised'] is not None:
                self.loss['supervised_updates']+=1
                if stats['supervised']['loss'] is not None:self.loss['supervised_loss_sum']+=stats['supervised']['loss']
        event=episode.step(self.replay,self.reservoir,self.rng['reservoir'],self.schedule,record=self.records,
                           on_reservoir_change=callback,on_record=learned_record)
        for role in range(2):
            self.metrics['visits'][index,role,event['actions'][role],int(episode.supplemental)]+=1
            self.metrics['q_eligible_visits'][index,role,event['actions'][role],int(episode.supplemental)]+=int(event['q_eligible'][role])
            self.per_role[role]+=1;self.best_response_visits[role]+=int(event['modes'][role]);self.acting_counts[event['acting_policies'][role]]+=1
        ordinary=0 if episode.supplemental else 2
        if event['terminated']:self._finish_episode(event)
        self.timings['learning']+=learning;self.timings['collection']+=max(0.,time.monotonic()-started-learning)
        return ordinary

    @torch.no_grad()
    def _observe_q(self,data,predictions,batch,epsilon):
        states,roles=self.adapter.sample_bins(data);origin=data['supplemental'].astype(np.int64)
        np.add.at(self.metrics['replay_starts'],(states,roles,data['action'],origin),1)
        pi=epsilon_greedy_distribution(predictions,batch.legal,epsilon).gather(1,batch.actions[:,None]).squeeze(1).cpu().double().numpy()
        mu=data['behavior_prob'].astype(float);age=self.records-1-data['ordinal'].astype(np.int64)
        if (age<0).any():raise ValueError('replay contains future records')
        bins=np.where(age<=12800,0,np.where(age<=128000,1,2));mode=data['mode'].astype(np.int64)
        np.add.at(self.metrics['overlap_counts'],(mode,origin,bins),1)
        for metric,values in enumerate((np.minimum(1,pi/mu),pi,mu,age)):
            np.add.at(self.metrics['overlap_sums'][...,metric],(mode,origin,bins),values)

    def evaluate(self):
        start=time.monotonic()
        roots=evaluate_task(self.adapter,self.backend,self.schedule.epsilon(self.records),batch_size=self.config.eval_batch)
        result={'records':self.records,'roots':roots,'average_ready':self.backend.average_ready,'counters':asdict(self.backend.counters),
            'coverage':self.coverage_summary(),'epsilon':self.schedule.epsilon(self.records),'handoff_probability':self.schedule.ramp(self.records),'elapsed_seconds':self.elapsed_now()}
        self.evaluations.append(result);self.timings['evaluation']+=time.monotonic()-start
        return result

    def coverage_summary(self):
        return {'visit_records':int(self.metrics['visits'].sum()),'q_eligible_records':int(self.metrics['q_eligible_visits'].sum()),
                'excluded_q_records':self.records-self.backend.counters.fresh_q,'sampled_start_targets':int(self.metrics['replay_starts'].sum()),
            'root_episodes':self.metrics['root_episodes'].tolist(),'overlap_counts':self.metrics['overlap_counts'].tolist(),
            'overlap_sums':self.metrics['overlap_sums'].tolist()}

    def _state(self):
        active=None
        if self.active is not None:
            active={'episode':self.active.state_dict(self.adapter.encode_state),'snapshot':self.active.actors[BLACK].snapshot.state_dict(),
                    'root_index':self.active_root,'start_record':self.active_start}
        return {'format_version':2,'task_schema':ADAPTER_VERSION,'config':self.config.plan(),'backend':self.backend.state_dict(),
            'active':active,'episodes':self.episodes,'completed_rounds':self.completed_rounds,'next_eval':self.next_eval,'evaluations':self.evaluations,
            'per_role':self.per_role,'per_role_q':self.per_role_q,'best_response_visits':self.best_response_visits,'acting_counts':self.acting_counts,
            'termination_counts':self.termination_counts,'outcomes':self.outcomes,'elapsed':self.elapsed_now(),'timings':self.timings,'loss':self.loss,
            'coverage':None if self.coverage is None else self.coverage.state_dict(),'rng':{k:v.bit_generator.state for k,v in self.rng.items()},
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [],'episode_audit':self.audit.prefix(self.episodes)}

    def save(self,*,weights=False):
        if (self.records%2 or self.per_role!=[self.records//2]*2 or self.replay.insertions!=self.records
                or self.replay.q_insertions!=self.backend.counters.fresh_q or sum(self.per_role_q)!=self.backend.counters.fresh_q):
            raise ValueError('only a complete jointly committed round may become durable')
        start=time.monotonic()
        if weights:
            destination=self.root/'weights'/f'{self.records:012d}'
            state={'config':self.config.plan(),'q':self.backend.q.state_dict(),'average':self.backend.average.state_dict()}
            if destination.exists():
                self.store.verify(destination);previous=torch.load(destination/'models.pt',weights_only=True,map_location=self.device)
                if previous['config']!=state['config'] or any(not torch.equal(previous[head][k],v) for head in ('q','average') for k,v in state[head].items()):raise ValueError('permanent weights conflict')
            else:
                file=io.BytesIO();torch.save(state,file);raw=file.getvalue()
                self.store.save(self.records,lambda p:(p/'models.pt').write_bytes(raw),kind='weights',metadata={'task_schema':ADAPTER_VERSION,'source_sha256':self.manifest['source_sha256']},estimated_peak_bytes=len(raw)+65536)
        if self.last_saved!=self.records:
            file=io.BytesIO();torch.save(self._state(),file);raw=file.getvalue()
            def writer(path):
                self.replay.save(path/'replay.npz');self.reservoir.save(path/'reservoir.npz')
                np.savez_compressed(path/'metrics.npz',**self.metrics);(path/'trainer.pt').write_bytes(raw)
            peak=math.ceil(1.05*(len(raw)+self.replay.snapshot_raw_bytes+self.reservoir.snapshot_raw_bytes+sum(a.nbytes for a in self.metrics.values())))+65536
            self.store.save(self.records,writer,metadata={'task_schema':ADAPTER_VERSION,'source_sha256':self.manifest['source_sha256']},estimated_peak_bytes=peak)
            self.last_saved=self.records
        self.timings['checkpoint']+=time.monotonic()-start

    def _restore(self):
        path=self.store.latest();meta=self.store.verify(path)
        if meta['metadata']['source_sha256']!=self.manifest['source_sha256']:raise ValueError('checkpoint source differs')
        state=torch.load(path/'trainer.pt',weights_only=True,map_location='cpu')
        if state['format_version']!=2 or state['task_schema']!=ADAPTER_VERSION or state['config']!=self.config.plan():raise ValueError('checkpoint plan/schema differs')
        self.backend.load_state_dict(state['backend']);self.replay=ReplayBuffer.load(path/'replay.npz');self.reservoir=ReservoirBuffer.load(path/'reservoir.npz')
        if self.replay.spec!=self.adapter.spec or self.reservoir.spec!=self.adapter.spec or self.replay.capacity!=self.config.capacity or self.reservoir.capacity!=self.config.capacity or self.reservoir.opening_record!=self.config.reservoir_open:raise ValueError('buffer checkpoint plan differs')
        for name in ('episodes','completed_rounds','next_eval','evaluations','per_role','per_role_q','best_response_visits','acting_counts','termination_counts','outcomes','timings','elapsed','loss'):setattr(self,name,state[name])
        with np.load(path/'metrics.npz',allow_pickle=False) as data:
            if set(data.files)!=set(self.metrics):raise ValueError('metrics inventory differs')
            for name,expected in self.metrics.items():
                if data[name].shape!=expected.shape or data[name].dtype!=expected.dtype or not np.isfinite(data[name]).all():raise ValueError('invalid metrics checkpoint')
                expected[:]=data[name]
        if self.coverage is not None:self.coverage.load_state_dict(state['coverage'])
        elif state['coverage'] is not None:raise ValueError('unexpected coverage schedule')
        for name,rng in self.rng.items():rng.bit_generator.state=state['rng'][name]
        if self.backend.exact_average:self.backend.average.rebuild_tracker(self.reservoir)
        active=state['active']
        if active is not None:
            saved=active['snapshot'];snapshot=FrozenPolicies(self.backend.q,self.backend.average,version=saved['version'],average_ready=saved['average_ready']);snapshot.load_state_dict(saved)
            self.active=JointEpisode.restore(self.adapter.game,active['episode'],self.adapter.decode_state,{snapshot.version:snapshot})
            self.active_root,self.active_start=active['root_index'],active['start_record']
            if not 0<=self.active_root<len(self.adapter.roots) or self.adapter.game.terminal(self.active.state):raise ValueError('invalid active episode')
            self.rng['action_black']=self.active.actors[BLACK].rng;self.rng['action_white']=self.active.actors[WHITE].rng
        active_steps=0 if self.active is None else self.active.steps
        if self.records!=2*(self.completed_rounds+active_steps) or self.replay.insertions!=self.records or meta['step']!=self.records or self.per_role!=[self.records//2]*2 or sum(self.acting_counts.values())!=self.records:raise ValueError('resume record accounting differs')
        if int(self.metrics['visits'].sum())!=self.records or int(self.metrics['replay_starts'].sum())!=self.backend.counters.q_loss_targets or int(self.metrics['overlap_counts'].sum())!=self.backend.counters.q_loss_targets:raise ValueError('metrics/fresh/start counts differ')
        if int(self.metrics['visits'][...,0].sum())!=self.backend.counters.fresh_ordinary or int(self.metrics['root_episodes'].sum())!=self.episodes+(self.active is not None):raise ValueError('coverage counters differ')
        if (self.replay.q_insertions!=self.backend.counters.fresh_q or int(self.metrics['q_eligible_visits'].sum())!=self.backend.counters.fresh_q
                or self.metrics['q_eligible_visits'].sum(axis=(0,2,3)).tolist()!=self.per_role_q):
            raise ValueError('Q eligibility/role/visit/replay counters differ')
        excluded=[0,0]
        if self.coverage is not None:
            for root_index,root in enumerate(self.adapter.roots):
                count=int(self.metrics['root_episodes'][root_index,1])
                if self.active is not None and self.active_root==root_index and self.active.supplemental and not self.active.steps:count-=1
                excluded[2-root.controller]+=count
        if self.per_role_q!=[total-missing for total,missing in zip(self.per_role,excluded)]:raise ValueError('per-role intervention exclusion differs')
        self.audit.restore_prefix(state['episode_audit'])
        if state['episode_audit']['episodes']!=self.episodes:raise ValueError('episode audit count differs')
        torch.set_rng_state(state['torch_rng'].cpu())
        if self.device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        self.last_saved=self.records
        previous=self.root/'result.json'
        if previous.exists():
            result=json.loads(previous.read_text())
            if result['records']==self.records and result['source_sha256']==self.manifest['source_sha256']:self.elapsed=max(self.elapsed,result['elapsed_seconds'])

    def result(self,status):
        final=self.evaluations[-5:];enough=status=='complete' and self.records==self.config.records and len(final)==5
        ready=enough and all(e['average_ready'] for e in final)
        avg=max((max(r['average_gaps']) for e in final for r in e['roots']),default=None)
        qgap=max((max(r['greedy_q_gaps']) for e in final for r in e['roots']),default=None)
        error=max((r['max_early_error'] for e in final for r in e['roots']),default=None)
        credit=bool(enough and qgap<=.05 and error<=.1)
        if self.adapter.delayed:band='eligible_for_limited_pilot' if ready and avg<=.1 and credit else ('averaging_lag' if ready and avg<=.25 and credit else 'not_ready')
        else:band='short_game_pass' if ready and avg<=.05 else 'not_ready'
        passed=ready and (band=='eligible_for_limited_pilot' if self.adapter.delayed else band=='short_game_pass')
        if self.config.task=='branching':passed=passed and credit
        return {'status':status,'task':self.config.task,'learner':self.config.learner,'seed':self.config.seed,'records':self.records,
            'completed_episodes':self.episodes,'active_episode_steps':0 if self.active is None else self.active.steps,'per_role_records':self.per_role,
            'per_role_q_records':self.per_role_q,'excluded_q_records':self.records-self.backend.counters.fresh_q,
            'best_response_visits':self.best_response_visits,'acting_policy_counts':self.acting_counts,'termination_counts':self.termination_counts,'outcomes':self.outcomes,
            'counters':asdict(self.backend.counters),'reservoir_seen':self.reservoir.seen,'reservoir_size':self.reservoir.size,
            'coverage':self.coverage_summary(),'evaluations':self.evaluations,'final_five_max_average_gap':avg,'final_five_max_greedy_gap':qgap,
            'final_five_max_early_error':error,'q_credit_passed':credit,'average_readiness':band,'formal_protocol':self.config.formal_protocol(),
            'formal_case_passed':bool(passed and self.config.formal_protocol()),'a2_complete':False,'elapsed_seconds':self.elapsed_now(),
            'timings':self.timings,'loss':self.loss,'source_sha256':self.manifest['source_sha256']}

    def run(self,*,until_records=None,stop_requested=lambda:False,verbose=True):
        stop=self.config.records if until_records is None else until_records
        if type(stop) is not int or stop%2 or not self.records<=stop<=self.config.records:raise ValueError('invalid paired stop')
        with self._lock():
            if self.last_saved>=0 and self.store.verify(self.store.latest())['step']!=self.last_saved:raise RuntimeError('stale writer; reload current checkpoint')
            status='complete';self._started=time.monotonic()
            try:
                while self.records<stop:
                    requested=stop_requested()
                    if requested:status=requested if isinstance(requested,str) else 'interrupted';break
                    if self.elapsed_now()>=self.config.time_cap_seconds:status='budget_limited';break
                    self.store.ensure_space();self.collect_round()
                    if self.records==self.next_eval or self.records==self.config.records:
                        if self.records==self.next_eval:self.next_eval+=self.config.eval_every
                        evaluation=self.evaluate();self.save(weights=True)
                        write_report(self.root/'progress.json',self.result('running'),reserve_bytes=self.store.reserve_bytes)
                        if verbose:print(json.dumps({'event':'nfsp-task-evaluation','task':self.config.task,'learner':self.config.learner,'records':self.records,'max_average_gap':max(max(r['average_gaps']) for r in evaluation['roots']),'max_early_error':max(r['max_early_error'] for r in evaluation['roots'])}),flush=True)
                if self.records<self.config.records and status=='complete':status='segment_paused'
                self.save(weights=self.records==self.config.records)
                if self.records==self.config.records:self.store.finalize()
                result=self.result(status);write_report(self.root/'result.json',result,reserve_bytes=self.store.reserve_bytes)
                return result
            finally:self.elapsed=self.elapsed_now();self._started=None

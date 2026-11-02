"""Resumable native NFSP matrix trainer using the production learning backend.

The role-only adapter batches inference over its two possible observations.
Both players contribute one fresh record per physical game. Behavior and greedy
labels are frozen before either commitment; no exact solution is a training label.
"""
from contextlib import contextmanager
from dataclasses import asdict,dataclass
import fcntl
import io
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from .buffers import ObservationSpec,ReplayBuffer,ReservoirBuffer
from .encoding import Observation
from .episode_audit import EpisodeAudit,EPISODE_DTYPE
from .manifest import make_manifest,source_digest,write_report
from .matrix_training import matrix_payoffs
from .models import epsilon_greedy_distribution,greedy_distribution,masked_log_probs
from .nfsp import NFSPBackend,NFSPConfig
from .schedules import LearnerSchedule
from .sampling import draw_actions
from .solving import solve_matrix,value_gaps
from .storage import CheckpointStore,DEFAULT_RESERVE
from .tabular_average import TabularAverage
from .task_models import CompactQ,CompactPolicy,TabularQ

TASK_VERSION='graygo/nfsp-matrix-role/1'


@dataclass(frozen=True)
class NFSPMatrixConfig:
    task: str='asymmetric'
    learner: str='neural'
    seed: int=0
    records: int=15_000_000
    eval_every: int=500_000
    batch: int=128
    learning_start: int=4096
    capacity: int=200_000
    q_reuse: int=1
    target_every: int=12800
    hidden: int=64
    learning_rate: float=3e-4
    max_grad_norm: float=.5
    reservoir_open: int=5_000_000
    handoff_end: int=7_000_000
    epsilon_decay: int=7_500_000
    uniform_comparison: bool=False
    label_rule: str='greedy'
    device: str='cpu'
    time_cap_seconds: float=8*3600

    def __post_init__(self):
        matrix_payoffs(self.task)
        if self.learner not in ('neural','tabular'):raise ValueError('neural or tabular learner required')
        if type(self.seed) is not int or self.seed<0:raise ValueError('nonnegative integral seed required')
        if any(type(n) is not int or n<2 or n%2 for n in (self.records,self.eval_every,self.batch)):
            raise ValueError('positive even record/evaluation/update counts required')
        if type(self.capacity) is not int or self.capacity<self.learning_start or type(self.hidden) is not int or self.hidden<1:
            raise ValueError('capacity must cover warm-up; hidden width must be positive')
        if self.label_rule not in ('greedy','executed') or (self.label_rule=='executed' and self.task!='pure82'):
            raise ValueError('executed-action label control is restricted to the pure-solution task')
        if not math.isfinite(self.time_cap_seconds) or self.time_cap_seconds<=0:raise ValueError('positive finite time ceiling required')
        self.backend_config();self.schedule()
        if self.reservoir_open%2 or self.handoff_end%2:raise ValueError('matrix opening/handoff must preserve role balance')

    def backend_config(self):
        return NFSPConfig(self.batch,self.learning_start,self.q_reuse,self.target_every,self.learning_rate,self.max_grad_norm)

    def schedule(self):return LearnerSchedule(self.epsilon_decay,self.reservoir_open,self.handoff_end)

    def formal_protocol(self):
        return (self.records==15_000_000 and self.eval_every==500_000 and self.batch==128 and self.learning_start==4096
                and self.capacity==200_000 and self.hidden==64 and self.max_grad_norm==.5 and self.q_reuse==1 and self.target_every==12800 and self.learning_rate==3e-4
                and self.reservoir_open==5_000_000 and self.handoff_end==7_000_000 and self.epsilon_decay==7_500_000
                and not self.uniform_comparison and self.label_rule=='greedy')


class NFSPMatrixTrainer:
    def __init__(self,config,run_dir,*,resume=False,reserve_bytes=DEFAULT_RESERVE):
        self.config=config;self.root=Path(run_dir);self.root.mkdir(parents=True,exist_ok=True)
        self.store=CheckpointStore(self.root,reserve_bytes=reserve_bytes)
        self.device=torch.device(config.device);torch.set_num_threads(1);torch.manual_seed(config.seed)
        self.payoffs=matrix_payoffs(config.task);self.solution=solve_matrix(self.payoffs)
        self.actions=max(self.payoffs.shape);self.spec=ObservationSpec((2,),0,self.actions,TASK_VERSION)
        self.schedule=config.schedule()
        self.legal=np.arange(self.actions)[None,:]<np.array(self.payoffs.shape)[:,None]
        self.observations=[Observation(np.eye(2,dtype=np.uint8)[r],np.empty(0,dtype=np.float32),self.legal[r].copy()) for r in range(2)]
        self.terminal=Observation(np.zeros(2,dtype=np.uint8),np.empty(0,dtype=np.float32),np.zeros(self.actions,dtype=bool))
        if config.learner=='neural':q,average=CompactQ(2,0,self.actions,config.hidden),CompactPolicy(2,0,self.actions,config.hidden)
        else:q,average=TabularQ(2,self.actions),TabularAverage(2,self.actions)
        self.backend=NFSPBackend(q,average,config.backend_config(),device=config.device)
        self.replay=ReplayBuffer(config.capacity,self.spec);self.reservoir=ReservoirBuffer(config.capacity,self.spec,opening_record=config.reservoir_open)
        # Draw streams are independent: an exact-average control does not shift Q sampling.
        offsets={'mode_black':101,'mode_white':102,'action_black':103,'action_white':104,
                 'reservoir':105,'q_replay':106,'supervised':107}
        self.rng={name:np.random.default_rng(config.seed+offset) for name,offset in offsets.items()}
        self.episodes=0;self.per_role=[0,0];self.best_response_visits=[0,0]
        self.acting_counts={'q':0,'average':0,'uniform':0};self.inserted_labels=0
        self.evaluations=[];self.next_eval=config.eval_every;self.elapsed=0.;self._started=None;self.last_saved=-1
        self.timings={'collection':0.,'learning':0.,'evaluation':0.,'checkpoint':0.}
        self.loss_stats={'q_updates':0,'q_loss_sum':0.,'supervised_loss_updates':0,'supervised_loss_sum':0.,'projected_targets':0,'raw_target_min':None,'raw_target_max':None}
        with self._run_lock():
            path=self.root/'run.json'
            if resume:
                self.manifest=json.loads(path.read_text())
                if self.manifest['config']!=asdict(config) or self.manifest['source_sha256']!=source_digest():
                    raise ValueError('resume plan/source differs; start an explicit new lineage')
                self.audit=EpisodeAudit(self.root)
                self._restore()
            else:
                if path.exists():raise FileExistsError('run already exists; resume it instead of overwriting')
                self.manifest=make_manifest(seed=config.seed,command={'operation':'nfsp-matrix','config':asdict(config)})
                self.manifest.update({'config':asdict(config),'task_schema':TASK_VERSION,'encoding_version':TASK_VERSION,
                    'rule_version':None,'baseline_version':None,'payoffs':self.payoffs.tolist(),'oracle_value':self.solution.value,
                    'backend_settings':asdict(config.backend_config()),'schedule':asdict(self.schedule),
                    'shared_parameters_across_roles':True,'counter_unit':'two own-side fresh records per physical one-round game',
                    'rng_stream_seeds':{k:config.seed+v for k,v in offsets.items()},
                    'model':{'learner':config.learner,'hidden':config.hidden,'public_inputs':'two-role one-hot; no opponent identity'},
                    'average_method':'exact reservoir conditional label mean' if config.learner=='tabular' else 'masked frozen-label cross-entropy',
                    'q_loss':'mean squared error; online-pi expected target values; projected target',
                    'q_optimizer':{'class':'Adam','defaults':dict(self.backend.q_optimizer.defaults)},
                    'supervised_optimizer':None if self.backend.average_optimizer is None else {'class':'Adam','defaults':dict(self.backend.average_optimizer.defaults)},
                    'episode_sampling':'independent role modes, ramp choices and actions; policy frozen for each one-round episode',
                    'target_refresh_order':'after optimizer work at exact fresh-record boundary','pid':os.getpid()})
                write_report(path,self.manifest,reserve_bytes=reserve_bytes)
                self.audit=EpisodeAudit(self.root,create=True)
                self.save()

    @contextmanager
    def _run_lock(self):
        with (self.root/'.trainer.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:yield
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

    @property
    def records(self):return self.backend.counters.fresh_q

    @torch.no_grad()
    def _tables(self):
        features=torch.eye(2,device=self.device);globals=torch.empty(2,0,device=self.device)
        q=self.backend.q(features,globals)
        logits=self.backend.average(features,globals)
        legal=torch.tensor(self.legal,device=self.device)
        if not bool(torch.isfinite(q[legal]).all()) or bool((q[legal].abs()>1+1e-6).any()):raise ValueError('invalid online Q')
        average=masked_log_probs(logits,legal).exp()
        return q,legal,average

    @torch.no_grad()
    def collect(self,count):
        if type(count) is not int or count<2 or count%2 or count>self.backend.counters.until_q_boundary(self.backend.config):
            raise ValueError('matrix collection must stop at a balanced update boundary')
        base=self.records;games=count//2;q,legal,average=self._tables()
        choices=[]
        for game in range(games):
            n=base+2*game
            choices.extend(self.schedule.choose_episode(n,self.rng[name],average_ready=self.backend.average_ready,
                            uniform_comparison=self.config.uniform_comparison) for name in ('mode_black','mode_white'))
        roles=torch.arange(count,device=self.device)%2
        frozen_q=q[roles];masks=legal[roles]
        eps=torch.tensor([c.epsilon for c in choices],device=self.device)
        distributions=epsilon_greedy_distribution(frozen_q,masks,eps).double()
        avg_rows=torch.tensor([c.acting_policy=='average' for c in choices],device=self.device)
        uniform_rows=torch.tensor([c.acting_policy=='uniform' for c in choices],device=self.device)
        distributions[avg_rows]=average[roles[avg_rows]].double()
        distributions[uniform_rows]=masks[uniform_rows].double()/masks[uniform_rows].sum(-1,keepdim=True)
        greedy=(greedy_distribution(frozen_q,masks)>0).cpu().numpy()
        p=distributions.double().cpu().numpy();actions=np.empty(count,dtype=np.int64);probabilities=np.empty(count)
        actions[::2],probabilities[::2]=draw_actions(p[::2],self.rng['action_black'])
        actions[1::2],probabilities[1::2]=draw_actions(p[1::2],self.rng['action_white'])
        rewards=np.repeat(self.payoffs[actions[::2],actions[1::2]],2);rewards[1::2]*=-1
        # Versions identify both public parameter sets; mode/epsilon/ramp are retained separately.
        generation=self.backend.counters.q_updates+self.backend.counters.supervised_updates
        audit=np.empty(games,dtype=EPISODE_DTYPE)
        for game in range(games):
            pair=choices[2*game:2*game+2]
            audit[game]=(self.episodes+game,base+2*game,generation,self.backend.average_ready,
                         [c.best_response_mode for c in pair],[{'q':0,'average':1,'uniform':2}[c.acting_policy] for c in pair],
                         [c.mode_draw for c in pair],[c.ramp_draw for c in pair],[c.ramp_probability for c in pair],
                         [c.epsilon for c in pair],actions[2*game:2*game+2],probabilities[2*game:2*game+2],rewards[2*game])
        for index,choice in enumerate(choices):
            role=index%2;other=index+(1 if role==0 else -1);obs=self.observations[role]
            version=2*generation+int(choice.acting_policy=='average')
            opponent_version=2*generation+int(choices[other].acting_policy=='average')
            self.replay.add(obs,int(actions[index]),float(rewards[index]),self.terminal,True,behavior_prob=float(probabilities[index]),
                            stream=2*(self.episodes+index//2)+role,step=0,opponent_version=opponent_version,policy_version=version,
                            mode=int(choice.best_response_mode))
            self.per_role[role]+=1;self.best_response_visits[role]+=int(choice.best_response_mode);self.acting_counts[choice.acting_policy]+=1
            if self.schedule.label_eligible(choice,base+index):
                label=greedy[index]
                if self.config.label_rule=='executed':
                    label=np.zeros(self.actions,dtype=bool);label[actions[index]]=True
                changed=self.reservoir.offer_index(obs,label,record=base+index,policy_version=version,rng=self.rng['reservoir'],best_response_mode=True)
                if changed is not None:
                    self.inserted_labels+=1
                    if self.backend.exact_average:self.backend.average.observe_reservoir_change(self.reservoir,changed)
        self.audit.append(audit)
        self.episodes+=games
        return {'logical_best_response':sum(c.best_response_mode for c in choices),'average_actions':int(avg_rows.sum()),
                'ramp_min':min(c.ramp_probability for c in choices),'ramp_max':max(c.ramp_probability for c in choices)}

    @torch.no_grad()
    def evaluate(self):
        started=time.monotonic();q,legal,average=self._tables()
        greedy=greedy_distribution(q,legal).double().cpu().numpy();avg=average.double().cpu().numpy()
        greedy/=greedy.sum(-1,keepdims=True);avg/=avg.sum(-1,keepdims=True)
        a,b=self.payoffs.shape
        avg_gaps=value_gaps(self.payoffs,avg[0,:a],avg[1,:b],self.solution.value)
        q_gaps=value_gaps(self.payoffs,greedy[0,:a],greedy[1,:b],self.solution.value)
        result={'records':self.records,'average_ready':self.backend.average_ready,'average_gaps':list(avg_gaps),
                'greedy_q_gaps':list(q_gaps),'average_policies':avg.tolist(),'greedy_q_policies':greedy.tolist(),
                'q_values':q.cpu().numpy().tolist(),'counters':asdict(self.backend.counters),'reservoir_seen':self.reservoir.seen,
                'reservoir_size':self.reservoir.size,'epsilon':self.schedule.epsilon(self.records),
                'handoff_probability':self.schedule.ramp(self.records),'elapsed_seconds':self.elapsed_now()}
        self.evaluations.append(result);self.timings['evaluation']+=time.monotonic()-started
        return result

    def elapsed_now(self):return self.elapsed+(time.monotonic()-self._started if self._started is not None else 0.)

    def _state(self):
        return {'format_version':1,'task_schema':TASK_VERSION,'config':asdict(self.config),'backend':self.backend.state_dict(),
                'episodes':self.episodes,'per_role':self.per_role,'best_response_visits':self.best_response_visits,
                'acting_counts':self.acting_counts,'inserted_labels':self.inserted_labels,'evaluations':self.evaluations,'next_eval':self.next_eval,
                'timings':self.timings,'loss_stats':self.loss_stats,'elapsed':self.elapsed_now(),
                'rng':{name:rng.bit_generator.state for name,rng in self.rng.items()},'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [],
                'episode_audit':self.audit.prefix(self.episodes)}

    def _weights(self):return {'config':asdict(self.config),'q':self.backend.q.state_dict(),'average':self.backend.average.state_dict()}

    def save(self,*,weights=False):
        start=time.monotonic()
        if weights:
            destination=self.root/'weights'/f'{self.records:012d}'
            current=self._weights()
            if destination.exists():
                self.store.verify(destination)
                previous=torch.load(destination/'models.pt',weights_only=True,map_location=self.device)
                if previous['config']!=current['config'] or any(not torch.equal(previous[head][k],v) for head in ('q','average') for k,v in current[head].items()):
                    raise ValueError('permanent evaluation archive conflicts with current model')
            else:
                data=io.BytesIO();torch.save(current,data);raw=data.getvalue()
                self.store.save(self.records,lambda p:(p/'models.pt').write_bytes(raw),kind='weights',
                                metadata={'task_schema':TASK_VERSION,'source_sha256':self.manifest['source_sha256']},estimated_peak_bytes=len(raw)+65536)
        if self.last_saved!=self.records:
            data=io.BytesIO();torch.save(self._state(),data);raw=data.getvalue()
            def writer(path):
                self.replay.save(path/'replay.npz');self.reservoir.save(path/'reservoir.npz')
                (path/'trainer.pt').write_bytes(raw)
            peak=math.ceil(1.05*(self.replay.snapshot_raw_bytes+self.reservoir.snapshot_raw_bytes+len(raw)))+65536
            self.store.save(self.records,writer,metadata={'task_schema':TASK_VERSION,'source_sha256':self.manifest['source_sha256']},estimated_peak_bytes=peak)
            self.last_saved=self.records
        self.timings['checkpoint']+=time.monotonic()-start

    def _restore(self):
        path=self.store.latest();meta=self.store.verify(path)
        if meta['metadata']['source_sha256']!=self.manifest['source_sha256']:raise ValueError('resume bundle source differs')
        state=torch.load(path/'trainer.pt',weights_only=True,map_location=self.device)
        if state['format_version']!=1 or state['task_schema']!=TASK_VERSION or state['config']!=asdict(self.config):raise ValueError('trainer checkpoint schema/plan differs')
        self.backend.load_state_dict(state['backend']);self.replay=ReplayBuffer.load(path/'replay.npz');self.reservoir=ReservoirBuffer.load(path/'reservoir.npz')
        if self.replay.spec!=self.spec or self.reservoir.spec!=self.spec or self.replay.capacity!=self.config.capacity or self.reservoir.capacity!=self.config.capacity or self.reservoir.opening_record!=self.config.reservoir_open:
            raise ValueError('buffer checkpoint differs from trial plan')
        if self.replay.insertions!=self.records or meta['step']!=self.records:raise ValueError('checkpoint fresh-record counts disagree')
        for name in ('episodes','per_role','best_response_visits','acting_counts','inserted_labels','evaluations','next_eval','timings','loss_stats','elapsed'):
            setattr(self,name,state[name])
        if self.per_role!=[self.records//2]*2 or self.episodes*2!=self.records or sum(self.acting_counts.values())!=self.records or self.inserted_labels!=self.reservoir.revisions:
            raise ValueError('checkpoint own-role/episode/label accounting differs')
        if self.backend.exact_average:self.backend.average.rebuild_tracker(self.reservoir)
        for name,rng_state in state['rng'].items():self.rng[name].bit_generator.state=rng_state
        torch.set_rng_state(state['torch_rng'].cpu())
        if self.device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        self.audit.restore_prefix(state['episode_audit'])
        if state['episode_audit']['episodes']!=self.episodes:raise ValueError('episode log/checkpoint count differs')
        self.last_saved=self.records
        # A finished segment report includes its final checkpoint IO time.
        result_path=self.root/'result.json'
        if result_path.exists():
            result=json.loads(result_path.read_text())
            if result['records']==self.records and result['source_sha256']==self.manifest['source_sha256']:
                self.elapsed=max(self.elapsed,result['elapsed_seconds'])

    def result(self,status):
        final=self.evaluations[-5:]
        passed=bool(status=='complete' and len(final)==5 and self.records==self.config.records
                    and all(e['average_ready'] and max(e['average_gaps'])<=.05 for e in final))
        return {'status':status,'task':self.config.task,'learner':self.config.learner,'seed':self.config.seed,'records':self.records,
                'physical_games':self.episodes,'episode_audit_bytes':self.episodes*EPISODE_DTYPE.itemsize,'per_role_records':self.per_role,'best_response_visits':self.best_response_visits,
                'acting_policy_counts':self.acting_counts,'counters':asdict(self.backend.counters),'reservoir_seen':self.reservoir.seen,
                'reservoir_size':self.reservoir.size,'reservoir_changes':self.reservoir.revisions,'formal_protocol':self.config.formal_protocol(),
                'average_final_five_passed':passed,'formal_gate_passed':passed and self.config.formal_protocol(),'evaluations':self.evaluations,'elapsed_seconds':self.elapsed_now(),
                'timings':self.timings,'loss_stats':self.loss_stats,'source_sha256':self.manifest['source_sha256'],
                'matrix_value':self.solution.value,'matrix_certificate_gap':self.solution.residual_gap}

    def run(self,*,until_records=None,stop_requested=lambda:False,verbose=True):
        stop=self.config.records if until_records is None else until_records
        if type(stop) is not int or not self.records<=stop<=self.config.records or stop%2:raise ValueError('invalid balanced segment endpoint')
        with self._run_lock():
            if self.last_saved>=0 and int(self.store.verify(self.store.latest())['step'])!=self.last_saved:
                raise RuntimeError('another writer advanced this run; reload before continuing')
            self._started=time.monotonic();status='complete'
            try:
                while self.records<stop:
                    requested=stop_requested()
                    if requested:status=requested if isinstance(requested,str) else 'interrupted';break
                    if self.elapsed_now()>=self.config.time_cap_seconds:status='budget_limited';break
                    self.store.ensure_space()
                    count=min(self.backend.counters.until_q_boundary(self.backend.config),stop-self.records,self.next_eval-self.records)
                    start=time.monotonic();self.collect(count);self.timings['collection']+=time.monotonic()-start
                    start=time.monotonic()
                    stats=self.backend.learn_after_fresh(count,count,self.replay,self.reservoir,self.rng['q_replay'],
                              epsilon=self.schedule.epsilon(self.records+count),supervised_rng=self.rng['supervised'])
                    self.timings['learning']+=time.monotonic()-start
                    for q in stats['q']:
                        self.loss_stats['q_updates']+=1;self.loss_stats['q_loss_sum']+=q['loss'];self.loss_stats['projected_targets']+=q['projected_targets']
                        for key,fn in (('raw_target_min',min),('raw_target_max',max)):
                            old=self.loss_stats[key];self.loss_stats[key]=q[key] if old is None else fn(old,q[key])
                    if stats['supervised'] is not None and stats['supervised']['loss'] is not None:
                        self.loss_stats['supervised_loss_updates']+=1;self.loss_stats['supervised_loss_sum']+=stats['supervised']['loss']
                    if self.records==self.next_eval or self.records==self.config.records:
                        if self.records==self.next_eval:self.next_eval+=self.config.eval_every
                        evaluation=self.evaluate();self.save(weights=True)
                        write_report(self.root/'progress.json',self.result('running'),reserve_bytes=self.store.reserve_bytes)
                        if verbose:print(json.dumps({'event':'nfsp-evaluation','task':self.config.task,'learner':self.config.learner,'seed':self.config.seed,**evaluation}),flush=True)
                if self.records<self.config.records and status=='complete':status='segment_paused'
                self.save(weights=self.records==self.config.records)
                if self.records==self.config.records:self.store.finalize()
                result=self.result(status);write_report(self.root/'result.json',result,reserve_bytes=self.store.reserve_bytes)
                return result
            finally:self.elapsed=self.elapsed_now();self._started=None

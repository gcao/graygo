"""Shared NFSP learning backend for compact tasks and board adapters.

Collectors own frozen episode actors and exact marginal behavior probabilities.
This backend owns online/target Q, independent supervised policy, optimizer work
and fresh-record cadence. There is no opponent identity input and no PPO critic.
One-step remains the default; Retrace is an explicitly configured comparison.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import torch
from .models import epsilon_greedy_distribution, masked_log_probs


@dataclass(frozen=True)
class NFSPConfig:
    batch: int = 512
    learning_start: int = 4096
    q_reuse: int = 1
    target_every: int = 12800
    learning_rate: float = 3e-4
    max_grad_norm: float = .5
    q_target: str = 'one-step'
    trace_length: int = 32
    trace_lambda: float = .95
    target_batch: int = 512

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in (self.batch,self.learning_start,self.target_every)):
            raise ValueError('positive integral NFSP cadence required')
        if self.learning_start % self.batch or self.target_every % self.batch:
            raise ValueError('learning start and refresh must align to fresh-record update boundaries')
        if type(self.q_reuse) is not int or self.q_reuse not in (1,4):
            raise ValueError('NFSP supports declared reuse 1 or 4')
        if self.q_target not in ('one-step','retrace'):
            raise ValueError('explicit one-step or Retrace target required')
        if type(self.trace_length) is not int or not 1<=self.trace_length<=32 or type(self.target_batch) is not int or self.target_batch<1:
            raise ValueError('one to 32 context rounds and positive target forward batch required')
        if not math.isfinite(self.trace_lambda) or not 0<=self.trace_lambda<=1:
            raise ValueError('finite trace lambda in [0,1] required')
        if not all(math.isfinite(x) and x > 0 for x in (self.learning_rate,self.max_grad_norm)):
            raise ValueError('finite positive optimizer settings required')

    @classmethod
    def matrix(cls,**kwargs):return cls(batch=128,**kwargs)


@dataclass
class OneStepBatch:
    inputs: dict
    legal: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_inputs: dict
    next_legal: torch.Tensor
    terminated: torch.Tensor

    @classmethod
    def from_replay(cls,data,device):
        def tensor(name,dtype):return torch.as_tensor(data[name],device=device,dtype=dtype)
        if 'value_mask' not in data:raise ValueError('replay Q eligibility is required')
        eligible=torch.as_tensor(data['value_mask'],device=device)
        if eligible.dtype!=torch.bool or eligible.shape!=(len(data['action']),) or not bool(eligible.all()):
            raise ValueError('ineligible replay start cannot enter a Q loss')
        return cls({'spatial':tensor('spatial',torch.float32),'globals':tensor('globals',torch.float32)},
                   tensor('legal',torch.bool),tensor('action',torch.long),tensor('reward',torch.float32),
                   {'spatial':tensor('next_spatial',torch.float32),'globals':tensor('next_globals',torch.float32)},
                   tensor('next_legal',torch.bool),tensor('terminated',torch.bool))

    def validate(self):
        n=len(self.actions)
        if n<1 or self.actions.dtype!=torch.long or self.legal.ndim!=2 or self.legal.shape[0]!=n:
            raise ValueError('invalid NFSP source/action dimensions')
        if self.legal.dtype!=torch.bool or self.next_legal.dtype!=torch.bool or self.terminated.dtype!=torch.bool:
            raise ValueError('boolean legality and terminal masks required')
        if self.next_legal.shape!=self.legal.shape or self.terminated.shape!=(n,) or self.rewards.shape!=(n,):
            raise ValueError('NFSP target dimensions disagree')
        if not bool(self.legal.any(-1).all()) or bool(((self.actions<0)|(self.actions>=self.legal.shape[-1])).any()):
            raise ValueError('nonterminal legal source actions required')
        if not bool(self.legal.gather(1,self.actions[:,None]).all()):raise ValueError('illegal replay action')
        if bool((self.next_legal.any(-1)==self.terminated).any()):raise ValueError('terminal/next mask disagree')
        if not bool(torch.isfinite(self.rewards).all()) or bool((self.rewards.abs()>1).any()):
            raise ValueError('finite bounded terminal-outcome rewards required')
        if bool(((~self.terminated)&(self.rewards!=0)).any()):
            raise ValueError('this terminal-outcome backend requires zero nonterminal rewards')
        if not self.inputs or set(self.inputs)!=set(self.next_inputs):raise ValueError('matching model inputs required')
        for inputs in (self.inputs,self.next_inputs):
            if any(x.shape[0]!=n or not bool(torch.isfinite(x).all()) for x in inputs.values()):
                raise ValueError('finite matching input batches required')


def _bounded(q,legal):
    if q.shape!=legal.shape or not bool(torch.isfinite(q[legal]).all()) or bool((q[legal].abs()>1+1e-6).any()):
        raise ValueError('bounded finite legal Q outputs required')


@torch.no_grad()
def one_step_targets(online,target,batch,epsilon):
    """Expected target-policy bootstrap; current online Q chooses pi, Q_bar values it.

Only real nonterminal successors are forwarded. A continuing replay boundary
bootstraps even if no following record was retained; true termination never does.
"""
    batch.validate()
    raw=batch.rewards.detach().clone()
    eps=torch.as_tensor(epsilon,device=raw.device,dtype=raw.dtype)
    if eps.ndim and eps.shape!=raw.shape:raise ValueError('epsilon must be scalar or one per replay start')
    if not bool(torch.isfinite(eps).all()) or bool(((eps<0)|(eps>1)).any()):raise ValueError('invalid epsilon')
    alive=~batch.terminated
    if bool(alive.any()):
        inputs={k:v[alive] for k,v in batch.next_inputs.items()}
        legal=batch.next_legal[alive]
        current=online(**inputs);frozen=target(**inputs)
        _bounded(current,legal);_bounded(frozen,legal)
        if eps.ndim:eps=eps[alive]
        pi=epsilon_greedy_distribution(current,legal,eps)
        raw[alive]+=(pi*frozen.masked_fill(~legal,0)).sum(-1)
    if not bool(torch.isfinite(raw).all()):raise ValueError('nonfinite raw Q target')
    projected=raw.clamp(-1,1)
    return projected,{'raw_target_min':float(raw.min()),'raw_target_max':float(raw.max()),
                     'projected_targets':int((raw!=projected).sum()),'targets':len(raw)}


def update_q(online,target,optimizer,batch,*,epsilon,max_grad_norm=.5,diagnostics=None):
    labels,stats=one_step_targets(online,target,batch,epsilon)
    return apply_q_update(online,optimizer,batch,labels,stats,max_grad_norm=max_grad_norm,diagnostics=diagnostics)


def apply_q_update(online,optimizer,batch,labels,stats,*,max_grad_norm=.5,diagnostics=None):
    predictions=online(**batch.inputs)
    _bounded(predictions,batch.legal)
    if diagnostics is not None:diagnostics(predictions.detach(),batch)
    selected=predictions.gather(1,batch.actions[:,None]).squeeze(1)
    loss=(selected-labels).square().mean()
    optimizer.zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(online.parameters(),max_grad_norm,error_if_nonfinite=True)
    optimizer.step()
    # A tabular control may project its direct Q parameters after an update.
    if hasattr(online,'project_'):online.project_()
    return dict(stats,loss=float(loss.detach()),gradient_norm=float(norm))


def update_average(policy,optimizer,data,*,device,max_grad_norm=.5):
    inputs={name:torch.tensor(data[name],device=device,dtype=torch.float32) for name in ('spatial','globals')}
    legal=torch.tensor(data['legal'],device=device,dtype=torch.bool)
    targets=torch.tensor(data['targets'],device=device,dtype=torch.float32)
    if legal.ndim!=2 or len(legal)<1 or targets.shape!=legal.shape:
        raise ValueError('matching supervised target dimensions required')
    if not bool(legal.any(-1).all()) or not bool(torch.isfinite(targets).all()) or bool((targets<0).any()):
        raise ValueError('finite legal supervised distributions required')
    if bool((targets.masked_select(~legal)!=0).any()) or not bool(torch.allclose(targets.sum(-1),torch.ones(len(targets),device=device),atol=1e-6,rtol=0)):
        raise ValueError('normalized frozen legal target distribution required')
    logits=policy(**inputs)
    if logits.shape!=legal.shape or not bool(torch.isfinite(logits[legal]).all()):raise ValueError('invalid policy logits')
    logp=masked_log_probs(logits,legal).masked_fill(~legal,0.)
    loss=-(targets.detach()*logp).sum(-1).mean()
    optimizer.zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),max_grad_norm,error_if_nonfinite=True)
    optimizer.step()
    return {'loss':float(loss.detach()),'targets':len(targets),'gradient_norm':float(norm)}


@dataclass
class NFSPCadence:
    fresh_records: int=0
    fresh_q: int=0
    fresh_ordinary: int=0
    q_updates: int=0
    supervised_updates: int=0
    q_loss_targets: int=0
    supervised_targets: int=0
    target_refreshes: int=0

    def validate(self):
        if any(type(v) is not int or v<0 for v in asdict(self).values()) or not self.fresh_ordinary<=self.fresh_q<=self.fresh_records:
            raise ValueError('invalid NFSP counters')

    def until_q_boundary(self,config):return config.batch-self.fresh_q%config.batch

    def advance(self,count,ordinary,config,*,q_eligible=None):
        """Call once per chronological chunk, never crossing an update boundary.

Returning a supervised cadence event with an empty reservoir does not create
backlogged updates. Target refresh occurs after the boundary's Q optimizer work.
"""
        self.validate()
        q_eligible=count if q_eligible is None else q_eligible
        if (type(count) is not int or type(ordinary) is not int or type(q_eligible) is not int
                or count<1 or not 0<=ordinary<=q_eligible<=count or q_eligible>self.until_q_boundary(config)
                or ordinary>config.batch-self.fresh_ordinary%config.batch):
            raise ValueError('fresh chunks must stop at each Q cadence boundary')
        old_ordinary=self.fresh_ordinary
        self.fresh_records+=count;self.fresh_q+=q_eligible;self.fresh_ordinary+=ordinary
        q_due=q_eligible>0 and self.fresh_q%config.batch==0 and self.fresh_q>=config.learning_start
        sl_due=self.fresh_ordinary//config.batch>old_ordinary//config.batch
        refresh=q_eligible>0 and self.fresh_q%config.target_every==0
        return q_due,sl_due,refresh


class NFSPBackend:
    """Shared one-step loss, independent networks, and serializable fresh clocks."""
    def __init__(self,q,average,config=NFSPConfig(),*,device='cpu'):
        self.config=config;self.device=torch.device(device)
        self.q=q.to(self.device);self.average=average.to(self.device)
        if {p.data_ptr() for p in self.q.parameters()} & {p.data_ptr() for p in self.average.parameters()}:
            raise ValueError('Q and average-policy parameters must be independent')
        self.target=deepcopy(q).to(self.device).eval().requires_grad_(False)
        self.q_optimizer=torch.optim.Adam(self.q.parameters(),lr=config.learning_rate)
        self.exact_average=bool(getattr(self.average,'exact_reservoir_average',False))
        self.average_optimizer=None if self.exact_average else torch.optim.Adam(self.average.parameters(),lr=config.learning_rate)
        self.counters=NFSPCadence();self.average_seeded=False
        from .sequence_targets import TraceDiagnostics
        self.trace=TraceDiagnostics() if config.q_target=='retrace' else None

    @property
    def average_ready(self):return self.average_seeded or self.counters.supervised_updates>0

    def learn_after_fresh(self,count,ordinary,replay,reservoir,rng,*,epsilon,q_eligible=None,supervised_rng=None,sample_observer=None,q_observer=None,q_transform=None,supervised_transform=None):
        if replay.spec!=reservoir.spec:raise ValueError('replay and reservoir schemas differ')
        # Validate available samples before advancing a recoverable clock.
        q_eligible=count if q_eligible is None else q_eligible
        if replay.insertions!=self.counters.fresh_records+count:
            raise ValueError('aggregate fresh records must match actual replay insertions exactly')
        if replay.q_insertions!=self.counters.fresh_q+q_eligible:
            raise ValueError('fresh Q-eligible records must match actual replay eligibility exactly')
        if replay.size!=min(replay.insertions,replay.capacity) or not 0<=replay.q_size<=min(replay.size,replay.q_insertions):
            raise ValueError('live replay/eligibility counts disagree')
        q_due,sl_due,refresh=self.counters.advance(count,ordinary,self.config,q_eligible=q_eligible)
        report={'q':[],'supervised':None,'target_refreshed':False}
        if q_due:
            for _ in range(self.config.q_reuse):
                data=(replay.sample_q(self.config.batch,rng) if self.config.q_target=='one-step' else
                      replay.sample_sequences(self.config.batch,self.config.trace_length,rng,q_only=True))
                if q_transform is not None:data=q_transform(data)
                starts=data if self.config.q_target=='one-step' else {name:array[:,0] for name,array in data.items()}
                if sample_observer is not None:sample_observer(starts)
                batch=OneStepBatch.from_replay(starts,self.device)
                callback=None if q_observer is None else lambda prediction,b: q_observer(starts,prediction,b,epsilon)
                if self.config.q_target=='one-step':
                    q_report=update_q(self.q,self.target,self.q_optimizer,batch,epsilon=epsilon,max_grad_norm=self.config.max_grad_norm,diagnostics=callback)
                else:
                    from .sequence_targets import update_retrace_q
                    trace_callback=lambda details: self.trace.observe(starts,details,self.counters.fresh_records,
                        self.counters.q_updates+self.counters.supervised_updates)
                    q_report=update_retrace_q(self.q,self.target,self.q_optimizer,data,device=self.device,epsilon=epsilon,
                        trace_lambda=self.config.trace_lambda,target_batch=self.config.target_batch,
                        max_grad_norm=self.config.max_grad_norm,diagnostics=callback,trace_observer=trace_callback)
                report['q'].append(q_report)
                self.counters.q_updates+=1;self.counters.q_loss_targets+=self.config.batch
        if sl_due and reservoir.size>=self.config.batch:
            if self.exact_average:
                self.average.publish_reservoir(reservoir)
                report['supervised']={'loss':None,'targets':reservoir.size,'method':'exact reservoir conditional label mean'}
            else:
                data=reservoir.sample(self.config.batch,rng if supervised_rng is None else supervised_rng)
                if supervised_transform is not None:data=supervised_transform(data)
                report['supervised']=update_average(self.average,self.average_optimizer,data,device=self.device,max_grad_norm=self.config.max_grad_norm)
            self.counters.supervised_updates+=1;self.counters.supervised_targets+=report['supervised']['targets']
        if refresh:
            self.target.load_state_dict(self.q.state_dict());self.counters.target_refreshes+=1
            report['target_refreshed']=True
        return report

    def state_dict(self):
        return {'clock_version':2,'config':asdict(self.config),'q':self.q.state_dict(),'target':self.target.state_dict(),
                'average':self.average.state_dict(),'q_optimizer':self.q_optimizer.state_dict(),
                'average_optimizer':None if self.average_optimizer is None else self.average_optimizer.state_dict(),'counters':asdict(self.counters),
                **({'trace_diagnostics':self.trace.state_dict()} if self.trace is not None else {}),
                **({'average_seeded':True} if self.average_seeded else {})}

    def load_state_dict(self,state):
        if state.get('clock_version')!=2:raise ValueError('NFSP clock/eligibility version differs; use the original source for historical continuation')
        if state['config']!=asdict(self.config):raise ValueError('NFSP resume settings differ')
        counters=NFSPCadence(**state['counters']);counters.validate()
        if counters.q_loss_targets!=counters.q_updates*self.config.batch or (not self.exact_average and counters.supervised_targets!=counters.supervised_updates*self.config.batch) or counters.supervised_targets<counters.supervised_updates*self.config.batch:
            raise ValueError('NFSP optimizer/target counts disagree')
        expected_q=max(0,counters.fresh_q//self.config.batch-self.config.learning_start//self.config.batch+1)*self.config.q_reuse
        if counters.q_updates!=expected_q or counters.supervised_updates>counters.fresh_ordinary//self.config.batch:
            raise ValueError('NFSP update cadence differs from fresh records')
        if counters.target_refreshes!=counters.fresh_q//self.config.target_every:
            raise ValueError('NFSP target clock differs from fresh records')
        for name in ('q','target','average'):getattr(self,name).load_state_dict(state[name])
        self.q_optimizer.load_state_dict(state['q_optimizer'])
        if self.average_optimizer is None:
            if state['average_optimizer'] is not None:raise ValueError('exact-average control cannot restore a neural optimizer')
        else:self.average_optimizer.load_state_dict(state['average_optimizer'])
        if self.trace is not None:self.trace.load_state_dict(state['trace_diagnostics'],counters.q_loss_targets)
        elif 'trace_diagnostics' in state:raise ValueError('one-step state cannot contain sequence diagnostics')
        self.counters=counters;self.average_seeded=bool(state.get("average_seeded",False))
        self.target.eval().requires_grad_(False)

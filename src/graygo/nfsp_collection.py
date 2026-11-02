"""Frozen native NFSP self-play with independent simultaneous commitments.

Logical mode and actual acting policy are separate. Snapshot parameters and
exploration remain fixed within an episode. Replay receives executed actions;
the reservoir receives eligible frozen greedy sets, including exploratory visits.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
import numpy as np
import torch
from .engine import BLACK,WHITE
from .encoding import encode
from .models import greedy_distribution,epsilon_greedy_distribution,masked_log_probs
from .schedules import EpisodeChoice
from .sampling import draw_actions
from .strategies import GrayGoGame


class BoardGame(GrayGoGame):
    observation=staticmethod(encode)


class RuleBoardGame(BoardGame):
    @staticmethod
    def observation(state,player):
        from .rule_state import RuleObservation
        return RuleObservation(state,player)


class FrozenPolicies:
    def __init__(self,q,average,*,version,average_ready):
        if type(version) is not int or not 0<=version<2**63:raise ValueError('bounded policy snapshot version required')
        self.q=deepcopy(q).eval().requires_grad_(False)
        self.average=(average.inference_copy() if hasattr(average,'inference_copy') else deepcopy(average)).eval().requires_grad_(False)
        self.version=version;self.average_ready=bool(average_ready)
        q_devices={p.device for p in list(self.q.parameters())+list(self.q.buffers())};avg_devices={p.device for p in list(self.average.parameters())+list(self.average.buffers())}
        if len(q_devices)!=1 or q_devices!=avg_devices:raise ValueError('snapshot models must share one device')
        self.device=next(iter(q_devices))

    def state_dict(self):
        return {'q':self.q.state_dict(),'average':self.average.state_dict(),'version':self.version,'average_ready':self.average_ready}

    def load_state_dict(self,state):
        if self.version!=state['version'] or self.average_ready!=state['average_ready']:
            raise ValueError('snapshot identity/availability changed')
        self.q.load_state_dict(state['q']);self.average.load_state_dict(state['average'])


@dataclass(frozen=True)
class Commitment:
    action: int
    probability: float
    greedy: np.ndarray|None
    logical_best_response: bool
    acting_policy: str
    policy_version: int
    epsilon: float
    forced: bool


class EpisodeActor:
    def __init__(self,snapshot,choice,rng):
        if choice.acting_policy not in ('q','average','uniform') or (choice.best_response_mode and choice.acting_policy!='q'):
            raise ValueError('logical best-response mode must act through Q')
        if choice.acting_policy=='average' and not snapshot.average_ready:raise ValueError('average actor has not been trained')
        if not 0<=choice.epsilon<=1:raise ValueError('invalid frozen epsilon')
        self.snapshot,self.choice,self.rng=snapshot,choice,rng

    @property
    def version(self):return self.snapshot.version*2+(self.choice.acting_policy=='average')

    @torch.no_grad()
    def commit(self,observation,*,label_eligible=False,forced_action=None,supplemental=False):
        legal=torch.tensor(observation.legal[None],device=self.snapshot.device,dtype=torch.bool)
        if not bool(legal.any()):raise ValueError('cannot commit a terminal action')
        inputs={'spatial':torch.tensor(observation.spatial[None],device=self.snapshot.device,dtype=torch.float32),
                'globals':torch.tensor(observation.global_features[None],device=self.snapshot.device,dtype=torch.float32)}
        greedy=None;choice=self.choice
        if label_eligible and (supplemental or not choice.best_response_mode):
            raise ValueError('label eligibility disagrees with logical mode/coverage origin')
        if choice.acting_policy=='q' or label_eligible:
            q=self.snapshot.q(**inputs)
            if q.shape!=legal.shape or not bool(torch.isfinite(q[legal]).all()) or bool((q[legal].abs()>1+1e-6).any()):
                raise ValueError('frozen Q must produce bounded legal action values')
            if label_eligible:
                greedy=(greedy_distribution(q,legal)[0]>0).cpu().numpy().copy();greedy.flags.writeable=False
        if choice.acting_policy=='q':probabilities=epsilon_greedy_distribution(q,legal,choice.epsilon)
        elif choice.acting_policy=='average':
            logits=self.snapshot.average(**inputs)
            if logits.shape!=legal.shape or bool(torch.isnan(logits[legal]).any()) or bool(torch.isposinf(logits[legal]).any()) or not bool(torch.isfinite(logits.masked_fill(~legal,-torch.inf)).any(-1).all()):raise ValueError('invalid average-policy logits')
            probabilities=masked_log_probs(logits,legal).exp()
        else:probabilities=legal.float()/legal.sum(-1,keepdim=True)
        p=probabilities[0].double().cpu().numpy();p/=p.sum()
        if forced_action is not None:
            if not supplemental or isinstance(forced_action,(bool,np.bool_)) or not isinstance(forced_action,(int,np.integer)):
                raise ValueError('only supplemental episodes may force an integral root action')
            action=int(forced_action)
            if not 0<=action<len(p) or not observation.legal[action]:raise ValueError('illegal forced root action')
            probability=1.
        else:
            drawn,logged=draw_actions(p[None],self.rng)
            action=int(drawn[0]);probability=float(logged[0])
        return Commitment(action,probability,greedy,choice.best_response_mode,choice.acting_policy,self.version,choice.epsilon,forced_action is not None)

    def state_dict(self):return {'choice':asdict(self.choice),'rng':deepcopy(self.rng.bit_generator.state),'snapshot_version':self.snapshot.version}

    @classmethod
    def restore(cls,snapshot,state):
        if snapshot.version!=state['snapshot_version']:raise ValueError('wrong frozen episode snapshot')
        rng=np.random.default_rng();rng.bit_generator.state=state['rng']
        return cls(snapshot,EpisodeChoice(**state['choice']),rng)


class JointEpisode:
    def __init__(self,game,root,black,white,*,episode_id,supplemental=False,forced_root=None):
        if type(episode_id) is not int or not 0<=episode_id<2**63:raise ValueError('bounded nonnegative episode ID required')
        if supplemental!=(forced_root is not None):raise ValueError('supplemental episodes require one declared forced root')
        if forced_root is not None and (len(forced_root)!=2 or forced_root[0] not in (BLACK,WHITE)):
            raise ValueError('forced root must name a role and action')
        self.game,self.state=game,root;self.actors={BLACK:black,WHITE:white}
        self.episode_id=episode_id;self.supplemental=bool(supplemental);self.forced_root=forced_root;self.steps=0
        if black.rng is white.rng:raise ValueError('players require independent private action RNG streams')
        self.log=[]

    def step(self,replay,reservoir,reservoir_rng,schedule,*,record,opening_enabled=True,on_reservoir_change=None,label_rule='greedy',precommitted=None,on_record=None):
        if self.game.terminal(self.state):raise ValueError('episode is complete')
        if label_rule not in ('greedy','executed'):raise ValueError('invalid averaging-label control')
        if type(record) is not int or record<0:raise ValueError('nonnegative aggregate record clock required')
        if record!=replay.insertions or replay.spec!=reservoir.spec:
            raise ValueError('joint record clock/schema disagrees with storage')
        if reservoir.opening_record!=schedule.reservoir_open:
            raise ValueError('reservoir storage and acting schedule disagree on opening')
        observations={role:self.game.observation(self.state,role) for role in (BLACK,WHITE)}
        commitments={} if precommitted is None else dict(precommitted)
        if precommitted is not None and set(commitments)!={BLACK,WHITE}:raise ValueError('both simultaneous commitments required')
        for role in (BLACK,WHITE):
            actor=self.actors[role]
            eligible=schedule.label_eligible(actor.choice,record,supplemental=self.supplemental,opening_enabled=opening_enabled)
            forced=self.forced_root[1] if self.steps==0 and self.forced_root is not None and self.forced_root[0]==role else None
            if precommitted is None:commitments[role]=actor.commit(observations[role],label_eligible=eligible,forced_action=forced,supplemental=self.supplemental)
            else:
                own=commitments[role]
                if own.policy_version!=actor.version or own.logical_best_response!=actor.choice.best_response_mode or own.acting_policy!=actor.choice.acting_policy or (own.greedy is not None)!=eligible or own.forced!=(forced is not None):raise ValueError('precommitted actor/mode/label context differs')
        # Both commitments are complete before transition sees either action.
        next_state=self.game.transition(self.state,commitments[BLACK].action,commitments[WHITE].action)
        done=self.game.terminal(next_state)
        reward=float(self.game.utility(next_state)) if done else 0.
        if not np.isfinite(reward) or not -1<=reward<=1:raise ValueError('invalid terminal utility')
        for index,role in enumerate((BLACK,WHITE)):
            own,other=commitments[role],commitments[3-role]
            # Forcing our action gives a conditional Q(s,a) sample. Forcing the
            # opponent changes our transition kernel at the pre-forcing root.
            q_eligible=not (self.steps==0 and self.forced_root is not None and self.forced_root[0]!=role)
            next_observation=self.game.observation(next_state,role)
            replay.add(observations[role],own.action,reward if role==BLACK else -reward,next_observation,done,
                       behavior_prob=own.probability,stream=2*self.episode_id+index,step=self.steps,
                       opponent_version=other.policy_version,policy_version=own.policy_version,mode=int(own.logical_best_response),
                       actor_mask=not self.supplemental,value_mask=q_eligible,supplemental=self.supplemental)
            if own.greedy is not None:
                label=own.greedy
                if label_rule=='executed':
                    label=np.zeros_like(own.greedy);label[own.action]=True
                changed=reservoir.offer_index(observations[role],label,record=record+index,policy_version=own.policy_version,
                                rng=reservoir_rng,best_response_mode=own.logical_best_response,
                                supplemental=self.supplemental,enabled=opening_enabled)
                if changed is not None and on_reservoir_change is not None:on_reservoir_change(reservoir,changed)
            if on_record is not None:on_record(role=role,record=record+index,ordinary=int(not self.supplemental),q_eligible=int(q_eligible))
        event={'step':self.steps,'record':record,'actions':[commitments[r].action for r in (BLACK,WHITE)],
               'probabilities':[commitments[r].probability for r in (BLACK,WHITE)],
               'modes':[commitments[r].logical_best_response for r in (BLACK,WHITE)],
               'acting_policies':[commitments[r].acting_policy for r in (BLACK,WHITE)],
               'versions':[commitments[r].policy_version for r in (BLACK,WHITE)],
               'epsilons':[commitments[r].epsilon for r in (BLACK,WHITE)],
               'supplemental':self.supplemental,'q_eligible':[not (self.steps==0 and self.forced_root is not None and self.forced_root[0]!=role) for role in (BLACK,WHITE)],'terminated':done,'reward_black':reward}
        self.state=next_state;self.steps+=1;self.log.append(event)
        return event

    def state_dict(self,encode_state):
        return {'state':encode_state(self.state),'episode_id':self.episode_id,'supplemental':self.supplemental,
                'forced_root':None if self.forced_root is None else list(self.forced_root),'steps':self.steps,
                'actors':{str(role):actor.state_dict() for role,actor in self.actors.items()},'log':deepcopy(self.log)}

    @classmethod
    def restore(cls,game,state,decode_state,snapshots):
        actors={int(role):EpisodeActor.restore(snapshots[item['snapshot_version']],item) for role,item in state['actors'].items()}
        if set(actors)!={BLACK,WHITE}:raise ValueError('episode checkpoint must contain both players')
        forced=None if state['forced_root'] is None else tuple(state['forced_root'])
        result=cls(game,decode_state(state['state']),actors[BLACK],actors[WHITE],episode_id=state['episode_id'],
                   supplemental=state['supplemental'],forced_root=forced)
        if type(state['steps']) is not int or state['steps']<0 or len(state['log'])!=state['steps']:
            raise ValueError('invalid episode step/log checkpoint')
        result.steps=state['steps'];result.log=deepcopy(state['log'])
        return result


@torch.no_grad()
def commit_many(requests):
    """Batch public inference by immutable version/policy; sample private RNGs."""
    if not requests:return []
    groups={};output=[None]*len(requests)
    for index,(actor,observation,eligible) in enumerate(requests):
        if eligible and not actor.choice.best_response_mode:raise ValueError('average fallback cannot create labels')
        key=(actor.snapshot.version,actor.choice.acting_policy)
        group=groups.setdefault(key,[])
        if group and requests[group[0]][0].snapshot is not actor.snapshot:raise ValueError('one version cannot identify different snapshots')
        group.append(index)
    for key in sorted(groups):
        indices=groups[key];actors=[requests[i][0] for i in indices];observations=[requests[i][1] for i in indices]
        snapshot=actors[0].snapshot;device=snapshot.device;kind=key[1]
        legal=torch.tensor(np.stack([o.legal for o in observations]),device=device,dtype=torch.bool)
        inputs={'spatial':torch.tensor(np.stack([o.spatial for o in observations]),device=device,dtype=torch.float32),
                'globals':torch.tensor(np.stack([o.global_features for o in observations]),device=device,dtype=torch.float32)}
        greedy=None
        if kind=='q':
            q=snapshot.q(**inputs)
            if q.shape!=legal.shape or not bool(torch.isfinite(q[legal]).all()) or bool((q[legal].abs()>1+1e-6).any()):raise ValueError('invalid batched legal Q values')
            eps=torch.tensor([a.choice.epsilon for a in actors],device=device,dtype=q.dtype)
            probabilities=epsilon_greedy_distribution(q,legal,eps)
            greedy=(greedy_distribution(q,legal)>0).cpu().numpy()
        elif kind=='average':
            logits=snapshot.average(**inputs)
            if logits.shape!=legal.shape or bool(torch.isnan(logits[legal]).any()) or bool(torch.isposinf(logits[legal]).any()):raise ValueError('invalid batched actor logits')
            probabilities=masked_log_probs(logits,legal).exp()
        elif kind=='uniform':probabilities=legal.float()/legal.sum(-1,keepdim=True)
        else:raise ValueError('unknown batched acting policy')
        probabilities=probabilities.double().cpu().numpy()
        for local,index in enumerate(indices):
            actor,_,eligible=requests[index];drawn,mu=draw_actions(probabilities[local:local+1],actor.rng)
            label=None if not eligible else greedy[local].copy()
            if label is not None:label.flags.writeable=False
            output[index]=Commitment(int(drawn[0]),float(mu[0]),label,actor.choice.best_response_mode,kind,actor.version,actor.choice.epsilon,False)
    return output

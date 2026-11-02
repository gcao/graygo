"""Opaque delayed-credit/meaningful-branch tasks and supported-coverage schedule.

State IDs are randomly permuted independently of depth, branch and outcome.
The public state structures are engine data; observations expose only one-hot
IDs and a countdown. No branch, return or reward-sign feature enters the model.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from collections import Counter
import numpy as np
from .encoding import Observation
from .engine import BLACK,WHITE
from .buffers import ObservationSpec

TASK_VERSION='graygo/opaque-tasks/1'


class OpaqueEncoding:
    def __init__(self,states,seed,actions):
        states=list(states)
        if len(set(states))!=len(states):raise ValueError('duplicate public task states')
        permutation=np.random.default_rng(seed).permutation(2*len(states))
        self.ids={(state,role):int(permutation[2*i+j]) for i,state in enumerate(states) for j,role in enumerate((BLACK,WHITE))}
        self.features=len(permutation);self.action_count=actions;self.seed=seed
        self.spec=ObservationSpec((self.features,),1,actions,TASK_VERSION)
        # Enumerated semantic keys and exact permuted IDs are a reproducible codebook.
        entries=[{'state':list(state),'role':role,'id':self.ids[state,role]} for state in states for role in (BLACK,WHITE)]
        self.manifest={'version':TASK_VERSION,'seed':seed,'entries':entries,'actions':actions}
        self.digest=hashlib.sha256(json.dumps(self.manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def encode(self,key,player,remaining_fraction,legal):
        if player not in (BLACK,WHITE):raise ValueError('invalid perspective')
        x=np.zeros(self.features,dtype=np.uint8);mask=np.zeros(self.action_count,dtype=bool)
        if key is not None:x[self.ids[key,player]]=1
        mask[list(legal)]=True
        g=np.array([remaining_fraction],dtype=np.float32)
        for array in (x,g,mask):array.flags.writeable=False
        return Observation(x,g,mask)


@dataclass(frozen=True)
class DelayedState:
    horizon: int
    controller: int
    depth: int=0
    branch: int=-1

    def __post_init__(self):
        if type(self.horizon) is not int or self.horizon<2 or self.controller not in (BLACK,WHITE):
            raise ValueError('invalid delayed state configuration')
        if type(self.depth) is not int or not 0<=self.depth<=self.horizon or self.branch not in ((-1,) if self.depth==0 else (0,1)):
            raise ValueError('invalid delayed state depth/branch')


class DelayedGame:
    def __init__(self,*,horizons=(50,100,200),equivalent_actions=40,codebook_seed=0):
        self.horizons=tuple(horizons);self.equivalent_actions=equivalent_actions
        if not self.horizons or len(set(self.horizons))!=len(self.horizons) or any(type(h) is not int or h<2 for h in self.horizons):
            raise ValueError('distinct horizons of at least two rounds required')
        if type(equivalent_actions) is not int or equivalent_actions<1:raise ValueError('positive intermediate action count required')
        keys=[]
        for h in self.horizons:
            for controller in (BLACK,WHITE):
                keys.append((h,controller,0,-1))
                keys.extend((h,controller,depth,branch) for depth in range(1,h) for branch in (0,1))
        self.encoding=OpaqueEncoding(keys,codebook_seed,max(2,equivalent_actions));self.spec=self.encoding.spec

    def root(self,horizon=50,controller=BLACK):
        if horizon not in self.horizons or controller not in (BLACK,WHITE):raise ValueError('unknown delayed-task root')
        return DelayedState(horizon,controller)

    @staticmethod
    def terminal(state):return state.depth==state.horizon

    def actions(self,state,player):
        if player not in (BLACK,WHITE):raise ValueError('invalid role')
        if self.terminal(state):return ()
        if state.horizon not in self.horizons:raise ValueError('unknown delayed-task horizon')
        if state.depth>0:return tuple(range(self.equivalent_actions))
        return (0,1) if player==state.controller else (0,)

    def transition(self,state,black,white):
        if black not in self.actions(state,BLACK) or white not in self.actions(state,WHITE):raise ValueError('illegal delayed-task joint action')
        branch=(black if state.controller==BLACK else white) if state.depth==0 else state.branch
        return DelayedState(state.horizon,state.controller,state.depth+1,branch)

    def utility(self,state):
        if not self.terminal(state):raise ValueError('utility requested before terminal')
        return (1. if state.branch==0 else -1.)*(1 if state.controller==BLACK else -1)

    def observation(self,state,player):
        key=None if self.terminal(state) else (state.horizon,state.controller,state.depth,state.branch)
        return self.encoding.encode(key,player,(state.horizon-state.depth)/state.horizon,self.actions(state,player))


@dataclass(frozen=True)
class BranchState:
    controller: int
    depth: int=0
    branch: int=-1
    black: int=-1
    white: int=-1

    def __post_init__(self):
        if self.controller not in (BLACK,WHITE) or type(self.depth) is not int or self.depth not in (0,1,2):
            raise ValueError('invalid branching state')
        if self.branch not in ((-1,) if self.depth==0 else (0,1)):
            raise ValueError('invalid root branch')
        if any(move not in ((0,1) if self.depth==2 else (-1,)) for move in (self.black,self.white)):
            raise ValueError('invalid stored terminal moves')


class BranchingGame:
    """A root branch choice followed by a meaningful simultaneous matrix game.

Canonical controller values are -1/7 and -7/22. The better root branch is 0.
Swapping the controller exchanges roles and negates/transposes final payoffs.
"""
    payoffs=(np.array([[1.,-1.],[-1.,.5]]),np.array([[.5,-1.],[-1.,.25]]))

    def __init__(self,codebook_seed=0):
        for matrix in self.payoffs:matrix.flags.writeable=False
        keys=[(controller,0,-1) for controller in (BLACK,WHITE)]
        keys.extend((controller,1,branch) for controller in (BLACK,WHITE) for branch in (0,1))
        self.encoding=OpaqueEncoding(keys,codebook_seed,2);self.spec=self.encoding.spec

    @staticmethod
    def root(controller=BLACK):
        if controller not in (BLACK,WHITE):raise ValueError('invalid controller')
        return BranchState(controller)

    @staticmethod
    def terminal(state):return state.depth==2

    def actions(self,state,player):
        if player not in (BLACK,WHITE):raise ValueError('invalid role')
        if self.terminal(state):return ()
        return (0,) if state.depth==0 and player!=state.controller else (0,1)

    def transition(self,state,black,white):
        if black not in self.actions(state,BLACK) or white not in self.actions(state,WHITE):raise ValueError('illegal branching-task joint action')
        if state.depth==0:return BranchState(state.controller,1,black if state.controller==BLACK else white)
        return BranchState(state.controller,2,state.branch,black,white)

    def utility(self,state):
        if not self.terminal(state):raise ValueError('utility requested before terminal')
        matrix=self.payoffs[state.branch]
        return float(matrix[state.black,state.white] if state.controller==BLACK else -matrix[state.white,state.black])

    def observation(self,state,player):
        key=None if self.terminal(state) else (state.controller,state.depth,state.branch)
        return self.encoding.encode(key,player,(2-state.depth)/2,self.actions(state,player))


@dataclass(frozen=True)
class CoverageEpisode:
    horizon: int
    controller: int
    forced_root: int|None

    @property
    def supplemental(self):return self.forced_root is not None


class BalancedCoverage:
    """Balanced round allocation and randomized ten-episode coverage blocks.

A macrocycle gives the same total rounds to every horizon/controller. Each
block has eight ordinary, one favorable and one unfavorable root episode.
"""
    def __init__(self,horizons=(50,100,200),*,seed=0):
        self.horizons=tuple(horizons)
        if not self.horizons or len(set(self.horizons))!=len(self.horizons) or any(type(h) is not int or h<2 for h in self.horizons):
            raise ValueError('valid distinct coverage horizons required')
        multiple=math.lcm(*self.horizons)
        if sum(multiple//h for h in self.horizons)>10000:raise ValueError('coverage macrocycle too large')
        self.groups=[(h,role) for h in self.horizons for role in (BLACK,WHITE) for _ in range(multiple//h)]
        self.rng=np.random.default_rng(seed);self.blocks=[];self.episodes=[];self.completed=0

    def next(self):
        if not self.episodes:
            if not self.blocks:self.blocks=[self.groups[int(i)] for i in self.rng.permutation(len(self.groups))]
            h,role=self.blocks.pop(0)
            forced=[None]*8+[0,1]
            self.episodes=[CoverageEpisode(h,role,forced[int(i)]) for i in self.rng.permutation(10)]
        self.completed+=1
        return self.episodes.pop(0)

    def state_dict(self):
        return {'horizons':list(self.horizons),'rng':self.rng.bit_generator.state,'blocks':[list(x) for x in self.blocks],
                'episodes':[e.__dict__.copy() for e in self.episodes],'completed':self.completed}

    def load_state_dict(self,state):
        if state['horizons']!=list(self.horizons):raise ValueError('coverage horizons changed on resume')
        if type(state['completed']) is not int or state['completed']<0:raise ValueError('invalid coverage counter')
        blocks=[tuple(x) for x in state['blocks']];episodes=[CoverageEpisode(**x) for x in state['episodes']]
        expected_blocks=(-((state['completed']+9)//10))%len(self.groups)
        if len(blocks)!=expected_blocks or len(episodes)!=(-state['completed'])%10 or Counter(blocks)-Counter(self.groups):
            raise ValueError('invalid coverage schedule state')
        if any((e.horizon,e.controller) not in self.groups or e.forced_root not in (None,0,1) for e in episodes):raise ValueError('invalid coverage episode')
        if episodes:
            if len({(e.horizon,e.controller) for e in episodes})!=1 or Counter(e.forced_root for e in episodes)-Counter({None:8,0:1,1:1}):
                raise ValueError('invalid coverage block composition')
        self.rng.bit_generator.state=state['rng'];self.blocks=blocks;self.episodes=episodes;self.completed=state['completed']

"""Finite task adapters with complete public-state identity and exact inventories."""
from dataclasses import asdict
import json
import numpy as np
from .buffers import ObservationSpec
from .engine import BLACK,WHITE,State,Rules,resolve
from .models import QNet,PolicyNet,ModelConfig
from .nfsp_collection import BoardGame
from .public_game import PublicGameSolver
from .tabular_average import TabularAverage
from .task_models import CompactQ,CompactPolicy,TabularQ
from .tasks import DelayedGame,DelayedState,BranchingGame,BranchState,OpaqueEncoding

ADAPTER_VERSION='graygo/nfsp-multiround-adapter/1'


class IndexedBoard:
    def __init__(self,base,states,seed):
        self.base=base;self.lookup={state:i for i,state in enumerate(states)}
        self.encoding=OpaqueEncoding([(i,) for i in range(len(states))],seed,states[0].rules.points+1)
        self.spec=self.encoding.spec
    def __getattr__(self,name):return getattr(self.base,name)
    def observation(self,state,player):
        key=None if self.terminal(state) else (self.lookup[state],)
        return self.encoding.encode(key,player,(state.rules.max_rounds-state.rounds_played)/state.rules.max_rounds,self.actions(state,player))


class TaskAdapter:
    def __init__(self,config):
        self.config=config;self.board=config.task in ('gray-pure','gray-mixed')
        self.delayed=config.task in ('delayed','delayed-forced')
        self.root_tags=[];self.values=[]
        if self.delayed:
            self.game=DelayedGame(horizons=config.horizons,equivalent_actions=1 if config.task=='delayed-forced' else config.equivalent_actions,codebook_seed=config.seed+501)
            self.roots=[self.game.root(h,p) for h in config.horizons for p in (BLACK,WHITE)]
            self.root_tags=[f'h{r.horizon}/controller{r.controller}' for r in self.roots]
            self.states=[DelayedState(*key) for key,role in self.game.encoding.ids if role==BLACK]
            self.values=[1. if root.controller==BLACK else -1. for root in self.roots]
            self.oracle={'method':'analytic terminal-outcome chain','values':self.values,'matrix_certificate_gap':0.}
        else:
            if self.board:
                self.game=BoardGame()
                if config.task=='gray-pure':root=State(Rules(3,1))
                else:
                    root=State(Rules(3,6))
                    for b,w in ((1,7),(6,4),(2,5),(9,3)):root=resolve(root,b,w).state
                self.roots=[root];self.root_tags=[config.task]
            elif config.task=='branching':
                self.game=BranchingGame(config.seed+501);self.roots=[self.game.root(p) for p in (BLACK,WHITE)]
                self.root_tags=[f'branching/controller{r.controller}' for r in self.roots]
            else:raise ValueError('unknown multiround task')
            solver=PublicGameSolver(self.game,max_nodes=config.max_nodes)
            self.values=[solver.solve(r)[0] for r in self.roots]
            self.states=sorted((s for s in solver.cache if not self.game.terminal(s)),key=lambda s:json.dumps(self.encode_state(s),sort_keys=True))
            self.oracle={'method':'exhaustive public-state finite game','values':self.values,'nodes':solver.nodes,'matrix_certificate_gap':solver.max_residual_gap}
        if self.board:
            if config.learner=='tabular':self.game=IndexedBoard(self.game,self.states,config.seed+501)
            self.spec=self.game.spec if config.learner=='tabular' else ObservationSpec.board(3)
        else:self.spec=self.game.spec
        self.index={state:i for i,state in enumerate(self.states)}
        self.observations=[self.game.observation(state,role) for state in self.states for role in (BLACK,WHITE)]
        self.actions=self.spec.actions
        self.flat=len(self.spec.spatial_shape)==1
        self._bins={}
        if self.flat:
            self._id_state=np.full(self.spec.spatial_shape[0],-1,dtype=np.int32)
            self._id_role=np.full(self.spec.spatial_shape[0],-1,dtype=np.int8)
            for i,obs in enumerate(self.observations):
                if obs.spatial.sum()!=1:raise ValueError('nonterminal table observation must have one state ID')
                token=int(obs.spatial.argmax());self._id_state[token]=i//2;self._id_role[token]=i%2
        else:
            for i,obs in enumerate(self.observations):
                key=self._observation_key(obs.spatial,obs.global_features,i%2)
                if key in self._bins:raise ValueError('full-state board encoding aliases distinct public states')
                self._bins[key]=i//2

    def models(self):
        c=self.config
        if c.learner=='tabular':return TabularQ(self.spec.spatial_shape[0],self.actions),TabularAverage(self.spec.spatial_shape[0],self.actions)
        if self.board:
            shape=ModelConfig(c.width,c.blocks,c.norm_groups)
            return QNet(shape),PolicyNet(shape)
        return CompactQ(self.spec.spatial_shape[0],self.spec.globals,self.actions,c.hidden),CompactPolicy(self.spec.spatial_shape[0],self.spec.globals,self.actions,c.hidden)

    def encode_state(self,state):return state.to_dict() if self.board else asdict(state)
    def decode_state(self,data):
        if self.board:return State.from_dict(data)
        return DelayedState(**data) if self.delayed else BranchState(**data)

    def manifest(self):
        result={'version':ADAPTER_VERSION,'task':self.config.task,'observation_spec':asdict(self.spec),
                'roots':[{'tag':tag,'state':self.encode_state(s),'value':v} for tag,s,v in zip(self.root_tags,self.roots,self.values)],
                'states':[self.encode_state(s) for s in self.states],'oracle':self.oracle,
                'early_value_states':'root legal action values plus target-policy values at depths 1, 2, 5 on both delayed branches; all nonterminal states in two-round controls',
                'late_value_states':'last nonterminal depth on both delayed branches',
                'q_value_continuation':'own current epsilon-greedy online-Q policy against the frozen current native opponent episode mixture; public-prefix-conditioned component weights; own-role utilities',
                'augmentation':'fixed A2 roots; no trajectory augmentation'}
        if hasattr(self.game,'encoding'):
            result['codebook']=self.game.encoding.manifest;result['codebook_sha256']=self.game.encoding.digest
        return result

    @staticmethod
    def _observation_key(x,g,role):return (role,np.packbits(np.asarray(x,dtype=np.uint8).reshape(-1),bitorder='little').tobytes(),np.asarray(g,dtype=np.float32).tobytes())

    def sample_bins(self,data):
        roles=(data['stream']%2).astype(np.int64)
        if self.flat:
            ids=data['spatial'].argmax(-1)
            if not np.all(data['spatial'].sum(-1)==1) or not np.array_equal(self._id_role[ids],roles):raise ValueError('sample identity/role disagrees with trajectory')
            states=self._id_state[ids]
        else:states=np.array([self._bins[self._observation_key(x,g,int(p))] for x,g,p in zip(data['spatial'],data['globals'],roles)])
        if (states<0).any():raise ValueError('unknown sampled public state')
        return states,roles

    def new_root(self,coverage):
        if coverage is None:return 0,None
        e=coverage.next()
        root=self.game.root(e.horizon,e.controller) if self.delayed else self.game.root(e.controller)
        root_index=self.roots.index(root)
        return root_index,(e.controller,e.forced_root) if e.supplemental else None

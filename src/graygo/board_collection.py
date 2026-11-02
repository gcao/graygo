"""Batched real-board NFSP collection; no hidden commitments enter inference."""
from copy import deepcopy
import numpy as np
from .engine import BLACK,WHITE,GRAY,EMPTY,State,resolve,neighbors
from .evaluation import EmptyStarts,seed_for
from .nfsp_collection import RuleBoardGame,FrozenPolicies,EpisodeActor,JointEpisode,commit_many

COLLECTOR_VERSION='graygo/board-collector/2'


class _TelemetryGame(RuleBoardGame):
    def transition(self,state,black,white):
        self.last=resolve(state,black,white);return self.last.state


class BoardCollector:
    def __init__(self,rules,envs,seed,*,starts=None,on_episode=None):
        if type(envs) is not int or envs<1 or type(seed) is not int or seed<0:raise ValueError('positive env count and seed required')
        self.rules,self.envs,self.seed=rules,envs,seed;self.starts=starts or EmptyStarts();self.on_episode=on_episode
        self.game=_TelemetryGame();self.active=[None]*envs;self.start_records=[None]*envs;self.start_families=[None]*envs;self.births=[{} for _ in range(envs)]
        self.cursor=self.records=self.started=self.completed=0
        self.rng=[{name:np.random.default_rng(seed_for(seed,'collector',env,name)) for name in ('mode_b','mode_w','action_b','action_w')} for env in range(envs)]
        self.metrics={'start_counts':{'empty':0,'gray':0},'starting_gray_counts':[0]*(rules.points+1),'family_starts':{},'physical_rounds':0,'collisions':0,'history_match_rounds':0,'restriction_point_rounds':[0,0],
            'captured_stones':[0]*4,'suicidal_stones':[0]*4,'created_gray':0,'removed_gray_lifetimes':[0]*(rules.max_rounds+1),
            'terminal_gray_ages':[0]*(rules.max_rounds+1),'terminal_gray_stones':0,'terminal_empty_points':0,'gray_boundary_empty_points':0,
            'fractional_bias_empty_points':0,'episode_lengths':[0]*(rules.max_rounds+1),'endings':{'mutual_pass':0,'round_limit':0},'outcomes':[0,0,0],
            'per_role_records':[0,0],'best_response_visits':[0,0],'acting_counts':{'q':0,'average':0,'uniform':0},'completed_episode_modes':[[0]*4 for _ in range(3)]}

    def _start(self,env,snapshot,schedule,record,uniform_comparison):
        start_seed=seed_for(self.seed,'start',self.started)
        if hasattr(self.starts,'sample_with_family'):root,family=self.starts.sample_with_family(start_seed,self.rules)
        else:root=self.starts.sample(start_seed,self.rules);family=None
        if family is not None and (not isinstance(family,str) or len(family)!=64 or any(c not in '0123456789abcdef' for c in family)):raise ValueError('invalid curriculum family identity')
        if root.rules!=self.rules or root.done:raise ValueError('invalid board curriculum root')
        gray=root.board.count(GRAY);self.start_families[env]=family
        self.metrics['start_counts']['gray' if gray else 'empty']+=1;self.metrics['starting_gray_counts'][gray]+=1
        if family is not None:self.metrics['family_starts'][family]=self.metrics['family_starts'].get(family,0)+1
        actors=[]
        for mode,action in (('mode_b','action_b'),('mode_w','action_w')):
            choice=schedule.choose_episode(record,self.rng[env][mode],average_ready=snapshot.average_ready,uniform_comparison=uniform_comparison)
            actors.append(EpisodeActor(snapshot,choice,self.rng[env][action]))
        self.active[env]=JointEpisode(self.game,root,*actors,episode_id=self.started);self.start_records[env]=record
        self.births[env]={p:0 for p,c in enumerate(root.board) if c==GRAY};self.started+=1

    def collect(self,count,snapshot,replay,reservoir,label_rng,schedule,*,opening_enabled=True,uniform_comparison=False):
        if type(count) is not int or count<2 or count%2 or count>2*self.envs or replay.insertions!=self.records:raise ValueError('balanced collector chunk/clock required')
        selected=[(self.cursor+k)%self.envs for k in range(count//2)];requests=[]
        for k,env in enumerate(selected):
            if self.active[env] is None:self._start(env,snapshot,schedule,self.records+2*k,uniform_comparison)
            episode=self.active[env]
            for player in (BLACK,WHITE):
                actor=episode.actors[player];observation=self.game.observation(episode.state,player)
                eligible=schedule.label_eligible(actor.choice,self.records+2*k,opening_enabled=opening_enabled)
                requests.append((actor,observation,eligible))
        commitments=commit_many(requests)
        base=self.records
        for k,env in enumerate(selected):
            episode=self.active[env];before=episode.state
            committed={BLACK:commitments[2*k],WHITE:commitments[2*k+1]}
            event=episode.step(replay,reservoir,label_rng,schedule,record=base+2*k,opening_enabled=opening_enabled,precommitted=committed)
            self._observe(env,before,episode,event,self.game.last,schedule)
        self.records+=count;self.cursor=(self.cursor+len(selected))%self.envs
        return count

    def _observe(self,env,before,episode,event,transition,schedule):
        m=self.metrics;after=episode.state;step=episode.steps;m['physical_rounds']+=1
        m['collisions']+=int(transition.collision is not None)
        m['history_match_rounds']+=int(any(h.before==after.board for h in before.history))
        m['restriction_point_rounds'][0]+=len(after.forbidden_black);m['restriction_point_rounds'][1]+=len(after.forbidden_white)
        for field,groups in (('captured_stones',transition.captured),('suicidal_stones',transition.suicidal)):
            for group in groups:m[field][group.color]+=len(group.stones)
        births=self.births[env]
        for point in list(births):
            if after.board[point]!=GRAY:m['removed_gray_lifetimes'][step-births.pop(point)]+=1
        if transition.collision is not None and self.rules.collision=='gray':
            m['created_gray']+=1;point=transition.collision
            if after.board[point]==GRAY:births[point]=step
            else:m['removed_gray_lifetimes'][0]+=1
        for role in range(2):
            m['per_role_records'][role]+=1;m['best_response_visits'][role]+=int(event['modes'][role]);m['acting_counts'][event['acting_policies'][role]]+=1
        if event['terminated']:
            m['episode_lengths'][step]+=1;m['endings'][after.terminal_reason]+=1
            reward=event['reward_black'];m['outcomes'][0 if reward>0 else 1 if reward<0 else 2]+=1
            phase=0 if self.start_records[env]<schedule.reservoir_open else 1 if self.start_records[env]<schedule.handoff_end else 2
            m['completed_episode_modes'][phase][int(event['modes'][0])+2*int(event['modes'][1])]+=1
            for point,birth in births.items():m['terminal_gray_stones']+=1;m['terminal_gray_ages'][step-birth]+=1
            self._territory(after.board)
            if self.on_episode is not None:self.on_episode(episode,self.start_records[env],event)
            self.completed+=1;self.active[env]=None;self.start_records[env]=None;self.start_families[env]=None;self.births[env]={}

    def _territory(self,board):
        adjacent=neighbors(self.rules.size);seen=set()
        for p,color in enumerate(board):
            if color!=EMPTY or p in seen:continue
            stack=[p];seen.add(p);count=0;boundary=0
            while stack:
                count+=1
                for q in adjacent[stack.pop()]:
                    if board[q]:boundary|=1<<(board[q]-1)
                    elif q not in seen:seen.add(q);stack.append(q)
            self.metrics['terminal_empty_points']+=count
            if boundary&4:self.metrics['gray_boundary_empty_points']+=count
            if boundary in (5,6):self.metrics['fractional_bias_empty_points']+=count

    def state_dict(self):
        snapshots={}
        for episode in self.active:
            if episode is not None:
                for actor in episode.actors.values():snapshots[actor.snapshot.version]=actor.snapshot.state_dict()
        return {'version':COLLECTOR_VERSION,'rules':{'size':self.rules.size,'max_rounds':self.rules.max_rounds,'collision':self.rules.collision},
            'envs':self.envs,'seed':self.seed,'starts_sha256':self.starts.sha256,'cursor':self.cursor,'records':self.records,'started':self.started,'completed':self.completed,
            'rng':[{k:v.bit_generator.state for k,v in rng.items()} for rng in self.rng],
            'active':[None if e is None else e.state_dict(lambda s:s.to_dict()) for e in self.active],
            'snapshots':snapshots,'start_records':self.start_records,'start_families':self.start_families,'births':self.births,'metrics':deepcopy(self.metrics)}

    def load_state_dict(self,state,q,average):
        expected=(COLLECTOR_VERSION,{'size':self.rules.size,'max_rounds':self.rules.max_rounds,'collision':self.rules.collision},self.envs,self.seed,self.starts.sha256)
        actual=(state['version'],state['rules'],state['envs'],state['seed'],state['starts_sha256'])
        if actual!=expected:raise ValueError('collector configuration/start distribution differs')
        snapshots={}
        for version,saved in state['snapshots'].items():
            snapshot=FrozenPolicies(q,average,version=int(version),average_ready=saved['average_ready']);snapshot.load_state_dict(saved);snapshots[int(version)]=snapshot
        for name in ('cursor','records','started','completed','start_records','start_families','births','metrics'):setattr(self,name,deepcopy(state[name]))
        if not 0<=self.cursor<self.envs or self.records<0 or self.records%2 or self.completed>self.started:raise ValueError('invalid collector clocks')
        if len(self.start_records)!=self.envs or len(self.start_families)!=self.envs or len(self.births)!=self.envs:raise ValueError('collector curriculum progress dimensions differ')
        if sum(self.metrics['start_counts'].values())!=self.started or sum(self.metrics['starting_gray_counts'])!=self.started:raise ValueError('collector start counts differ')
        if len(state['active'])!=self.envs or len(state['rng'])!=self.envs:raise ValueError('collector environment count differs')
        for env in range(self.envs):
            for key,value in state['rng'][env].items():self.rng[env][key].bit_generator.state=value
            saved=state['active'][env]
            self.active[env]=None if saved is None else JointEpisode.restore(self.game,saved,State.from_dict,snapshots)
            if self.active[env] is not None:
                self.rng[env]['action_b']=self.active[env].actors[BLACK].rng;self.rng[env]['action_w']=self.active[env].actors[WHITE].rng
        if self.started!=self.completed+sum(e is not None for e in self.active) or self.metrics['physical_rounds']*2!=self.records or self.metrics['per_role_records']!=[self.records//2]*2:raise ValueError('collector episode/round accounting differs')

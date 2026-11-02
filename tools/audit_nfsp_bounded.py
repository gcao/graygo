"""Read-only bounded-Go/meaningful-branch audit, independent of trainer grades.

Run with the trial's frozen package on PYTHONPATH. Matrix security is verified
with Fraction arithmetic; native values condition on the hidden episode policy.
Historical actions/labels discarded by bounded storage cannot be reconstructed.
"""
import argparse
from collections import deque
from fractions import Fraction
from functools import lru_cache
from importlib import metadata
import json
from pathlib import Path
import runpy
import time
import numpy as np
import torch
from graygo.buffers import ReplayBuffer,ReservoirBuffer
from graygo.confidence import canonical_sha
from graygo.manifest import source_digest,write_report
from graygo.solving import solve_matrix,SolveBudgetExceeded
from graygo.storage import CheckpointStore,digest
from graygo.task_adapter import TaskAdapter
from graygo.task_training import NFSPTaskConfig,HEADER_DTYPE
from graygo.tasks import BalancedCoverage

HELPER=Path(__file__).with_name('audit_nfsp_delayed.py')
shared=runpy.run_path(str(HELPER))
require,close,predict,compare_tree,q_updates=(shared[k] for k in ('require','close','predict','compare_tree','q_updates'))


def policy_tables(adapter,weights,config,epsilon):
    # Independent CPU verification must not consume the caller's training RNG
    # or inherit an ambient CUDA default device.
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):
        return predict(adapter,weights,config,epsilon)


def reason(state,board):
    if not board:return 1
    return {'mutual_pass':2,'round_limit':3}[state.terminal_reason]


class ExactGame:
    """Independent complete finite-game recursion with rational LP witnesses."""
    def __init__(self,adapter):
        self.adapter=adapter;self.game=adapter.game;self.nodes=0;self.layers=[];self.ends=[];self.certificates={}
        self.value=lru_cache(maxsize=None)(self._value)
        for root in adapter.roots:
            layers={};ends=set();seen=set()
            def walk(state,depth):
                if (state,depth) in seen:return
                seen.add((state,depth))
                if len(seen)>adapter.config.max_nodes:raise SolveBudgetExceeded('independent full-game inventory exceeds its declared bound')
                if self.game.terminal(state):
                    ends.add((depth,float(self.game.utility(state)),reason(state,adapter.board)));return
                layers.setdefault(depth,set()).add(state)
                for black in self.game.actions(state,1):
                    for white in self.game.actions(state,2):walk(self.game.transition(state,black,white),depth+1)
            walk(root,0);self.layers.append(layers);self.ends.append(ends)
        self.root_values=[self.value(root) for root in adapter.roots]
        close(adapter.values,[float(v) for v in self.root_values],'direct rational game values differ from adapter oracle')

    def _value(self,state):
        self.nodes+=1
        if self.nodes>self.adapter.config.max_nodes:raise SolveBudgetExceeded('independent rational game solve exceeds its declared bound')
        if self.game.terminal(state):return Fraction(str(float(self.game.utility(state))))
        require(state in self.adapter.index,'complete direct recursion found a state missing from the task inventory')
        matrix=tuple(tuple(self.value(self.game.transition(state,b,w)) for w in self.game.actions(state,2)) for b in self.game.actions(state,1))
        solution=solve_matrix([[float(x) for x in row] for row in matrix])
        p=tuple(Fraction(float(x)).limit_denominator(1000000) for x in solution.row)
        q=tuple(Fraction(float(x)).limit_denominator(1000000) for x in solution.column)
        require(min((*p,*q))>=0 and sum(p)==sum(q)==1,'rational strategy reconstruction failed')
        lower=min(sum(a*x for a,x in zip(p,column)) for column in zip(*matrix))
        upper=max(sum(a*x for a,x in zip(row,q)) for row in matrix)
        require(lower==upper,'direct rational primal/dual security bounds do not agree')
        self.certificates[state]={'state_index':self.adapter.index[state],'state':self.adapter.encode_state(state),
            'black_actions':list(self.game.actions(state,1)),'white_actions':list(self.game.actions(state,2)),
            'continuation_matrix':[[str(x) for x in row] for row in matrix],
            'row_strategy':[str(x) for x in p],'column_strategy':[str(x) for x in q],
            'value':str(lower),'security_lower':str(lower),'security_upper':str(upper)}
        return lower

    def certificate(self):
        return {'version':'graygo/rational-bounded-game-dp/1','root_state_indices':[self.adapter.index[root] for root in self.adapter.roots],
            'states':[self.certificates[state] for state in sorted(self.certificates,key=self.adapter.index.get)],
            'matrix_semantics':'complete simultaneous legal-action matrices with exact solved continuation values; every nonterminal public state has equal rational primal/dual security bounds'}


def independent_roots(adapter,exact,q,average,target,greedy,records,ready):
    """Component-wise dynamic values plus public-prefix Bayes, not trainer DP."""
    game=adapter.game;c=adapter.config;ramp=min(1.,max(0.,(records-c.reservoir_open)/(c.handoff_end-c.reservoir_open)))
    avg_mass=.9*ramp if ready else 0.
    weights=np.array([.1 if c.uniform_comparison else 1-avg_mass,avg_mass,.9-avg_mass if c.uniform_comparison else 0.])
    uniform=np.zeros_like(target)
    for state,index in adapter.index.items():
        for role in (1,2):
            actions=game.actions(state,role);uniform[index,role-1,list(actions)]=1/len(actions)
    components=(target,average,uniform)

    def best_value(root,policy,role):
        @lru_cache(maxsize=None)
        def visit(state):
            if game.terminal(state):return float(game.utility(state))
            rows=game.actions(state,1);cols=game.actions(state,2);i=adapter.index[state]
            matrix=np.array([[visit(game.transition(state,b,w)) for w in cols] for b in rows])
            return float((policy[i,0,list(rows)]@matrix).min()) if role==1 else float((matrix@policy[i,1,list(cols)]).max())
        return visit(root)

    results=[]
    for root_index,root in enumerate(adapter.roots):
        early=[];actions=[]
        for role in (1,2):
            sign=1 if role==1 else -1;other=3-role;seen={};calls=0
            @lru_cache(maxsize=None)
            def component_value(component,state):
                if game.terminal(state):return sign*float(game.utility(state))
                own=game.actions(state,role);opposing=game.actions(state,other);i=adapter.index[state]
                result=0.
                for a in own:
                    for b in opposing:
                        black,white=(a,b) if role==1 else (b,a)
                        result+=target[i,role-1,a]*components[component][i,other-1,b]*component_value(component,game.transition(state,black,white))
                return float(result)
            def conditioned(state,belief):
                nonlocal calls
                calls+=1
                if calls>c.max_nodes:raise SolveBudgetExceeded('independent native-prefix inventory exceeds its declared bound')
                if game.terminal(state):return
                own=tuple(game.actions(state,role));opposing=tuple(game.actions(state,other));i=adapter.index[state]
                values=[]
                for a in own:
                    value=0.
                    for component,mass in enumerate(belief):
                        for b in opposing:
                            black,white=(a,b) if role==1 else (b,a)
                            value+=mass*components[component][i,other-1,b]*component_value(component,game.transition(state,black,white))
                    values.append(float(value))
                # Traversal establishes each public prefix; values above were
                # computed with separately cached fixed-component continuations.
                for a in own:
                    for b in opposing:
                        joint=belief*np.array([table[i,other-1,b] for table in components]);total=joint.sum()
                        if total:
                            black,white=(a,b) if role==1 else (b,a)
                            conditioned(game.transition(state,black,white),joint/total)
                if state in seen:close(seen[state],values,'public-state alias has incompatible prefix-conditioned Q values')
                seen[state]=values
            conditioned(root,weights)
            for state,values in seen.items():
                index=adapter.index[state];legal=list(game.actions(state,role));truth=float(target[index,role-1,legal]@np.array(values))
                prediction=float(target[index,role-1,legal]@q[index,role-1,legal])
                early.append({'state_index':index,'player':role,'expected':truth,'predicted':prediction,'error':abs(truth-prediction)})
            index=adapter.index[root]
            for action,truth in zip(game.actions(root,role),seen[root]):
                prediction=float(q[index,role-1,action]);actions.append({'player':role,'action':action,'expected':truth,'predicted':prediction,'error':abs(truth-prediction)})
        value=float(exact.root_values[root_index]);index=adapter.index[root]
        def gaps(policy):return [value-best_value(root,policy,1),best_value(root,policy,2)-value]
        results.append({'root_index':root_index,'root_tag':adapter.root_tags[root_index],'value':value,
            'native_opponent_weights':weights.tolist(),'average_gaps':gaps(average),'greedy_q_gaps':gaps(greedy),
            'root_action_values':actions,'early_values':early,'late_values':[],
            'max_early_error':max(row['error'] for row in early+actions),'max_late_error':None,
            'root_average_policies':average[index].tolist(),'root_greedy_policies':greedy[index].tolist()})
    return results


def partial_header(active):
    episode=active['episode'];choices=[episode['actors'][str(p)]['choice'] for p in (1,2)]
    row=np.zeros(1,dtype=HEADER_DTYPE);row['episode']=episode['episode_id'];row['start_record']=active['start_record']
    row['steps']=episode['steps'];row['root']=active['root_index'];row['generation']=active['snapshot']['version']
    row['supplemental']=episode['supplemental'];row['average_ready']=active['snapshot']['average_ready'];forced=episode['forced_root']
    row['forced_role']=0 if forced is None else forced[0];row['forced_action']=65535 if forced is None else forced[1]
    for key,field in (('best_response_mode','best_response'),('mode_draw','mode_draw'),('ramp_draw','ramp_draw'),('ramp_probability','ramp_probability'),('epsilon','epsilon')):
        row[field]=[[c[key] for c in choices]]
    row['acting']=[[{'q':0,'average':1,'uniform':2}[c['acting_policy']] for c in choices]]
    return row


def accounting(adapter,exact,config,state,headers,completed,replay,reservoir,n):
    counts=state['backend']['counters']
    excluded=np.array([int(row['start_record'])+2-int(row['forced_role']) for row in headers if row['supplemental']],dtype=np.int64)
    excluded.sort()
    def q_at(record):return record-int(np.searchsorted(excluded,record,side='left'))
    q_total=q_at(n)
    require(counts['fresh_records']==n and counts['fresh_q']==q_total,'eligible Q/aggregate accounting differs')
    require(replay.insertions==n and replay.size==min(n,config.capacity) and replay.q_insertions==q_total,'replay insertion/eligibility history differs')
    require(replay.q_size==int(replay.arrays['value_mask'][:replay.size].sum()),'live Q eligibility differs')
    require(reservoir.opening_record==config.reservoir_open,'reservoir opening differs')
    expected_records=np.full(config.capacity,-1,dtype=np.int64);expected_versions=np.zeros(config.capacity,dtype=np.uint64)
    rng={name:np.random.default_rng(config.seed+offset) for name,offset in
         (('mode_black',101),('mode_white',102),('reservoir',105),('q_replay',106),('supervised',107))}
    coverage=None if adapter.board else BalancedCoverage((2,),seed=config.seed+108)
    cursor=ordinary=seen=changes=sl_targets=0;events=[];recent=deque(maxlen=config.batch);last_accepted=-1
    mode_visits=np.zeros(2,dtype=np.int64);acting=np.zeros(3,dtype=np.int64);forced_counts=np.zeros(2,dtype=np.int64)
    root_episodes=np.zeros((len(adapter.roots),2),dtype=np.uint64);outcomes=[0,0,0];endings={'horizon':0,'mutual_pass':0,'round_limit':0}
    for episode,row in enumerate(headers):
        start=int(row['start_record']);steps=int(row['steps']);root_index=int(row['root'])
        require(0<=root_index<len(adapter.roots),'episode root index is invalid')
        root=adapter.roots[root_index];supplemental=bool(row['supplemental'])
        require(int(row['episode'])==episode and start==cursor and steps>0,'episode/record continuity differs')
        if episode<len(completed):
            require((steps,float(row['reward_black']),int(row['termination'])) in exact.ends[root_index],'completed episode has an impossible length/outcome/termination')
            outcomes[0 if row['reward_black']>0 else 1 if row['reward_black']<0 else 2]+=1
            endings[{1:'horizon',2:'mutual_pass',3:'round_limit'}[int(row['termination'])]]+=1
        else:require(steps in exact.layers[root_index] and row['termination']==0,'active episode must remain nonterminal')
        if coverage is None:
            require(root_index==0 and not supplemental and row['forced_role']==0 and row['forced_action']==65535,'board root/supplemental origin differs')
        else:
            entry=coverage.next()
            require(root.controller==entry.controller and supplemental==entry.supplemental,'balanced randomized root/coverage differs')
            require(int(row['forced_role'])==(entry.controller if supplemental else 0)
                    and int(row['forced_action'])==(entry.forced_root if supplemental else 65535),'forced root origin differs')
        if supplemental:forced_counts[int(row['forced_role'])-1]+=1
        ready=bool(events);require(bool(row['average_ready'])==ready,'average publication availability differs')
        require(int(row['generation'])==q_updates(q_at(start),config)+len(events),'frozen snapshot generation differs')
        ramp=min(1.,max(0.,(start-config.reservoir_open)/(config.handoff_end-config.reservoir_open)))
        epsilon=.01+.09*max(0.,1-start/config.epsilon_decay)
        for role,name in enumerate(('mode_black','mode_white')):
            draws=rng[name].random(2);close([row['mode_draw'][role],row['ramp_draw'][role]],draws,'private episode mode/ramp RNG differs',atol=0)
            best=draws[0]<.1;policy=1 if not best and ready and draws[1]<ramp else (2 if config.uniform_comparison and not best else 0)
            require(bool(row['best_response'][role])==best and int(row['acting'][role])==policy,'logical/acting policy differs')
            close(row['epsilon'][role],epsilon,'episode epsilon clock differs',atol=0);close(row['ramp_probability'][role],ramp,'episode ramp differs',atol=0)
            mode_visits[role]+=steps*best;acting[policy]+=steps
        root_episodes[root_index,int(supplemental)]+=1;labels=np.empty(0,dtype=np.int64)
        if not supplemental:
            ordinals=start+np.arange(2*steps);labels=ordinals[(ordinals>=config.reservoir_open)&row['best_response'][ordinals%2].astype(bool)]
            for boundary_ordinary in range((ordinary//config.batch+1)*config.batch,ordinary+2*steps+1,config.batch):
                boundary=start+boundary_ordinary-ordinary;boundary_seen=seen+int(np.searchsorted(labels,boundary))
                if boundary_seen>=config.batch:
                    events.append((boundary,boundary_seen));sl_targets+=min(boundary_seen,config.capacity) if config.learner=='tabular' else config.batch
                    if config.learner=='neural':rng['supervised'].integers(0,min(boundary_seen,config.capacity),config.batch)
            ordinary+=2*steps
        for record in labels:
            slot=seen if seen<config.capacity else int(rng['reservoir'].integers(0,seen+1));seen+=1
            if slot<config.capacity:
                recent.append((int(record),slot,int(expected_records[slot])));expected_records[slot]=record
                expected_versions[slot]=int(row['generation'])*2;changes+=1;last_accepted=int(record)
        cursor=start+2*steps
    require(cursor==n and ordinary==counts['fresh_ordinary'],'ordinary/aggregate record totals differ')
    require(state['completed_rounds']==sum(int(row['steps']) for row in completed) and state['per_role']==[n//2]*2,'completed/per-role round counts differ')
    require(state['coverage']==(None if coverage is None else coverage.state_dict()),'coverage continuation differs')
    require(outcomes==state['outcomes'] and endings==state['termination_counts'],'completed game outcome/termination metrics differ')
    require(mode_visits.tolist()==state['best_response_visits'] and acting.tolist()==[state['acting_counts'][k] for k in ('q','average','uniform')],'record-weighted mode/acting metrics differ')
    require(state['per_role_q']==[n//2-int(forced_counts[1]),n//2-int(forced_counts[0])],'per-role Q exclusions differ')
    require(counts['q_updates']==q_updates(q_total,config) and counts['q_loss_targets']==counts['q_updates']*config.batch
            and counts['target_refreshes']==q_total//config.target_every,'Q/reuse/refresh accounting differs')
    require(counts['supervised_updates']==len(events) and counts['supervised_targets']==sl_targets,'supervised publication/target counts differ')
    excluded_before=0
    for boundary in range(config.learning_start,q_total+1,config.batch):
        aggregate=boundary+excluded_before
        while excluded_before<len(excluded) and excluded[excluded_before]<aggregate:excluded_before+=1;aggregate+=1
        low=max(0,aggregate-config.capacity);invalid=int(np.searchsorted(excluded,aggregate)-np.searchsorted(excluded,low))
        for _ in range(config.q_reuse):rng['q_replay'].integers(0,min(aggregate,config.capacity)-invalid,config.batch)
    for name,generator in rng.items():require(generator.bit_generator.state==state['rng'][name],name+' continuation RNG differs')
    for role,name in enumerate(('action_black','action_white')):
        generator=np.random.default_rng(config.seed+103+role);generator.bit_generator.advance(n//2-int(forced_counts[role]))
        require(generator.bit_generator.state==state['rng'][name],name+' private draw count differs')
    for name,updates in (('q_optimizer',counts['q_updates']),('average_optimizer',counts['supervised_updates'])):
        optimizer=state['backend'][name]
        if optimizer is not None:require(all(int(v['step'].item())==updates for v in optimizer['state'].values()),'Adam update count differs')
    require(reservoir.seen==seen and reservoir.revisions==changes,'eligible labels/Algorithm R replacements differ')
    np.testing.assert_array_equal(reservoir.arrays['record'][:reservoir.size],expected_records[:reservoir.size])
    np.testing.assert_array_equal(reservoir.arrays['policy_version'][:reservoir.size],expected_versions[:reservoir.size])
    return {'q_at':q_at,'events':events,'recent':recent,'expected_records':expected_records,'root_episodes':root_episodes,
            'ordinary':ordinary,'q_total':q_total,'labels':seen,'changes':changes,'last_accepted':last_accepted}


def observation_key(x,g,legal):return (np.asarray(x,dtype=np.uint8).tobytes(),np.asarray(g,dtype=np.float32).tobytes(),np.asarray(legal,dtype=bool).tobytes())


def retained_records(adapter,exact,config,state,headers,replay,reservoir,account):
    game=adapter.game;starts=headers['start_record'].astype(np.int64);lookup={}
    for public,index in adapter.index.items():
        for role in (1,2):
            obs=game.observation(public,role);key=observation_key(obs.spatial,obs.global_features,obs.legal)
            key=(role,key);require(key not in lookup,'public observation aliases distinct task states');lookup[key]=(public,role,index)
    def unpack(arrays,prefix,indices):return adapter.spec.unpack(arrays[prefix+'spatial'][indices],arrays[prefix+'globals'][indices],arrays[prefix+'legal'][indices])
    def identity(arrays,indices,records):
        x,g,legal=unpack(arrays,'',indices);items=[]
        for a,b,c,record in zip(x,g,legal,records):
            key=(int(record)%2+1,observation_key(a,b,c));require(key in lookup,'retained observation is not a complete legal task state')
            public,role,index=lookup[key];header=headers[np.searchsorted(starts,record,side='right')-1];depth=(int(record)-int(header['start_record']))//2
            require(role==int(record)%2+1 and public in exact.layers[int(header['root'])].get(depth,set()),'retained state/root/role/depth differs')
            items.append((public,role,index,header,depth))
        return items,legal
    a=replay.arrays;ordinals=a['ordinal'][:replay.size].astype(np.int64);n=config.records
    np.testing.assert_array_equal(np.sort(ordinals),np.arange(n-replay.size,n))
    slots={int(record):i for i,record in enumerate(ordinals)};decoded={};completed_logs={};paired=orphan=0
    for start in range(0,replay.size,2048):
        selection=slice(start,min(start+2048,replay.size));records=ordinals[selection];items,legal=identity(a,selection,records)
        next_x,next_g,next_legal=unpack(a,'next_',selection)
        for local,(record,item) in enumerate(zip(records,items)):
            slot=start+local;public,role,index,header,depth=item;decoded[int(record)]=(public,role,index)
            require(int(a['stream'][slot])==2*int(header['episode'])+role-1 and int(a['step'][slot])==depth,'retained stream/step differs')
            require(int(a['mode'][slot])==int(header['best_response'][role-1]),'retained mode differs')
            versions=2*int(header['generation'])+(header['acting']==1).astype(np.int64)
            require(int(a['policy_version'][slot])==versions[role-1] and int(a['opponent_version'][slot])==versions[2-role],'retained frozen policy/opponent version differs')
            supplemental=bool(header['supplemental']);eligible=not(supplemental and depth==0 and int(header['forced_role'])!=role)
            require(bool(a['supplemental'][slot])==supplemental and bool(a['actor_mask'][slot])==(not supplemental),'retained origin/actor mask differs')
            require(bool(a['value_mask'][slot])==eligible,'retained passive-root Q eligibility differs')
            action=int(a['action'][slot]);require(0<=action<adapter.actions and legal[local,action],'retained action is illegal')
            probability=float(a['behavior_prob'][slot]);require(np.isfinite(probability) and 0<probability<=1,'retained behavior probability invalid')
            if supplemental and depth==0 and int(header['forced_role'])==role:
                require(action==int(header['forced_action']) and probability==1,'forced root action/probability differs')
            other_slot=slots.get(int(record)^1)
            opposing=game.actions(public,3-role) if other_slot is None else (int(a['action'][other_slot]),)
            matches=[]
            for other in opposing:
                black,white=(action,other) if role==1 else (other,action)
                try:following=game.transition(public,black,white)
                except ValueError:continue
                obs=game.observation(following,role)
                if (np.array_equal(obs.spatial,next_x[local]) and np.array_equal(obs.global_features,next_g[local]) and np.array_equal(obs.legal,next_legal[local])):matches.append((following,black,white))
            require(matches,'retained next observation does not follow its actual legal joint transition')
            if other_slot is None:orphan+=1
            else:
                other_obs=unpack(a,'',slice(other_slot,other_slot+1));other_key=(3-role,observation_key(other_obs[0][0],other_obs[1][0],other_obs[2][0]))
                require(other_key in lookup and lookup[other_key][:2]==(public,3-role),'paired commitments did not use the same public state')
                paired+=1
            following,black,white=matches[0];done=game.terminal(following);reward=float(game.utility(following)) if done else 0.
            require(bool(a['terminated'][slot])==done,'retained terminal flag differs')
            close(float(a['reward'][slot]),reward*(1 if role==1 else -1),'retained terminal reward differs',atol=0)
            if done:require(depth+1==int(header['steps']) and reward==float(header['reward_black']) and reason(following,adapter.board)==int(header['termination']),'terminal transition disagrees with original episode header')
            if role==1 and other_slot is not None:
                episode=int(header['episode']);row={'step':depth,'record':int(record),'actions':[black,white],
                    'probabilities':[float(a['behavior_prob'][slot]),float(a['behavior_prob'][other_slot])],
                    'modes':[bool(x) for x in header['best_response']],
                    'acting_policies':[{0:'q',1:'average',2:'uniform'}[int(x)] for x in header['acting']],
                    'versions':[int(x) for x in versions],'epsilons':[float(x) for x in header['epsilon']],
                    'supplemental':supplemental,'q_eligible':[not(supplemental and depth==0 and int(header['forced_role'])!=p) for p in (1,2)],
                    'terminated':done,'reward_black':reward}
                completed_logs.setdefault(episode,{})[depth]=row
    trajectories=0
    for episode,logs in completed_logs.items():
        header=headers[episode]
        if set(logs)!=set(range(int(header['steps']))):continue
        log=[logs[d] for d in range(int(header['steps']))]
        if episode<state['episodes']:
            # Header hashes cover original float64 probabilities, while replay
            # intentionally retains float32. Verify the actual path/metadata;
            # do not claim recovery of discarded probability bits or their hash.
            require(log[-1]['terminated'],'retained complete action path is not terminal');trajectories+=1
        else:
            saved=state['active']['episode']['log'];require(len(saved)==len(log),'active trajectory length differs')
            for original,reconstructed in zip(saved,log):
                close(np.asarray(original['probabilities'],dtype=np.float32),reconstructed['probabilities'],'active replay probabilities differ',atol=0)
                compare_tree({k:v for k,v in original.items() if k!='probabilities'},
                             {k:v for k,v in reconstructed.items() if k!='probabilities'},'active trajectory')
            last_record=int(header['start_record'])+2*(int(header['steps'])-1);slot=slots[last_record]
            public=decoded[last_record][0];following=game.transition(public,*log[-1]['actions'])
            require(adapter.encode_state(following)==state['active']['episode']['state'],'active public-state continuation differs')
    label_identities=[];label_greedy=[]
    for start in range(0,reservoir.size,2048):
        selection=slice(start,min(start+2048,reservoir.size));records=reservoir.arrays['record'][selection].astype(np.int64)
        items,legal=identity(reservoir.arrays,selection,records)
        greedy=np.unpackbits(reservoir.arrays['greedy'][selection],axis=-1,count=adapter.actions,bitorder='little').astype(bool)
        require(greedy.any(-1).all() and not(greedy&~legal).any(),'frozen reservoir target is empty or illegal')
        for item in items:require(not item[3]['supplemental'] and item[3]['best_response'][item[1]-1],'reservoir label does not belong to an ordinary best-response visit')
        label_identities.extend((item[0],item[1]) for item in items);label_greedy.extend(greedy)
    mean=tabular_mean_bounds(adapter,exact,config,state,headers,reservoir,account,label_identities,label_greedy,decoded)
    return {'retained_executed_records_checked':replay.size,'retained_joint_transition_records_checked':paired,
            'retained_unpaired_transition_records_checked':orphan,'retained_complete_action_paths_reconstructed':trajectories,
            'completed_original_log_hashes_reconstructed':False,
            'retained_label_records_checked':reservoir.size,**mean}


def tabular_mean_bounds(adapter,exact,config,state,headers,reservoir,account,identities,greedies,decoded):
    result={'published_tabular_mean_verified':False,'published_tabular_mean_interval_checked':False,
            'published_tabular_exact_mean_states':0,'published_tabular_uncertain_mean_states':0,'unresolved_published_labels':0,
            'published_tabular_uncertain_tokens':[],'published_tabular_exact_root_mean_states':0}
    if config.learner!='tabular' or not account['events']:return result
    publication,seen=account['events'][-1];current=account['expected_records'];published=current.copy()
    for record,slot,previous in reversed([event for event in account['recent'] if event[0]>=publication]):published[slot]=previous
    require(int((published>=0).sum())==min(seen,config.capacity),'published reservoir occupancy differs')
    features=adapter.spec.spatial_shape[0];known=np.zeros(features,dtype=np.int64);totals=np.zeros((features,adapter.actions))
    def token(public,role):return int(adapter.game.observation(public,role).spatial.argmax())
    for slot,(identity,greedy) in enumerate(zip(identities,greedies)):
        if published[slot]>=0 and published[slot]==current[slot]:
            index=token(*identity);known[index]+=1;totals[index]+=greedy/greedy.sum()
    upper=np.zeros(features,dtype=np.int64);unresolved=0;starts=headers['start_record'].astype(np.int64)
    for slot in np.flatnonzero((published>=0)&(published!=current)):
        record=int(published[slot]);unresolved+=1
        if record in decoded:candidates={token(decoded[record][0],decoded[record][1])}
        else:
            header=headers[np.searchsorted(starts,record,side='right')-1];depth=(record-int(header['start_record']))//2;role=record%2+1
            # The replaced target's old observation is absent from bounded
            # storage. Bound every legal public state at its original depth.
            candidates={token(public,role) for public in exact.layers[int(header['root'])][depth]}
        require(candidates,'unknown published label has no legal original state')
        upper[list(candidates)]+=1
    table=state['backend']['average']['probabilities'].numpy()
    require(table.shape==totals.shape and np.isfinite(table).all() and (table>=0).all() and (table<=1).all(),'published tabular probabilities invalid')
    close(table.sum(-1),np.ones(features),'published tabular rows are not normalized',atol=1e-10)
    certain=upper==0;expected=np.full_like(totals,1/adapter.actions);present=known>0;expected[present]=totals[present]/known[present,None]
    close(table[certain],expected[certain],'published exact tabular reservoir means differ',atol=1e-10)
    uncertain=(~certain)&present;denominator=known+upper
    lower=np.divide(totals,denominator[:,None],out=np.zeros_like(totals),where=denominator[:,None]>0)
    high=np.divide(totals+upper[:,None],denominator[:,None],out=np.ones_like(totals),where=denominator[:,None]>0)
    require((table[uncertain]>=lower[uncertain]-1e-10).all() and (table[uncertain]<=high[uncertain]+1e-10).all(),
            'published means exceed bounded unresolved-label uncertainty')
    return {**result,'published_tabular_mean_verified':unresolved==0,'published_tabular_mean_interval_checked':True,
            'published_tabular_exact_mean_states':int(certain.sum()),'published_tabular_uncertain_mean_states':int((~certain).sum()),
            'unresolved_published_labels':unresolved,'published_tabular_uncertain_tokens':np.flatnonzero(~certain).tolist(),
            'published_tabular_exact_root_mean_states':sum(bool(certain[token(root,role)]) for root in adapter.roots for role in (1,2)),
            'tabular_mean_scope':'exact unaffected rows; necessary simplex intervals for overwritten unpublished-tail labels, not a proof of joint historical assignment'}


def active_episode(adapter,config,state,headers,reservoir):
    active=state['active']
    if active is None:return {'active_frozen_commitments_reconstructed':0,'active_private_rng_draws_reconstructed':0,'active_frozen_greedy_labels_verified':0}
    episode=active['episode'];snapshot=active['snapshot'];header=headers[-1];start=int(header['start_record'])
    require(snapshot['version']==int(header['generation']) and snapshot['average_ready']==bool(header['average_ready']),'active snapshot identity differs')
    epsilon=float(header['epsilon'][0]);q,average,target,greedy=policy_tables(adapter,snapshot,config,epsilon)
    generators={};components={'q':target,'average':average};prior_forced={p:int((headers[:-1]['forced_role']==p).sum()) for p in (1,2)}
    for role,name in ((1,'action_black'),(2,'action_white')):
        actor=episode['actors'][str(role)]
        require(actor['rng']==state['rng'][name],'active actor RNG differs from global continuation')
        choice=actor['choice']
        require(actor['snapshot_version']==snapshot['version'] and choice['start_record']==start,'active actor snapshot/start differs')
        generator=np.random.default_rng(config.seed+102+role);generator.bit_generator.advance(start//2-prior_forced[role]);generators[role]=generator
    public=adapter.roots[active['root_index']];commitments=draws=labels=0
    for depth,event in enumerate(episode['log']):
        index=adapter.index[public]
        for role in (1,2):
            choice=episode['actors'][str(role)]['choice'];forced=episode['forced_root']
            if depth==0 and forced is not None and forced[0]==role:action=forced[1];probability=1.
            else:
                acting=choice['acting_policy']
                if acting=='uniform':
                    p=np.zeros(adapter.actions);legal=adapter.game.actions(public,role);p[list(legal)]=1/len(legal)
                else:p=components[acting][index,role-1].copy()
                p/=p.sum();cdf=p.cumsum();cdf[-1]=1.;action=int((generators[role].random()>=cdf).sum());probability=float(p[action]);draws+=1
            require(action==event['actions'][role-1],'active frozen private action differs')
            close(event['probabilities'][role-1],probability,'active frozen action probability differs',atol=1e-12);commitments+=1
            record=start+2*depth+role-1
            for slot in np.flatnonzero(reservoir.arrays['record'][:reservoir.size]==record):
                bits=np.unpackbits(reservoir.arrays['greedy'][slot],count=adapter.actions,bitorder='little').astype(bool)
                require(np.array_equal(bits,greedy[index,role-1]>0),'active frozen greedy reservoir target differs');labels+=1
        public=adapter.game.transition(public,*event['actions'])
    for role,generator in generators.items():require(generator.bit_generator.state==episode['actors'][str(role)]['rng'],'active frozen private draw continuation differs')
    require(adapter.encode_state(public)==episode['state'],'active frozen path continuation differs')
    return {'active_frozen_commitments_reconstructed':commitments,'active_private_rng_draws_reconstructed':draws,'active_frozen_greedy_labels_verified':labels}


def _audit(root,require_formal,budget_profile):
    manifest=json.loads((root/'run.json').read_text());config=NFSPTaskConfig(**manifest['config'])
    require(config.task in ('branching','gray-pure','gray-mixed'),'bounded auditor requires branching or a bounded Gray Go task; use the separate chain auditor for delayed tasks')
    require(manifest['source_sha256']==source_digest(),'use the trial pinned package on PYTHONPATH')
    require(set(manifest['dependencies'])=={'numpy','torch','scipy'},'original dependency inventory differs')
    for package,version in manifest['dependencies'].items():
        require(version is not None and metadata.version(package)==version,'use the trial pinned dependency versions: '+package)
    require(manifest.get('eligibility_version')==2,'bounded auditor requires explicit corrected eligibility')
    if require_formal:require(config.formal_protocol(),'trial does not use the formal primary protocol')
    if config.budget_profile_sha256:
        path=Path(budget_profile) if budget_profile else root.parent/(config.learner+'-budget.json')
        require(digest(path)==config.budget_profile_sha256,'funding profile checksum differs');funded=json.loads(path.read_text())
        require(funded['source_sha256']==source_digest() and funded['learner']==config.learner and funded['task']==config.task
                and funded['q_reuse']==config.q_reuse and funded['complete_pipeline'] and set(funded['phases'])=={'pre','post'},'funding profile differs from complete selected learner')
    torch.set_num_threads(1);adapter=TaskAdapter(config);exact=ExactGame(adapter)
    require(manifest['adapter']==json.loads(json.dumps(adapter.manifest())),'task roots/codebook/oracle/observation inventory differs')
    n=config.records;bundle=root/'resume'/f'{n:012d}';seal=CheckpointStore.verify(bundle)
    require(seal['kind']=='resume' and seal['step']==n and seal['metadata']['source_sha256']==source_digest(),'final resume/source binding differs')
    state=torch.load(bundle/'trainer.pt',weights_only=True,map_location='cpu');counts=state['backend']['counters']
    require(state['format_version']==2 and state['backend']['clock_version']==2 and state['config']==config.plan(),'state/config/counter semantics differ')
    path=root/'episodes.bin';audit=state['episode_audit']
    require(path.stat().st_size==audit['bytes']==state['episodes']*HEADER_DTYPE.itemsize and digest(path)==audit['sha256'],'episode header bytes/checksum differ')
    schema=json.loads((root/'episodes.schema.json').read_text())
    require(schema['itemsize']==HEADER_DTYPE.itemsize and schema['dtype']==json.loads(json.dumps(HEADER_DTYPE.descr)),'header schema differs')
    completed=np.memmap(path,dtype=HEADER_DTYPE,mode='r',shape=(state['episodes'],)) if state['episodes'] else np.empty(0,dtype=HEADER_DTYPE)
    headers=np.concatenate((completed,partial_header(state['active']))) if state['active'] else completed
    require(len(headers)>0,'completed positive task run has no episode evidence')
    replay=ReplayBuffer.load(bundle/'replay.npz');reservoir=ReservoirBuffer.load(bundle/'reservoir.npz')
    require(replay.spec==reservoir.spec==adapter.spec and replay.capacity==reservoir.capacity==config.capacity,'buffer schema/capacity differs')
    account=accounting(adapter,exact,config,state,headers,completed,replay,reservoir,n)
    with np.load(bundle/'metrics.npz',allow_pickle=False) as metrics:
        shape=(len(adapter.states),2,adapter.actions,2)
        require(all(metrics[key].shape==shape for key in ('visits','q_eligible_visits','replay_starts')),'state/role/action/origin metrics shape differs')
        require(int(metrics['visits'].sum())==n and int(metrics['visits'][...,0].sum())==account['ordinary'],'visit/origin metrics differ')
        require(int(metrics['q_eligible_visits'].sum())==account['q_total'] and metrics['q_eligible_visits'].sum(axis=(0,2,3)).tolist()==state['per_role_q'],'Q eligibility/role visit metrics differ')
        require((metrics['q_eligible_visits']<=metrics['visits']).all(),'Q-eligible visits exceed visits')
        require(int(metrics['replay_starts'].sum())==int(metrics['overlap_counts'].sum())==counts['q_loss_targets'],'replay/overlap target totals differ')
        np.testing.assert_array_equal(metrics['root_episodes'],account['root_episodes'])
    retained=retained_records(adapter,exact,config,state,headers,replay,reservoir,account)
    retained.update(active_episode(adapter,config,state,headers,reservoir))
    expected_steps=list(range(config.eval_every,n+1,config.eval_every))
    if not expected_steps or expected_steps[-1]!=n:expected_steps.append(n)
    require([e['records'] for e in state['evaluations']]==expected_steps,'scheduled evaluation inventory differs')
    require(sorted(int(p.name) for p in (root/'weights').glob('[0-9]*'))==expected_steps,'permanent public archive inventory differs')
    events=np.array([event[0] for event in account['events']],dtype=np.int64);evaluations=[];archive_hashes={}
    for evaluation in state['evaluations']:
        step=evaluation['records'];archive=root/'weights'/f'{step:012d}';sealed=CheckpointStore.verify(archive)
        require(sealed['kind']=='weights' and sealed['step']==step and sealed['metadata']['source_sha256']==source_digest(),'archive step/source differs')
        weights=torch.load(archive/'models.pt',weights_only=True,map_location='cpu');require(weights['config']==config.plan(),'archive configuration differs')
        archive_hashes[str(step)]=digest(archive/'manifest.json');ready=bool(np.searchsorted(events,step,side='right'))
        require(evaluation['average_ready']==ready,'scheduled average availability differs')
        clock=evaluation['counters'];eligible=account['q_at'](step)
        require(clock['fresh_records']==step and clock['fresh_q']==eligible and clock['q_updates']==q_updates(eligible,config)
                and clock['q_loss_targets']==clock['q_updates']*config.batch and clock['target_refreshes']==eligible//config.target_every,
                'scheduled aggregate/Q/update/refresh counters differ')
        epsilon=.01+.09*max(0.,1-step/config.epsilon_decay);close(evaluation['epsilon'],epsilon,'scheduled epsilon differs',atol=0)
        q,average,target,greedy=policy_tables(adapter,weights,config,epsilon)
        oracle=independent_roots(adapter,exact,q,average,target,greedy,step,ready)
        compare_tree(evaluation['roots'],oracle,f'evaluation/{step}')
        if step==n:
            for head in ('q','average'):
                require(weights[head].keys()==state['backend'][head].keys()
                        and all(torch.equal(value,state['backend'][head][key]) for key,value in weights[head].items()),'final public parameters differ from final learner')
        evaluations.append({'records':step,'average_ready':ready,'max_average_gap':max(max(r['average_gaps']) for r in oracle),
            'max_greedy_gap':max(max(r['greedy_q_gaps']) for r in oracle),'max_early_error':max(r['max_early_error'] for r in oracle)})
    final=evaluations[-5:];enough=len(final)==5;ready=enough and all(e['average_ready'] for e in final)
    average_gap=max(e['max_average_gap'] for e in final);greedy_gap=max(e['max_greedy_gap'] for e in final);error=max(e['max_early_error'] for e in final)
    credit=enough and greedy_gap<=.05 and error<=.1;band='short_game_pass' if ready and average_gap<=.05 else 'not_ready'
    passed=bool(ready and band=='short_game_pass' and (credit or config.task!='branching') and config.formal_protocol())
    result=json.loads((root/'result.json').read_text())
    require(result['status']=='complete' and result['records']==n and result['task']==config.task and result['learner']==config.learner
            and result['seed']==config.seed and result['source_sha256']==source_digest(),'completed result provenance differs')
    require(result['formal_protocol']==config.formal_protocol() and result['formal_case_passed']==passed
            and result['q_credit_passed']==credit and result['average_readiness']==band,'reported completed-run decisions differ')
    compare_tree(result['evaluations'],state['evaluations'],'reported evaluations')
    close([result['final_five_max_average_gap'],result['final_five_max_greedy_gap'],result['final_five_max_early_error']],
          [average_gap,greedy_gap,error],'reported final-five maxima differ')
    certificate=exact.certificate()
    return {'format_version':1,'audit_semantics':'graygo/independent-bounded-task-audit/1','verified':True,
        'task':config.task,'learner':config.learner,'seed':config.seed,'records':n,'eligibility_version':2,
        'source_sha256':source_digest(),'auditor_sha256':digest(Path(__file__)),'shared_helper_sha256':digest(HELPER),
        'training_manifest_sha256':digest(root/'run.json'),'result_sha256':digest(root/'result.json'),
        'checkpoint_manifest_sha256':digest(bundle/'manifest.json'),'episode_audit_sha256':digest(path),
        'archive_manifests_sha256':archive_hashes,'rational_root_values':[str(v) for v in exact.root_values],
        'exact_game_certificate':certificate,'exact_game_certificate_sha256':canonical_sha(certificate),
        'rational_value_states_solved':exact.nodes,'physical_complete_games':state['episodes'],
        'active_episode_steps':0 if not state['active'] else state['active']['episode']['steps'],
        'q_eligible_records':account['q_total'],'excluded_passive_root_records':n-account['q_total'],
        'ordinary_records':account['ordinary'],'eligible_labels':account['labels'],'accepted_reservoir_changes':account['changes'],
        'q_updates':counts['q_updates'],'q_loss_targets':counts['q_loss_targets'],'supervised_updates':counts['supervised_updates'],
        'target_refreshes':counts['target_refreshes'],'scheduled_evaluations':evaluations,
        'formal_protocol':config.formal_protocol(),'q_credit_passed':credit,'average_readiness':band,'formal_case_passed':passed,
        'a2_complete':False,**retained,
        'scope':'independent rational game values, complete-strategy fixed-policy responses and prefix-conditioned native values; exact header/RNG/eligibility/Algorithm-R clocks and retained legal joint transitions. Discarded historical actions/targets and original float64 trajectory hashes are not reconstructed; unresolved tabular publication tails retain stated interval bounds'}


def audit(root,*,require_formal=True,budget_profile=None):
    import fcntl
    started=time.monotonic();root=Path(root).resolve();threads=torch.get_num_threads()
    with (root/'.trainer.lock').open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        try:result=_audit(root,require_formal,budget_profile)
        finally:
            torch.set_num_threads(threads)
            fcntl.flock(lock,fcntl.LOCK_UN)
    result['measured_audit_entry_wall_seconds']=time.monotonic()-started
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_dir');parser.add_argument('--output',required=True)
    parser.add_argument('--budget-profile');parser.add_argument('--allow-diagnostic',action='store_true')
    parser.add_argument('--time-cap-seconds',type=int,default=3600);args=parser.parse_args()
    if not 0<args.time_cap_seconds<=14400:parser.error('audit watchdog must be positive and at most four elapsed hours; aggregate CPU stage budgeting remains separate')
    root=Path(args.run_dir).resolve();output=Path(args.output).resolve()
    if output.is_relative_to(root):parser.error('save audit output outside the original training inputs')
    import signal
    def stop(signum,frame):raise TimeoutError('original read-only audit watchdog exhausted; no success receipt written')
    signal.signal(signal.SIGALRM,stop);signal.alarm(args.time_cap_seconds)
    result=audit(root,require_formal=not args.allow_diagnostic,budget_profile=args.budget_profile);signal.alarm(0)
    result['audit_watchdog_seconds']=args.time_cap_seconds
    write_report(output,result);print(json.dumps({k:result[k] for k in ('verified','task','learner','records','formal_case_passed')},sort_keys=True))

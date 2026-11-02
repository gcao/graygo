"""Exact policy-gap and supported-value grading for NFSP task adapters."""
import numpy as np
import torch
from .engine import BLACK,WHITE
from .models import epsilon_greedy_distribution,greedy_distribution,masked_log_probs
from .public_game import markov_gaps
from .solving import SolveBudgetExceeded


@torch.no_grad()
def predict(adapter,backend,epsilon,batch_size=128):
    rows=[];values=[];targets=[];greedies=[]
    observations=adapter.observations;device=backend.device
    for start in range(0,len(observations),batch_size):
        chunk=observations[start:start+batch_size]
        x=torch.tensor(np.stack([o.spatial for o in chunk]),device=device,dtype=torch.float32)
        g=torch.tensor(np.stack([o.global_features for o in chunk]),device=device,dtype=torch.float32)
        legal=torch.tensor(np.stack([o.legal for o in chunk]),device=device)
        q=backend.q(x,g)
        if q.shape!=legal.shape or not bool(torch.isfinite(q[legal]).all()) or bool((q[legal].abs()>1+1e-6).any()):raise ValueError('invalid legal Q table')
        avg=masked_log_probs(backend.average(x,g),legal).exp()
        if not bool(torch.isfinite(avg).all()):raise ValueError('invalid deployed policy table')
        rows.append(avg.cpu().double().numpy());values.append(q.cpu().double().numpy())
        targets.append(epsilon_greedy_distribution(q,legal,epsilon).cpu().double().numpy())
        greedies.append(greedy_distribution(q,legal).cpu().double().numpy())
    shape=(len(adapter.states),2,adapter.actions)
    def policy(parts):
        a=np.concatenate(parts).reshape(shape);return a/a.sum(-1,keepdims=True)
    return np.concatenate(values).reshape(shape),policy(rows),policy(targets),policy(greedies)


def native_opponent_weights(adapter,backend):
    """One component is selected once per episode; weights follow the live ramp."""
    schedule=adapter.config.schedule();eta=schedule.best_response_probability
    average=(1-eta)*schedule.ramp(backend.counters.fresh_records) if backend.average_ready else 0.
    fallback=1-eta-average
    return np.array([eta if adapter.config.uniform_comparison else eta+fallback,average,
                     fallback if adapter.config.uniform_comparison else 0.])


def _native_q_values(adapter,root,player,target,components,weights):
    """Own target-policy values against a hidden frozen opponent component.

Posterior weights update from public moves. The bounded adapters retain their
complete two-round prefix; an incompatible public-state alias fails explicitly.
"""
    game=adapter.game;nodes=0;seen={};opponent=3-player
    def visit(state,belief):
        nonlocal nodes
        nodes+=1
        if nodes>adapter.config.max_nodes:raise SolveBudgetExceeded('native-opponent value budget exhausted')
        if game.terminal(state):return float(game.utility(state))*(1 if player==BLACK else -1)
        index=adapter.index[state];own=tuple(game.actions(state,player));other=tuple(game.actions(state,opponent))
        own_policy=target[index,player-1,list(own)]
        table=np.stack([component[index,opponent-1,list(other)] for component in components])
        joint=belief[:,None]*table;marginal=joint.sum(0);values=[]
        for action in own:
            value=0.
            for j,opposing in enumerate(other):
                if marginal[j]==0:continue
                b,w=(action,opposing) if player==BLACK else (opposing,action)
                value+=marginal[j]*visit(game.transition(state,b,w),joint[:,j]/marginal[j])
            values.append(float(value))
        value=float(own_policy@np.array(values))
        row={'value':value,'q':tuple(values),'belief':tuple(map(float,belief))}
        if state in seen and not np.allclose(seen[state]['q'],values,atol=1e-10,rtol=0):
            raise ValueError('public-state alias has incompatible history-conditioned Q values')
        seen[state]=row
        return value
    visit(root,np.asarray(weights,dtype=float));return seen


def evaluate_task(adapter,backend,epsilon,*,batch_size=128):
    q,average,target,greedy=predict(adapter,backend,epsilon,batch_size)
    game=adapter.game;results=[]
    weights=native_opponent_weights(adapter,backend)
    masks=np.stack([o.legal for o in adapter.observations]).reshape(target.shape)
    uniform=masks.astype(float)/masks.sum(-1,keepdims=True)
    components=(target,average,uniform)
    for root_index,(root,value) in enumerate(zip(adapter.roots,adapter.values)):
        i=adapter.index[root];early=[];late=[];root_actions=[]
        def append_error(state,role,expected,*,last=False):
            j=adapter.index[state];acts=game.actions(state,role);predicted=float(target[j,role-1,list(acts)]@q[j,role-1,list(acts)])
            row={'state_index':j,'player':role,'expected':float(expected),'predicted':predicted,'error':abs(float(expected)-predicted)}
            (late if last else early).append(row)
        if adapter.delayed:
            controller=root.controller;sign=1 if controller==BLACK else -1
            p0=target[i,controller-1,0]
            opponent_p0=sum(weight*component[i,controller-1,0] for weight,component in zip(weights,components))
            for role in (BLACK,WHITE):
                expected=(1.,-1.) if role==controller else (-(2*opponent_p0-1),)
                for action,truth in zip(game.actions(root,role),expected):
                    root_actions.append({'player':role,'action':action,'expected':float(truth),'predicted':float(q[i,role-1,action]),'error':float(abs(float(truth)-q[i,role-1,action]))})
                append_error(root,role,(2*p0-1) if role==controller else -(2*opponent_p0-1))
            for branch in (0,1):
                for depth in sorted({d for d in (1,2,5) if d<root.horizon}):
                    state=type(root)(root.horizon,controller,depth,branch)
                    for role in (BLACK,WHITE):append_error(state,role,(1 if branch==0 else -1)*(1 if role==controller else -1))
                state=type(root)(root.horizon,controller,root.horizon-1,branch)
                for role in (BLACK,WHITE):append_error(state,role,(1 if branch==0 else -1)*(1 if role==controller else -1),last=True)
            def gaps(policy):
                gap=2*(1-policy[i,controller-1,0])
                return [float(gap),0.] if controller==BLACK else [0.,float(gap)]
            avg_gaps,greedy_gaps=gaps(average),gaps(greedy)
        else:
            def policy(table,role):return lambda state:table[adapter.index[state],role-1,list(game.actions(state,role))]
            for role in (BLACK,WHITE):
                exact=_native_q_values(adapter,root,role,target,components,weights)
                for state,node in exact.items():append_error(state,role,node['value'])
                for action,truth in zip(game.actions(root,role),exact[root]['q']):
                    root_actions.append({'player':role,'action':action,'expected':float(truth),'predicted':float(q[i,role-1,action]),'error':float(abs(float(truth)-q[i,role-1,action]))})
            avg_gaps=list(markov_gaps(game,root,policy(average,BLACK),policy(average,WHITE),value,max_nodes=adapter.config.max_nodes))
            greedy_gaps=list(markov_gaps(game,root,policy(greedy,BLACK),policy(greedy,WHITE),value,max_nodes=adapter.config.max_nodes))
        results.append({'root_index':root_index,'root_tag':adapter.root_tags[root_index],'value':value,
              'native_opponent_weights':weights.tolist(),
              'average_gaps':avg_gaps,'greedy_q_gaps':greedy_gaps,'root_action_values':root_actions,
              'early_values':early,'late_values':late,'max_early_error':max(r['error'] for r in early+root_actions),
              'max_late_error':max((r['error'] for r in late),default=None),
              'root_average_policies':average[i].tolist(),'root_greedy_policies':greedy[i].tolist()})
    return results

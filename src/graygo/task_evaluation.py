"""Exact policy-gap and supported-value grading for NFSP task adapters."""
import numpy as np
import torch
from .engine import BLACK,WHITE
from .models import epsilon_greedy_distribution,greedy_distribution,masked_log_probs
from .public_game import evaluate_markov,markov_gaps


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


def evaluate_task(adapter,backend,epsilon,*,batch_size=128):
    q,average,target,greedy=predict(adapter,backend,epsilon,batch_size)
    game=adapter.game;results=[]
    for root_index,(root,value) in enumerate(zip(adapter.roots,adapter.values)):
        i=adapter.index[root];early=[];late=[];root_actions=[]
        def append_error(state,role,expected,*,last=False):
            j=adapter.index[state];acts=game.actions(state,role);predicted=float(target[j,role-1,list(acts)]@q[j,role-1,list(acts)])
            row={'state_index':j,'player':role,'expected':float(expected),'predicted':predicted,'error':abs(float(expected)-predicted)}
            (late if last else early).append(row)
        if adapter.delayed:
            controller=root.controller;sign=1 if controller==BLACK else -1
            p0=target[i,controller-1,0]
            for role in (BLACK,WHITE):
                expected=(1.,-1.) if role==controller else (-(2*p0-1),)
                for action,truth in zip(game.actions(root,role),expected):
                    root_actions.append({'player':role,'action':action,'expected':float(truth),'predicted':float(q[i,role-1,action]),'error':float(abs(float(truth)-q[i,role-1,action]))})
                append_error(root,role,(2*p0-1)*(1 if role==controller else -1))
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
            exact=evaluate_markov(game,root,policy(target,BLACK),policy(target,WHITE),max_nodes=adapter.config.max_nodes)
            for state,node in exact.items():
                if game.terminal(state):continue
                for role in (BLACK,WHITE):append_error(state,role,node.value*(1 if role==BLACK else -1))
            node=exact[root]
            for role,truths in ((BLACK,node.black_q),(WHITE,node.white_q)):
                for action,truth in zip(game.actions(root,role),truths):
                    root_actions.append({'player':role,'action':action,'expected':float(truth),'predicted':float(q[i,role-1,action]),'error':float(abs(float(truth)-q[i,role-1,action]))})
            avg_gaps=list(markov_gaps(game,root,policy(average,BLACK),policy(average,WHITE),value,max_nodes=adapter.config.max_nodes))
            greedy_gaps=list(markov_gaps(game,root,policy(greedy,BLACK),policy(greedy,WHITE),value,max_nodes=adapter.config.max_nodes))
        results.append({'root_index':root_index,'root_tag':adapter.root_tags[root_index],'value':value,
              'average_gaps':avg_gaps,'greedy_q_gaps':greedy_gaps,'root_action_values':root_actions,
              'early_values':early,'late_values':late,'max_early_error':max(r['error'] for r in early+root_actions),
              'max_late_error':max((r['error'] for r in late),default=None),
              'root_average_policies':average[i].tolist(),'root_greedy_policies':greedy[i].tolist()})
    return results

"""Finite complete-public-history evaluation of independent episode mixtures.

Components are pure callables (state, public_action_history) -> probabilities in
legal-action order. A component is selected once per episode, not every state.
Bayesian weights carry that latent selection through the full observed history.
Floating-point arithmetic is used; no belief rounding or heuristic cutoffs occur.
"""
from dataclasses import dataclass
import numpy as np
from .engine import BLACK,WHITE,resolve,score
from .solving import SolveBudgetExceeded


class GrayGoGame:
    @staticmethod
    def actions(state,player):return state.legal_actions(player)
    @staticmethod
    def transition(state,black,white):return resolve(state,black,white).state
    @staticmethod
    def terminal(state):return state.done
    @staticmethod
    def utility(state):return float(score(state.board,state.rules.size,scoring=state.rules.scoring).utility_black)


@dataclass(frozen=True)
class StrategyEvaluation:
    black_value: float
    visited_nodes: int


def _priors(components,weights):
    if not components:raise ValueError("nonempty strategy mixture required")
    p=np.asarray(weights,dtype=np.float64)
    if p.shape!=(len(components),) or not np.isfinite(p).all() or (p<0).any() or abs(p.sum()-1)>1e-10:
        raise ValueError("mixture weights must be normalized and nonnegative")
    return p/p.sum()


def _terminal_utility(game,state):
    value=float(game.utility(state))
    if not np.isfinite(value):raise ValueError("nonfinite terminal utility")
    return value


def condition_episode_mixture(game,root,components,weights,player,prefix):
    """Condition a player's component prior on an observed public action prefix.

Pass the returned belief/state/history into evaluation after a prefix; using the
unconditional prior again would define a fresh component selection instead.
"""
    if player not in (BLACK,WHITE):raise ValueError("invalid component owner")
    belief=_priors(components,weights);state=root;history=()
    for b,w in prefix:
        if game.terminal(state):raise ValueError("prefix continues past termination")
        black,white=game.actions(state,BLACK),game.actions(state,WHITE)
        if b not in black or w not in white:raise ValueError("illegal prefix move")
        actions=black if player==BLACK else white
        selected=b if player==BLACK else w
        joint,marginal=_predict(components,belief,state,history,actions)
        index=actions.index(selected)
        if marginal[index]==0:raise ValueError("prefix has zero probability under this mixture")
        belief=joint[:,index]/marginal[index]
        state=game.transition(state,b,w);history+=((b,w),)
    return state,belief,history


def _predict(components,belief,state,history,actions):
    table=np.zeros((len(components),len(actions)),dtype=np.float64)
    for index,component in enumerate(components):
        if belief[index]==0:continue
        row=np.asarray(component(state,history),dtype=np.float64)
        if row.shape!=(len(actions),) or not np.isfinite(row).all() or (row<0).any() or abs(row.sum()-1)>1e-7:
            raise ValueError("component must return a legal normalized distribution")
        table[index]=row/row.sum()
    joint=belief[:,None]*table
    marginal=joint.sum(0)
    return joint,marginal


def episode_best_response(game,root,components,weights,player,*,max_nodes=100000,history=()):
    """Optimal black utility when player responds to a hidden opponent component.

For player WHITE this minimizes black utility. The responder observes both past
moves, but never the current opposing move or the sampled component identity.
"""
    if player not in (BLACK,WHITE) or max_nodes<1:raise ValueError("invalid best-response settings")
    initial=_priors(components,weights);nodes=0
    def visit(state,belief,history):
        nonlocal nodes
        nodes+=1
        if nodes>max_nodes:raise SolveBudgetExceeded("complete-history best-response node budget exhausted")
        if game.terminal(state):return _terminal_utility(game,state)
        black,white=game.actions(state,BLACK),game.actions(state,WHITE)
        opponent=white if player==BLACK else black
        joint,marginal=_predict(components,belief,state,history,opponent)
        values=[]
        for own in (black if player==BLACK else white):
            expected=0.
            for index,other in enumerate(opponent):
                mass=marginal[index]
                if mass==0:continue
                b,w=(own,other) if player==BLACK else (other,own)
                expected+=mass*visit(game.transition(state,b,w),joint[:,index]/mass,history+((b,w),))
            values.append(expected)
        return max(values) if player==BLACK else min(values)
    return StrategyEvaluation(float(visit(root,initial,tuple(history))),nodes)


def evaluate_episode_mixtures(game,root,black_components,black_weights,white_components,white_weights,*,max_nodes=100000,history=()):
    """Expected payoff under independent per-player episode component draws."""
    if max_nodes<1:raise ValueError("positive node budget required")
    initial_b=_priors(black_components,black_weights);initial_w=_priors(white_components,white_weights);nodes=0
    def visit(state,belief_b,belief_w,history):
        nonlocal nodes
        nodes+=1
        if nodes>max_nodes:raise SolveBudgetExceeded("complete-history evaluation node budget exhausted")
        if game.terminal(state):return _terminal_utility(game,state)
        black,white=game.actions(state,BLACK),game.actions(state,WHITE)
        jb,p=_predict(black_components,belief_b,state,history,black)
        jw,q=_predict(white_components,belief_w,state,history,white)
        expected=0.
        for i,b in enumerate(black):
            if p[i]==0:continue
            for j,w in enumerate(white):
                if q[j]==0:continue
                expected+=p[i]*q[j]*visit(game.transition(state,b,w),jb[:,i]/p[i],jw[:,j]/q[j],history+((b,w),))
        return expected
    return StrategyEvaluation(float(visit(root,initial_b,initial_w,tuple(history))),nodes)


def episode_value_gaps(game,root,value,black_components,black_weights,white_components,white_weights,*,max_nodes=100000,history=()):
    if not np.isfinite(value):raise ValueError("finite solved value required")
    lower=episode_best_response(game,root,black_components,black_weights,WHITE,max_nodes=max_nodes,history=history)
    upper=episode_best_response(game,root,white_components,white_weights,BLACK,max_nodes=max_nodes,history=history)
    return {"row_gap":max(0.,float(value-lower.black_value)),
            "column_gap":max(0.,float(upper.black_value-value)),
            "response_to_black_nodes":lower.visited_nodes,"response_to_white_nodes":upper.visited_nodes}

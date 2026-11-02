"""Exact finite public-state Markov controls for task adapters.

These memoized calculations support Markov policies only. Hidden episode-level
checkpoint mixtures use strategies.py, which carries complete-history beliefs.
Budgets fail explicitly and never substitute neural or heuristic leaf values.
"""
from dataclasses import dataclass
import numpy as np
from .engine import BLACK,WHITE
from .solving import solve_matrix,SolveBudgetExceeded


def utility(game,state):
    result=float(game.utility(state))
    if not np.isfinite(result) or not -1<=result<=1:raise ValueError('finite bounded terminal utility required')
    return result


def distribution(policy,state,actions):
    result=np.asarray(policy(state),dtype=float)
    if result.shape!=(len(actions),) or not np.isfinite(result).all() or (result<0).any() or abs(result.sum()-1)>1e-7:
        raise ValueError('normalized legal-action distribution required')
    return result/result.sum()


class PublicGameSolver:
    def __init__(self,game,*,max_nodes=100000):
        if type(max_nodes) is not int or max_nodes<1:raise ValueError('positive finite-game node budget required')
        self.game,self.max_nodes=game,max_nodes;self.nodes=0;self.cache={};self.active=set();self.max_residual_gap=0.

    def solve(self,state):
        if state in self.cache:return self.cache[state]
        if state in self.active:raise ValueError('finite task contains an unbounded state cycle')
        if self.nodes>=self.max_nodes:raise SolveBudgetExceeded('public-game solve budget exhausted')
        self.nodes+=1;self.active.add(state)
        try:
            if self.game.terminal(state):result=(utility(self.game,state),(),(),None)
            else:
                black,white=tuple(self.game.actions(state,BLACK)),tuple(self.game.actions(state,WHITE))
                if not black or not white:raise ValueError('nonterminal action sets cannot be empty')
                matrix=np.array([[self.solve(self.game.transition(state,b,w))[0] for w in white] for b in black])
                solved=solve_matrix(matrix);self.max_residual_gap=max(self.max_residual_gap,solved.residual_gap)
                result=(solved.value,black,white,solved)
            self.cache[state]=result
            return result
        finally:self.active.remove(state)

    def policy(self,state,player):
        if player not in (BLACK,WHITE):raise ValueError('invalid role')
        solved=self.solve(state)[3]
        return np.array([]) if solved is None else (solved.row if player==BLACK else solved.column)


@dataclass(frozen=True)
class PolicyNode:
    value: float
    black_actions: tuple
    white_actions: tuple
    black_q: tuple  # own-role utility after each possible own action
    white_q: tuple


def evaluate_markov(game,root,black_policy,white_policy,*,max_nodes=100000):
    """Exact values and own-action values for a declared fixed policy pair."""
    if type(max_nodes) is not int or max_nodes<1:raise ValueError('positive evaluation budget required')
    cache={};active=set()
    def visit(state):
        if state in cache:return cache[state]
        if state in active:raise ValueError('cyclic finite task')
        if len(cache)+len(active)>=max_nodes:raise SolveBudgetExceeded('public Markov evaluation budget exhausted')
        active.add(state)
        try:
            if game.terminal(state):result=PolicyNode(utility(game,state),(),(),(),())
            else:
                black,white=tuple(game.actions(state,BLACK)),tuple(game.actions(state,WHITE))
                p,q=distribution(black_policy,state,black),distribution(white_policy,state,white)
                a=np.array([[visit(game.transition(state,b,w)).value for w in white] for b in black])
                result=PolicyNode(float(p@a@q),black,white,tuple(a@q),tuple(-(p@a)))
            cache[state]=result;return result
        finally:active.remove(state)
    visit(root)
    return cache


def markov_response(game,root,opponent,player,*,max_nodes=100000):
    """Black utility when player optimally responds to a fixed Markov opponent."""
    if player not in (BLACK,WHITE) or type(max_nodes) is not int or max_nodes<1:raise ValueError('invalid best-response settings')
    cache={};active=set()
    def visit(state):
        if state in cache:return cache[state]
        if state in active:raise ValueError('cyclic finite task')
        if len(cache)+len(active)>=max_nodes:raise SolveBudgetExceeded('public Markov response budget exhausted')
        active.add(state)
        try:
            if game.terminal(state):value=utility(game,state)
            else:
                black,white=tuple(game.actions(state,BLACK)),tuple(game.actions(state,WHITE))
                a=np.array([[visit(game.transition(state,b,w)) for w in white] for b in black])
                if player==BLACK:value=float((a@distribution(opponent,state,white)).max())
                else:value=float((distribution(opponent,state,black)@a).min())
            cache[state]=value;return value
        finally:active.remove(state)
    return visit(root)


def markov_gaps(game,root,black_policy,white_policy,value,*,max_nodes=100000):
    if not np.isfinite(value):raise ValueError('finite solved value required')
    lower=markov_response(game,root,black_policy,WHITE,max_nodes=max_nodes)
    upper=markov_response(game,root,white_policy,BLACK,max_nodes=max_nodes)
    return max(0.,float(value-lower)),max(0.,float(upper-value))

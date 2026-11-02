"""Certified numerical matrix solutions and exhaustive finite Gray Go games.

No heuristic leaf values are used. A budget failure produces no claimed solution.
The best-response evaluator here accepts public-state Markov policies; episode
mixtures require a history/latent-strategy evaluator before their A2 acceptance.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.optimize import linprog
from .engine import BLACK, WHITE, State, resolve, score


@dataclass(frozen=True)
class MatrixSolution:
    value: float
    row: np.ndarray
    column: np.ndarray
    residual_gap: float


def solve_matrix(payoffs):
    a = np.asarray(payoffs, dtype=np.float64)
    if a.ndim != 2 or not all(a.shape) or not np.isfinite(a).all():
        raise ValueError("finite nonempty payoff matrix required")
    m, n = a.shape
    row = linprog(np.r_[np.zeros(m), -1.], A_ub=np.c_[-a.T, np.ones(n)], b_ub=np.zeros(n),
                  A_eq=[np.r_[np.ones(m),0]], b_eq=[1], bounds=[(0,None)]*m+[(None,None)], method="highs")
    column = linprog(np.r_[np.zeros(n), 1.], A_ub=np.c_[a, -np.ones(m)], b_ub=np.zeros(m),
                     A_eq=[np.r_[np.ones(n),0]], b_eq=[1], bounds=[(0,None)]*n+[(None,None)], method="highs")
    if not row.success or not column.success:
        raise RuntimeError(f"matrix optimization failed: {row.message}; {column.message}")
    p, q = np.maximum(row.x[:-1], 0), np.maximum(column.x[:-1], 0)
    p, q = p/p.sum(), q/q.sum()
    lower, upper = float((p@a).min()), float((a@q).max())
    if upper-lower > 1e-7:
        raise RuntimeError(f"matrix certificate gap too large: {upper-lower}")
    p.flags.writeable = q.flags.writeable = False
    return MatrixSolution((lower+upper)/2, p, q, max(0.,upper-lower))


def value_gaps(payoffs, row, column, value=None):
    a = np.asarray(payoffs, dtype=np.float64)
    p, q = np.asarray(row, dtype=float), np.asarray(column, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all() or p.shape != (a.shape[0],) or q.shape != (a.shape[1],):
        raise ValueError("policy/matrix dimensions do not match")
    for policy in (p,q):
        if not np.isfinite(policy).all() or (policy < 0).any() or abs(policy.sum()-1) > 1e-7:
            raise ValueError("policy must be a normalized nonnegative distribution")
    value = solve_matrix(a).value if value is None else value
    return max(0., float(value-(p@a).min())), max(0., float((a@q).max()-value))


class SolveBudgetExceeded(RuntimeError):
    pass


class FiniteSolver:
    def __init__(self, *, max_nodes=100_000):
        if max_nodes < 1:
            raise ValueError("positive node budget required")
        self.max_nodes, self.nodes = max_nodes, 0
        self.max_matrix_gap = 0.0
        self._cache = {}

    def solve(self, state):
        if state in self._cache:
            return self._cache[state]
        if self.nodes >= self.max_nodes:
            raise SolveBudgetExceeded(f"finite solver exhausted {self.max_nodes} states")
        self.nodes += 1
        if state.done:
            value = float(score(state.board, state.rules.size).utility_black)
            result = (value, (), (), None)
        else:
            black, white = state.legal_actions(BLACK), state.legal_actions(WHITE)
            a = np.empty((len(black),len(white)))
            for i, b in enumerate(black):
                for j, w in enumerate(white):
                    a[i,j] = self.solve(resolve(state,b,w).state)[0]
            solution = solve_matrix(a)
            self.max_matrix_gap = max(self.max_matrix_gap, solution.residual_gap)
            a.flags.writeable = False
            result = (solution.value, black, white, solution)
        self._cache[state] = result
        return result

    def policy(self, state, player):
        if player not in (BLACK,WHITE):
            raise ValueError("invalid player")
        _, black, white, solution = self.solve(state)
        if solution is None:
            return np.array([])
        return solution.row if player == BLACK else solution.column

    def numerical_value_error_bound(self, state):
        # Value is 1-Lipschitz in the child payoff entries. Each remaining
        # matrix solve adds at most half its primal/dual certificate width.
        return (state.rules.max_rounds-state.rounds_played)*self.max_matrix_gap/2


def markov_best_response(state, opponent, player, *, max_nodes=100_000):
    """Black utility with one player best-responding to a fixed Markov opponent."""
    if player not in (BLACK,WHITE):
        raise ValueError("invalid player")
    count = 0
    @lru_cache(maxsize=None)
    def visit(s):
        nonlocal count
        count += 1
        if count > max_nodes:
            raise SolveBudgetExceeded("best-response state budget exhausted")
        if s.done:
            return float(score(s.board,s.rules.size).utility_black)
        black, white = s.legal_actions(BLACK), s.legal_actions(WHITE)
        probs = np.asarray(opponent(s), dtype=float)
        expected_shape = (len(white) if player == BLACK else len(black),)
        if probs.shape != expected_shape or not np.isfinite(probs).all() or (probs < 0).any() or abs(probs.sum()-1)>1e-7:
            raise ValueError("invalid opponent Markov distribution")
        if player == BLACK:
            return max(sum(float(probs[j])*visit(resolve(s,b,w).state)
                           for j,w in enumerate(white) if probs[j]>0) for b in black)
        return min(sum(float(probs[i])*visit(resolve(s,b,w).state)
                       for i,b in enumerate(black) if probs[i]>0) for w in white)
    return visit(state)

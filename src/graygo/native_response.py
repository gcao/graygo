"""Observable best responses to the branching task's hidden episode mixture.

The opponent selects its component once per episode. A response may condition
on public moves, never the hidden component identifier. Keep this scoped to the
two-round branching adapter, whose public state retains the complete prefix.
"""
import numpy as np
from .engine import BLACK
from .solving import SolveBudgetExceeded


def branching_response(adapter, root, player, components, weights, greedy):
    if adapter.config.task != 'branching' or player not in (1, 2):
        raise ValueError('native-response grading requires the two-round branching task and a valid role')
    components = tuple(components)
    belief = np.asarray(weights, dtype=np.float64)
    if (not components or belief.shape != (len(components),) or not np.isfinite(belief).all()
            or (belief < 0).any() or abs(float(belief.sum()) - 1) > 1e-12):
        raise ValueError('complete normalized native episode-mixture weights required')
    shape = (len(adapter.states), 2, adapter.actions)
    if any(np.asarray(table).shape != shape for table in (*components, greedy)):
        raise ValueError('complete public-state policy tables required')
    game = adapter.game
    opponent = 3 - player

    def policy(table, state, role, legal):
        values = np.asarray(table[adapter.index[state], role - 1, list(legal)], dtype=np.float64)
        if (not np.isfinite(values).all() or (values < 0).any()
                or abs(float(values.sum()) - 1) > 1e-10):
            raise ValueError('complete normalized legal policy distribution required')
        return values

    nodes = 0

    def visit(state, posterior):
        nonlocal nodes
        nodes += 1
        if nodes > adapter.config.max_nodes:
            raise SolveBudgetExceeded('native best-response node budget exhausted')
        if game.terminal(state):
            value = float(game.utility(state)) * (1 if player == BLACK else -1)
            return value, value
        own = tuple(game.actions(state, player))
        other = tuple(game.actions(state, opponent))
        table = np.stack([policy(component, state, opponent, other) for component in components])
        joint = posterior[:, None] * table
        marginal = joint.sum(0)
        best_values = []
        greedy_values = []
        for action in own:
            best_value = greedy_value = 0.
            for index, move in enumerate(other):
                probability = float(marginal[index])
                if probability == 0:
                    continue
                black, white = (action, move) if player == BLACK else (move, action)
                best, actual = visit(game.transition(state, black, white), joint[:, index] / probability)
                best_value += probability * best
                greedy_value += probability * actual
            best_values.append(best_value)
            greedy_values.append(greedy_value)
        # Maximize only after summing the hidden components. A fixed candidate
        # uses its own public-state policy for the same opponent mixture.
        return max(best_values), float(policy(greedy, state, player, own) @ np.asarray(greedy_values))

    best, actual = visit(root, belief)
    gap = best - actual
    if not np.isfinite([best, actual, gap]).all() or not -1e-12 <= gap <= 2 + 1e-12:
        raise ValueError('invalid native response value or gap')
    return {'player': player, 'best_response_value': float(best),
            'greedy_value': float(actual), 'gap': max(0., float(gap))}

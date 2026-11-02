"""Frozen baseline v1: seeded random ties, no access to current commitments.

Tactical candidate effects are evaluated with the opponent passing. Priorities:
1. Most opposing (including gray) stones captured.
2. Most own atari stones saved in a surviving group with >=2 liberties.
3. Uniform safe placement (>=2 liberties) that does not fill a simple own eye.
4. Pass. All tied candidates are sampled uniformly using private caller RNG.
"""
import random
from .engine import BLACK, WHITE, EMPTY, groups, neighbors, PlacementResolver, resolve_reference as resolve

BASELINE_VERSION = "graygo/baselines/1"


def uniform(state, player, rng: random.Random):
    actions = state.legal_actions(player)
    if not actions:
        raise ValueError("cannot choose an action at a terminal state")
    return rng.choice(actions)


def _tactical_reference(state, player, rng: random.Random):
    """Original resolver oracle retained verbatim after this docstring."""
    actions = state.legal_actions(player)
    if not actions:
        raise ValueError("cannot choose an action at a terminal state")
    passed = state.rules.pass_action
    atari = frozenset(p for group in groups(state.board, state.rules.size)
                      if group.color == player and len(group.liberties) == 1 for p in group.stones)
    ranked = []
    for action in actions:
        if action == passed:
            continue
        result = resolve(state, action, passed) if player == BLACK else resolve(state, passed, action)
        captures = sum(len(g.stones) for g in result.captured if g.color != player)
        after_groups = groups(result.state.board, state.rules.size)
        own_group = next((g for g in after_groups if action in g.stones and g.color == player), None)
        safe = own_group is not None and len(own_group.liberties) >= 2
        saved = len(own_group.stones & atari) if safe else 0
        own_eye = all(state.board[p] == player for p in neighbors(state.rules.size)[action])
        if captures:
            rank = (3, captures)
        elif saved:
            rank = (2, saved)
        elif safe and not own_eye:
            rank = (1, 0)
        else:
            continue
        ranked.append((rank, action))
    if not ranked:
        return passed
    best = max(rank for rank, _ in ranked)
    return rng.choice([action for rank, action in ranked if rank == best])


def _tactical_ranks(state, player):
    """Frozen v1 ranks using complete prepared game-resolver outcomes."""
    actions = state.legal_actions(player)
    if not actions:
        raise ValueError("cannot choose an action at a terminal state")
    board, rules = state.board, state.rules
    adjacent = neighbors(rules.size)
    prepared = PlacementResolver(state, player)
    atari = frozenset(p for group in prepared.original_groups
                      if group.color == player and len(group.liberties) == 1 for p in group.stones)
    ranked = []
    for action in actions:
        if action == rules.pass_action:
            continue
        outcome = prepared.resolve(action)
        captures = sum(len(group.stones) for group in outcome.transition.captured if group.color != player)
        if captures:
            ranked.append(((3, captures), action))
            continue
        own_group = outcome.placed_group
        if own_group is None or len(own_group.liberties) < 2:
            continue
        saved = len(own_group.stones & atari)
        if saved:
            ranked.append(((2, saved), action))
        elif not all(board[p] == player for p in adjacent[action]):
            ranked.append(((1, 0), action))
    return ranked


def tactical(state, player, rng: random.Random):
    """Frozen v1 semantics with one shared analysis of the pre-round board."""
    ranked = _tactical_ranks(state, player)
    if not ranked:
        return state.rules.pass_action
    best = max(rank for rank, _ in ranked)
    return rng.choice([action for rank, action in ranked if rank == best])

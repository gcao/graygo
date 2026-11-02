"""Frozen baseline v1: seeded random ties, no access to current commitments.

Tactical candidate effects are evaluated with the opponent passing. Priorities:
1. Most opposing (including gray) stones captured.
2. Most own atari stones saved in a surviving group with >=2 liberties.
3. Uniform safe placement (>=2 liberties) that does not fill a simple own eye.
4. Pass. All tied candidates are sampled uniformly using private caller RNG.
"""
import random
from .engine import BLACK, WHITE, EMPTY, groups, neighbors, resolve

BASELINE_VERSION = "graygo/baselines/1"


def uniform(state, player, rng: random.Random):
    actions = state.legal_actions(player)
    if not actions:
        raise ValueError("cannot choose an action at a terminal state")
    return rng.choice(actions)


def tactical(state, player, rng: random.Random):
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

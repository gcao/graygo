"""Versioned perspective-relative observations; v2 history hints are conditional."""
from dataclasses import dataclass
import numpy as np
from .engine import BLACK, WHITE, GRAY, EMPTY, State, groups

ENCODING_VERSION = "graygo/encoding/1"
SPATIAL_CHANNELS, GLOBAL_FEATURES = 30, 14
KO_ENCODING_VERSION = "graygo/encoding/ko-hints/2"


@dataclass(frozen=True)
class Observation:
    spatial: np.ndarray  # uint8 [30 or 32,N,N]
    global_features: np.ndarray  # float32 [14]
    legal: np.ndarray  # bool [N*N+1]; all false at true terminal


def encode(state: State, player: int, *, ko_hints=False, hint_points=None) -> Observation:
    state.forbidden(player)  # validate player even on an empty board
    n, points = state.rules.size, state.rules.points
    opponent = WHITE if player == BLACK else BLACK
    channels=32 if ko_hints else 30
    x = np.zeros((channels, points), dtype=np.uint8)
    g = np.zeros(14, dtype=np.float32)
    faction = {player: 0, opponent: 1, GRAY: 2, EMPTY: 3}
    for p, color in enumerate(state.board):
        x[faction[color], p] = 1
    for group in groups(state.board, n):
        channel = 4 + 4 * faction[group.color] + min(len(group.liberties), 4) - 1
        x[channel, list(group.stones)] = 1
    # Fixed slots: older record 0, newest record 1; missing entries are zero.
    for slot, record in enumerate(state.history, start=2 - len(state.history)):
        g[slot] = 1
        for p, color in enumerate(record.before):
            x[16 + 4 * slot + faction[color], p] = 1
        mine, theirs = (record.black, record.white) if player == BLACK else (record.white, record.black)
        for who, move in enumerate((mine, theirs)):
            if move == points:
                g[2 + 2 * slot + who] = 1
            else:
                x[24 + 2 * slot + who, move] = 1
    x[28, list(state.forbidden(player))] = 1
    x[29, list(state.forbidden(opponent))] = 1
    remaining = state.rules.max_rounds - state.rounds_played
    g[6] = state.pass_streak
    g[7] = remaining / state.rules.max_rounds
    g[8:] = [remaining <= k for k in (1, 2, 3, 5, 10, 25)]
    legal = np.zeros(points + 1, dtype=np.bool_)
    legal[list(state.legal_actions(player))] = True
    if ko_hints:
        from .engine import solo_repeat_hints
        hints=solo_repeat_hints(state) if hint_points is None else hint_points
        x[30,list(hints[player-1])]=1
        x[31,list(hints[opponent-1])]=1
    x = x.reshape(channels, n, n)
    for array in (x, g, legal):
        array.flags.writeable = False
    return Observation(x, g, legal)

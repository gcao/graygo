"""Gating evaluation for Gray Go v6 — uses v6 model inference.

Same logic as gate_v4 but with v6 predict/predict_batch functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from graygo_engine import BLACK_PLAYER, WHITE_PLAYER, GameState
except ImportError:
    from engine import BLACK_PLAYER, WHITE_PLAYER, GameState

from model import encode_player_relative, predict_batch


@dataclass
class GateConfig:
    board_size: int = 9
    num_games: int = 100
    max_turns: int = 100
    rng_seed: Optional[int] = None


@dataclass
class GateResult:
    win_rate: float
    wins: int
    losses: int
    draws: int


def _normalize(policy: np.ndarray, legal: np.ndarray) -> np.ndarray:
    p = policy.astype(np.float64).copy()
    p[~legal] = 0.0
    total = p.sum()
    if total <= 0:
        p[:] = 0.0
        p[-1] = 1.0
    else:
        p /= total
    return p.astype(np.float32)


def evaluate_models(candidate, champion, cfg: GateConfig) -> GateResult:
    """Play N games: candidate vs champion using color-flip inference."""
    rng = np.random.default_rng(cfg.rng_seed)
    N = cfg.num_games
    s = cfg.board_size

    games = [GameState(size=s) for _ in range(N)]
    cand_is_black = [(i % 2) == 0 for i in range(N)]
    active = list(range(N))

    turn = 0
    while active and turn < cfg.max_turns:
        turn += 1
        n_active = len(active)

        black_states = np.empty((n_active, 6, s, s), dtype=np.float32)
        white_states = np.empty((n_active, 6, s, s), dtype=np.float32)
        for i, gi in enumerate(active):
            black_states[i] = encode_player_relative(games[gi], BLACK_PLAYER, s)
            white_states[i] = encode_player_relative(games[gi], WHITE_PLAYER, s)

        cand_bp_all, _ = predict_batch(candidate, black_states)
        cand_wp_all, _ = predict_batch(candidate, white_states)
        champ_bp_all, _ = predict_batch(champion, black_states)
        champ_wp_all, _ = predict_batch(champion, white_states)

        still_active = []
        for i, gi in enumerate(active):
            g = games[gi]
            legal_b = g.legal_actions(BLACK_PLAYER).astype(bool)
            legal_w = g.legal_actions(WHITE_PLAYER).astype(bool)

            if cand_is_black[gi]:
                bp = _normalize(cand_bp_all[i], legal_b)
                wp = _normalize(champ_wp_all[i], legal_w)
            else:
                bp = _normalize(champ_bp_all[i], legal_b)
                wp = _normalize(cand_wp_all[i], legal_w)

            b_action = int(rng.choice(len(bp), p=bp))
            w_action = int(rng.choice(len(wp), p=wp))

            g.step(b_action, w_action)
            if not g.game_over:
                still_active.append(gi)

        active = still_active

    wins = losses = draws = 0
    for gi in range(N):
        winner = games[gi].winner_player()
        if winner is None or winner == -1:
            draws += 1
        elif (cand_is_black[gi] and winner == BLACK_PLAYER) or \
             (not cand_is_black[gi] and winner == WHITE_PLAYER):
            wins += 1
        else:
            losses += 1

    win_rate = (wins + 0.5 * draws) / float(N)
    return GateResult(win_rate=win_rate, wins=wins, losses=losses, draws=draws)

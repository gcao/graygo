"""Reduced-visit MCTS gating evaluation for Gray Go v6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from graygo_engine import BLACK_PLAYER, WHITE_PLAYER, GameState
except ImportError:
    from engine import BLACK_PLAYER, WHITE_PLAYER, GameState

from model import encode_player_relative, predict
from selfplay import run_mcts_with_eval_fn, sample_joint_move


@dataclass
class GateConfig:
    board_size: int = 9
    num_games: int = 100
    max_turns: int = 100
    rng_seed: Optional[int] = None

    # Reduced-visit search used for promotion gating.
    num_visits: int = 64
    c_puct: float = 1.5
    tau: float = 0.01
    move_temperature: float = 0.01
    max_candidates: int = 20


@dataclass
class GateResult:
    win_rate: float
    wins: int
    losses: int
    draws: int


def _make_match_eval_fn(black_model, white_model, board_size: int):
    """Build an MCTS callback for a black-model vs white-model game."""
    def eval_fn(state):
        black_encoded = encode_player_relative(state, BLACK_PLAYER, board_size)
        white_encoded = encode_player_relative(state, WHITE_PLAYER, board_size)

        bp, bv = predict(black_model, black_encoded)
        wp, wv = predict(white_model, white_encoded)

        legal_b = state.legal_actions(BLACK_PLAYER).astype(np.uint8)
        legal_w = state.legal_actions(WHITE_PLAYER).astype(np.uint8)
        game_over = state.game_over

        if game_over:
            winner = state.winner_player()
            if winner == BLACK_PLAYER:
                terminal_value = 1.0
            elif winner == WHITE_PLAYER:
                terminal_value = -1.0
            else:
                terminal_value = 0.0
        else:
            terminal_value = 0.0

        return (
            bp.astype(np.float32),
            wp.astype(np.float32),
            float((bv - wv) / 2.0),
            legal_b,
            legal_w,
            game_over,
            terminal_value,
        )

    return eval_fn


def evaluate_models(candidate, champion, cfg: GateConfig) -> GateResult:
    """Play N reduced-visit joint-MCTS games: candidate vs champion."""
    rng = np.random.default_rng(cfg.rng_seed)
    wins = losses = draws = 0

    for game_idx in range(cfg.num_games):
        state = GameState(size=cfg.board_size)
        candidate_is_black = (game_idx % 2) == 0

        if candidate_is_black:
            black_model = candidate
            white_model = champion
        else:
            black_model = champion
            white_model = candidate

        eval_fn = _make_match_eval_fn(black_model, white_model, cfg.board_size)

        turn = 0
        while not state.game_over and turn < cfg.max_turns:
            root = run_mcts_with_eval_fn(
                state,
                eval_fn,
                num_visits=cfg.num_visits,
                c_puct=cfg.c_puct,
                tau=cfg.tau,
                rng=rng,
                dirichlet_alpha=0.0,
                dirichlet_epsilon=0.0,
                add_noise=False,
                max_candidates=cfg.max_candidates,
            )
            b_action, w_action = sample_joint_move(root, cfg.move_temperature, rng)
            state.step(b_action, w_action)
            turn += 1

        winner = state.winner_player()
        if winner is None or winner == -1:
            draws += 1
        elif (candidate_is_black and winner == BLACK_PLAYER) or \
             (not candidate_is_black and winner == WHITE_PLAYER):
            wins += 1
        else:
            losses += 1

    win_rate = (wins + 0.5 * draws) / float(cfg.num_games)
    return GateResult(win_rate=win_rate, wins=wins, losses=losses, draws=draws)

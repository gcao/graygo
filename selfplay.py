"""MCTS self-play for Gray Go v6 — joint MCTS + 3 aux targets.

New auxiliary targets:
- aux3: opponent action distribution (S*S+1, including pass)
- aux4: position complexity (scalar, MCTS visit entropy)
- aux5: influence map (9x9 spatial, territory ownership)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import time
import numpy as np

# Try C++ game engine first
try:
    from graygo_engine import GameState, EMPTY, BLACK, WHITE, GRAY, BLACK_PLAYER, WHITE_PLAYER
    CPP_ENGINE_AVAILABLE = True
    print('Using C++ game engine', flush=True)

    def action_count(size: int) -> int:
        return size * size + 1
except ImportError:
    from engine import (
        BLACK, WHITE, GRAY, EMPTY,
        BLACK_PLAYER, WHITE_PLAYER,
        GameState, action_count,
    )
    CPP_ENGINE_AVAILABLE = False

from model import encode_player_relative, evaluate_position

# Try C++ MCTS engine
try:
    import mcts_engine_callback as _mcts_cpp
    CPP_MCTS_AVAILABLE = True
    print('Using C++ MCTS callback engine', flush=True)
except ImportError:
    CPP_MCTS_AVAILABLE = False
    print('C++ MCTS callback engine not available; build mcts_engine_callback before Python self-play', flush=True)


@dataclass
class TrainingSample:
    """Training sample with policy + value + 5 aux targets."""
    state: np.ndarray          # (6, S, S) — player-relative encoding
    policy: np.ndarray         # (S*S+1,) — MCTS policy for current player
    value_target: float        # +1 if current player won, -1 if lost, 0 draw
    aux1_target: np.ndarray    # (S, S) — spatial: next-state stone delta
    aux2_target: float         # scalar: territory margin
    aux3_target: np.ndarray    # (S*S+1,) — opponent action distribution
    aux4_target: float         # scalar: position complexity (NEW)
    aux5_target: np.ndarray    # (S, S) — spatial: influence map (NEW)


@dataclass
class SelfPlayConfig:
    board_size: int = 9
    games_per_iteration: int = 300
    max_turns: int = 100
    rng_seed: Optional[int] = None

    # MCTS
    num_visits: int = 400
    c_puct: float = 1.5
    tau: float = 0.01
    dirichlet_alpha: float = 0.15
    dirichlet_epsilon: float = 0.30

    # Temperature
    temp_high: float = 1.0
    temp_low: float = 0.3
    temp_threshold: int = 15

    # Opening randomization
    randomize_first_n: int = 4

# ─────────────────────────────────────────────────────────────
# MCTS wrapper
# ─────────────────────────────────────────────────────────────

class _MCTSResult:
    def __init__(self, visit_counts: dict, total_visits: int):
        self.child_N = visit_counts
        self.total_visits = total_visits


def _make_eval_fn(model, board_size):
    def eval_fn(state):
        bp, wp, value = evaluate_position(model, state, board_size)
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
            float(value),
            legal_b,
            legal_w,
            game_over,
            terminal_value,
        )
    return eval_fn


def _copy_state(state):
    return state.copy()


def _step_state(state, black_action, white_action):
    state.step(black_action, white_action)


def run_mcts_with_eval_fn(state, eval_fn, num_visits=400, c_puct=1.5, tau=0.01,
                          rng=None, dirichlet_alpha=0.15, dirichlet_epsilon=0.30,
                          add_noise=True, max_candidates=20):
    """Run MCTS search using a supplied eval callback."""
    board_size = state.size
    seed = int(rng.integers(0, 2**31)) if rng is not None else 42

    if CPP_MCTS_AVAILABLE:
        result = _mcts_cpp.run_mcts_cpp(
            state, eval_fn, _copy_state, _step_state,
            num_visits=num_visits,
            c_puct=c_puct,
            tau=tau,
            dirichlet_alpha=dirichlet_alpha if add_noise else 0.0,
            dirichlet_epsilon=dirichlet_epsilon if add_noise else 0.0,
            board_size=board_size,
            max_candidates=max_candidates,
            seed=seed,
        )
        visit_counts = {}
        for key, count in result["visit_counts"].items():
            visit_counts[tuple(key)] = count
        return _MCTSResult(visit_counts, result["total_visits"])
    else:
        raise ImportError("Python MCTS fallback not implemented. Build mcts_engine_callback.")


def run_mcts(state, model, num_visits=400, c_puct=1.5, tau=0.01,
             rng=None, dirichlet_alpha=0.15, dirichlet_epsilon=0.30,
             add_noise=True, max_candidates=20):
    """Run MCTS search using color-flip inference."""
    return run_mcts_with_eval_fn(
        state,
        _make_eval_fn(model, state.size),
        num_visits=num_visits,
        c_puct=c_puct,
        tau=tau,
        rng=rng,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        add_noise=add_noise,
        max_candidates=max_candidates,
    )


def marginalize_policy(result: _MCTSResult, player: int, n_actions: int) -> np.ndarray:
    """Extract per-player policy from MCTS result."""
    target = np.zeros(n_actions, dtype=np.float32)
    total = 0
    for (a_b, a_w), n in result.child_N.items():
        if player == BLACK_PLAYER:
            target[a_b] += n
        else:
            target[a_w] += n
        total += n
    if total > 0:
        target /= total
    return target


def sample_joint_move(result: _MCTSResult, temperature: float,
                      rng: np.random.Generator) -> tuple[int, int]:
    """Sample a joint move from MCTS visit counts."""
    if not result.child_N:
        raise ValueError("No visits in MCTS result")
    actions = list(result.child_N.keys())
    counts = np.array([result.child_N[a] for a in actions], dtype=np.float64)
    if temperature < 1e-6:
        idx = int(np.argmax(counts))
    else:
        log_counts = np.log(counts + 1e-30)
        log_probs = log_counts / temperature
        log_probs -= log_probs.max()
        probs = np.exp(log_probs)
        probs /= probs.sum()
        idx = int(rng.choice(len(actions), p=probs))
    return actions[idx]


# ─────────────────────────────────────────────────────────────
# New auxiliary target computation (v6)
# ─────────────────────────────────────────────────────────────

def _get_grid(state):
    if CPP_ENGINE_AVAILABLE:
        return state.get_board_numpy()
    else:
        return state.board.grid


def compute_aux1_target(state_before, state_after, player: int, board_size: int) -> np.ndarray:
    """Spatial: next-state stone delta from player's perspective."""
    s = board_size
    target = np.zeros((s, s), dtype=np.float32)
    my_color = BLACK if player == BLACK_PLAYER else WHITE
    grid_before = _get_grid(state_before)
    grid_after = _get_grid(state_after)
    for y in range(s):
        for x in range(s):
            before = grid_before[y, x]
            after = grid_after[y, x]
            if before != after:
                if after == my_color and before != my_color:
                    target[y, x] = 1.0
                elif before == my_color and after != my_color:
                    target[y, x] = -1.0
                elif after == GRAY:
                    target[y, x] = 0.5
    return target


def compute_aux2_target(final_state, player: int) -> float:
    """Scalar: territory margin from player's perspective."""
    score_result = final_state.score()
    black_score, white_score = score_result[0], score_result[1]
    total = final_state.size * final_state.size
    if player == BLACK_PLAYER:
        return (black_score - white_score) / total
    else:
        return (white_score - black_score) / total


def compute_aux3_target(opponent_action: int, board_size: int) -> np.ndarray:
    """Categorical opponent action distribution, including pass."""
    s = board_size
    n_actions = s * s + 1
    target = np.zeros(n_actions, dtype=np.float32)
    if 0 <= opponent_action < n_actions:
        target[opponent_action] = 1.0
    return target


def compute_policy_entropy(policy: np.ndarray) -> float:
    """Normalized entropy for an action distribution."""
    nonzero = policy[policy > 1e-10]
    if len(nonzero) == 0:
        return 0.0
    entropy = -np.sum(nonzero * np.log(nonzero))
    max_entropy = np.log(policy.shape[0])
    if max_entropy > 0:
        return float(entropy / max_entropy)
    return 0.0


def compute_aux4_from_mcts(result: _MCTSResult, player: int, board_size: int) -> float:
    """Scalar: position complexity = entropy of marginalized visit distribution."""
    n_actions = board_size * board_size + 1
    policy = marginalize_policy(result, player, n_actions)
    return compute_policy_entropy(policy)


def compute_aux5_target(state, player: int, board_size: int) -> np.ndarray:
    """Spatial: influence map — territory ownership estimate via BFS distance."""
    s = board_size
    grid = _get_grid(state)
    target = np.zeros((s, s), dtype=np.float32)

    my_color = BLACK if player == BLACK_PLAYER else WHITE
    opp_color = WHITE if player == BLACK_PLAYER else BLACK

    # BFS distance from each color's stones
    from collections import deque

    dist_my = np.full((s, s), np.inf, dtype=np.float32)
    dist_opp = np.full((s, s), np.inf, dtype=np.float32)

    q_my = deque()
    q_opp = deque()

    for y in range(s):
        for x in range(s):
            if grid[y, x] == my_color:
                dist_my[y, x] = 0
                q_my.append((y, x))
            elif grid[y, x] == opp_color:
                dist_opp[y, x] = 0
                q_opp.append((y, x))

    # BFS for my stones
    while q_my:
        y, x = q_my.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = (y + dy) % s, (x + dx) % s  # torus wrap
            if grid[ny, nx] == EMPTY or grid[ny, nx] == GRAY:
                new_dist = dist_my[y, x] + 1
                if new_dist < dist_my[ny, nx]:
                    dist_my[ny, nx] = new_dist
                    q_my.append((ny, nx))

    # BFS for opponent stones
    while q_opp:
        y, x = q_opp.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = (y + dy) % s, (x + dx) % s
            if grid[ny, nx] == EMPTY or grid[ny, nx] == GRAY:
                new_dist = dist_opp[y, x] + 1
                if new_dist < dist_opp[ny, nx]:
                    dist_opp[ny, nx] = new_dist
                    q_opp.append((ny, nx))

    # Influence: +1 = my territory, -1 = opponent territory.
    finite_my = np.isfinite(dist_my)
    finite_opp = np.isfinite(dist_opp)

    both = finite_my & finite_opp
    total = dist_my[both] + dist_opp[both]
    valid = total > 1e-6
    both_indices = np.where(both)
    if both_indices[0].size:
        y_idx = both_indices[0][valid]
        x_idx = both_indices[1][valid]
        target[y_idx, x_idx] = (dist_opp[y_idx, x_idx] - dist_my[y_idx, x_idx]) / total[valid]

    target[finite_my & ~finite_opp] = 1.0
    target[finite_opp & ~finite_my] = -1.0

    # Stones: already owned
    target[grid == my_color] = 1.0
    target[grid == opp_color] = -1.0
    target[grid == GRAY] = 0.0

    return np.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Single game
# ─────────────────────────────────────────────────────────────

def play_game(model, cfg: SelfPlayConfig, rng: np.random.Generator):
    """Play one MCTS self-play game with v6 aux targets."""
    s = cfg.board_size
    n_actions = action_count(s)
    state = GameState(size=s)

    records: list[tuple] = []

    turn = 0
    while not state.game_over and turn < cfg.max_turns:
        state_before = state.copy()

        if turn < cfg.randomize_first_n:
            legal_b = state.legal_actions(BLACK_PLAYER)
            legal_w = state.legal_actions(WHITE_PLAYER)
            b_idx = np.flatnonzero(legal_b)
            w_idx = np.flatnonzero(legal_w)
            b_move = int(rng.choice(b_idx)) if len(b_idx) > 0 else n_actions - 1
            w_move = int(rng.choice(w_idx)) if len(w_idx) > 0 else n_actions - 1

            bp = np.zeros(n_actions, dtype=np.float32)
            wp = np.zeros(n_actions, dtype=np.float32)
            if len(b_idx) > 0:
                bp[b_idx] = 1.0 / len(b_idx)
            else:
                bp[-1] = 1.0
            if len(w_idx) > 0:
                wp[w_idx] = 1.0 / len(w_idx)
            else:
                wp[-1] = 1.0

            aux4_b = compute_policy_entropy(bp)
            aux4_w = compute_policy_entropy(wp)
        else:
            root = run_mcts(
                state, model,
                num_visits=cfg.num_visits,
                c_puct=cfg.c_puct,
                tau=cfg.tau,
                rng=rng,
                dirichlet_alpha=cfg.dirichlet_alpha,
                dirichlet_epsilon=cfg.dirichlet_epsilon,
                add_noise=True,
            )

            bp = marginalize_policy(root, BLACK_PLAYER, n_actions)
            wp = marginalize_policy(root, WHITE_PLAYER, n_actions)

            temp = cfg.temp_high if turn < cfg.temp_threshold else cfg.temp_low
            b_move, w_move = sample_joint_move(root, temp, rng)

            aux4_b = compute_aux4_from_mcts(root, BLACK_PLAYER, s)
            aux4_w = compute_aux4_from_mcts(root, WHITE_PLAYER, s)

        state.step(b_move, w_move)

        # Compute aux3 (opponent action distribution) and aux5 (influence map)
        # From Black's perspective, opponent is White (w_move)
        # From White's perspective, opponent is Black (b_move)
        aux3_b = compute_aux3_target(w_move, s)  # Black's view: opponent=White played w_move
        aux3_w = compute_aux3_target(b_move, s)  # White's view: opponent=Black played b_move

        aux5_b = compute_aux5_target(state_before, BLACK_PLAYER, s)
        aux5_w = compute_aux5_target(state_before, WHITE_PLAYER, s)

        records.append((state_before, bp, wp, state.copy(),
                        aux3_b, aux3_w, aux4_b, aux4_w, aux5_b, aux5_w))
        turn += 1

        if turn % 20 == 0:
            print(f"    turn {turn}", flush=True)

    winner = state.winner_player()
    if winner == BLACK_PLAYER:
        black_outcome = 1.0
    elif winner == WHITE_PLAYER:
        black_outcome = -1.0
    else:
        black_outcome = 0.0

    aux2_black = compute_aux2_target(state, BLACK_PLAYER)
    aux2_white = compute_aux2_target(state, WHITE_PLAYER)

    samples: list[TrainingSample] = []

    for rec in records:
        (state_before, bp, wp, state_after,
         aux3_b, aux3_w, aux4_b, aux4_w, aux5_b, aux5_w) = rec

        aux1_black = compute_aux1_target(state_before, state_after, BLACK_PLAYER, s)
        aux1_white = compute_aux1_target(state_before, state_after, WHITE_PLAYER, s)

        black_encoded = encode_player_relative(state_before, BLACK_PLAYER, s)
        samples.append(TrainingSample(
            state=black_encoded,
            policy=bp,
            value_target=black_outcome,
            aux1_target=aux1_black,
            aux2_target=aux2_black,
            aux3_target=aux3_b,
            aux4_target=aux4_b,
            aux5_target=aux5_b,
        ))

        white_encoded = encode_player_relative(state_before, WHITE_PLAYER, s)
        samples.append(TrainingSample(
            state=white_encoded,
            policy=wp,
            value_target=-black_outcome,
            aux1_target=aux1_white,
            aux2_target=aux2_white,
            aux3_target=aux3_w,
            aux4_target=aux4_w,
            aux5_target=aux5_w,
        ))

    return samples, state


# ─────────────────────────────────────────────────────────────
# Batch generation
# ─────────────────────────────────────────────────────────────

def generate_selfplay_data(model, cfg: SelfPlayConfig) -> list[TrainingSample]:
    """Generate raw self-play data. Training applies online augmentation."""
    rng = np.random.default_rng(cfg.rng_seed)
    all_samples: list[TrainingSample] = []
    t0 = time.time()
    wins = {BLACK_PLAYER: 0, WHITE_PLAYER: 0, None: 0}

    for game_idx in range(cfg.games_per_iteration):
        gt0 = time.time()
        pt0 = time.time()
        samples, final_state = play_game(model, cfg, rng)
        play_time = time.time() - pt0

        all_samples.extend(samples)

        winner = final_state.winner_player()
        if winner == -1:
            winner = None
        wins[winner] = wins.get(winner, 0) + 1
        dt = time.time() - gt0
        elapsed = time.time() - t0

        if (game_idx + 1) % 10 == 0 or game_idx == 0:
            print(
                f"  Game {game_idx + 1}/{cfg.games_per_iteration}: "
                f"{final_state.turn_number} turns, {dt:.1f}s (play={play_time:.1f}s), "
                f"{len(samples)} raw samples | "
                f"B/W/D: {wins[BLACK_PLAYER]}/{wins[WHITE_PLAYER]}/{wins[None]} | "
                f"total: {len(all_samples)}, {elapsed:.0f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    print(
        f"  Self-play complete: {cfg.games_per_iteration} games, "
        f"{len(all_samples)} total samples in {elapsed:.1f}s "
        f"({elapsed / cfg.games_per_iteration:.1f}s/game). "
        f"B/W/D: {wins[BLACK_PLAYER]}/{wins[WHITE_PLAYER]}/{wins[None]}",
        flush=True,
    )

    return all_samples

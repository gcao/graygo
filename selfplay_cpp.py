"""C++ self-play wrapper for v6 — drop-in replacement for generate_selfplay_data().

Uses mcts_engine (pure C++ MCTS + libtorch inference) for dramatically
higher GPU utilization compared to the Python self-play loop.

Requires:
  - A TorchScript-traced model (see export_model.py)
  - Built mcts_engine.so (see Makefile, `make cpp-selfplay`)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from selfplay import (
    SelfPlayConfig,
    TrainingSample,
    augment_game,
)

try:
    import torch  # must import before mcts_engine for CUDA lib init
    import mcts_engine
    CPP_SELFPLAY_AVAILABLE = True
except ImportError:
    CPP_SELFPLAY_AVAILABLE = False


def generate_selfplay_data_cpp(
    model_path: str | Path,
    cfg: SelfPlayConfig,
    device: str = "cuda",
) -> list[TrainingSample]:
    """Generate self-play data using the C++ engine.

    This is a drop-in replacement for selfplay.generate_selfplay_data(),
    but takes a model_path (TorchScript .pt) instead of a live model.
    """
    if not CPP_SELFPLAY_AVAILABLE:
        raise ImportError(
            "mcts_engine not available. Build with `make cpp-selfplay`."
        )

    model_path = str(model_path)
    seed = cfg.rng_seed if cfg.rng_seed is not None else 42

    t0 = time.time()
    print(f"  C++ self-play: {cfg.games_per_iteration} games, "
          f"{cfg.num_visits} visits, device={device}", flush=True)

    result = mcts_engine.run_selfplay_games(
        model_path=model_path,
        num_games=cfg.games_per_iteration,
        board_size=cfg.board_size,
        max_turns=cfg.max_turns,
        num_visits=cfg.num_visits,
        c_puct=cfg.c_puct,
        tau=cfg.tau,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_epsilon=cfg.dirichlet_epsilon,
        temp_high=cfg.temp_high,
        temp_low=cfg.temp_low,
        temp_threshold=cfg.temp_threshold,
        randomize_first_n=cfg.randomize_first_n,
        max_candidates=20,
        seed=seed,
        device=device,
    )

    raw_time = time.time() - t0
    n_raw = result["n_samples"]
    wins_b = result["wins_black"]
    wins_w = result["wins_white"]
    draws = result["draws"]

    print(f"  C++ self-play raw: {n_raw} samples in {raw_time:.1f}s "
          f"({raw_time / cfg.games_per_iteration:.1f}s/game). "
          f"B/W/D: {wins_b}/{wins_w}/{draws}", flush=True)

    if n_raw == 0:
        return []

    # Convert numpy arrays to TrainingSample list
    states = result["states"]      # (N, 6, S, S)
    policies = result["policies"]  # (N, S*S+1)
    values = result["values"]      # (N,)
    aux1 = result["aux1"]          # (N, S, S)
    aux2 = result["aux2"]          # (N,)
    aux3 = result["aux3"]          # (N, S, S)
    aux4 = result["aux4"]          # (N,)
    aux5 = result["aux5"]          # (N, S, S)

    # Group samples by game (2 samples per turn, alternating black/white)
    # Then apply augmentation per-game
    s = cfg.board_size
    rng = np.random.default_rng(seed + 99)

    # Each game produces 2*turns samples. We need to figure out game boundaries.
    # Since we don't have explicit boundaries from C++, we augment all samples
    # together in one batch (augment_game handles color-flip + sym + shift).
    samples = []
    for i in range(n_raw):
        samples.append(TrainingSample(
            state=states[i],
            policy=policies[i],
            value_target=float(values[i]),
            aux1_target=aux1[i],
            aux2_target=float(aux2[i]),
            aux3_target=aux3[i],
            aux4_target=float(aux4[i]),
            aux5_target=aux5[i],
        ))

    # Apply 8x augmentation
    t_aug = time.time()
    augmented = augment_game(
        samples, cfg.board_size, cfg.aug_symmetries, cfg.aug_shifts, rng
    )
    aug_time = time.time() - t_aug

    total_time = time.time() - t0
    print(f"  C++ self-play total: {len(augmented)} samples (8x aug) in {total_time:.1f}s "
          f"(raw={raw_time:.1f}s, aug={aug_time:.1f}s)", flush=True)

    return augmented

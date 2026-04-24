"""Focused smoke tests for Gray Go training orchestration.

Run from the repository root:
    python tests/smoke_graygo.py
"""

from __future__ import annotations

import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from export_model import export
from gate import GateConfig, evaluate_models
from model import GrayGoNet
from selfplay import SelfPlayConfig, compute_aux5_target, generate_selfplay_data

try:
    from graygo_engine import BLACK_PLAYER, GameState
except ImportError:
    from engine import BLACK_PLAYER, GameState

try:
    from selfplay_cpp import CPP_SELFPLAY_AVAILABLE, generate_selfplay_data_cpp
except ImportError:
    CPP_SELFPLAY_AVAILABLE = False


def _assert_sample_batch(samples, board_size: int) -> None:
    n_actions = board_size * board_size + 1
    assert len(samples) == 2, f"expected 2 samples from one joint move, got {len(samples)}"
    for sample in samples:
        assert sample.state.shape == (6, board_size, board_size)
        assert sample.policy.shape == (n_actions,)
        assert sample.aux1_target.shape == (board_size, board_size)
        assert sample.aux3_target.shape == (n_actions,)
        assert sample.aux5_target.shape == (board_size, board_size)
        assert np.isfinite(sample.state).all()
        assert np.isfinite(sample.policy).all()
        assert np.isfinite(sample.value_target)
        assert np.isfinite(sample.aux1_target).all()
        assert np.isfinite(sample.aux2_target)
        assert np.isfinite(sample.aux3_target).all()
        assert np.isfinite(sample.aux4_target)
        assert np.isfinite(sample.aux5_target).all()
        assert np.isclose(sample.policy.sum(), 1.0, atol=1e-5)
        assert np.isclose(sample.aux3_target.sum(), 1.0, atol=1e-5)


def test_aux5_empty_board_is_finite() -> None:
    board_size = 5
    state = GameState(size=board_size)
    aux5 = compute_aux5_target(state, BLACK_PLAYER, board_size)
    assert aux5.shape == (board_size, board_size)
    assert np.isfinite(aux5).all()


def test_python_selfplay_targets() -> None:
    board_size = 5
    model = GrayGoNet(board_size=board_size, blocks=1, filters=16).eval()
    cfg = SelfPlayConfig(
        board_size=board_size,
        games_per_iteration=1,
        max_turns=1,
        rng_seed=123,
        num_visits=2,
        randomize_first_n=1,
    )
    samples = generate_selfplay_data(model, cfg)
    _assert_sample_batch(samples, board_size)


def test_cpp_selfplay_targets() -> None:
    if not CPP_SELFPLAY_AVAILABLE:
        print("Skipping C++ self-play smoke: mcts_engine extension is not built.")
        return

    board_size = 5
    model = GrayGoNet(board_size=board_size, blocks=1, filters=16).eval()
    cfg = SelfPlayConfig(
        board_size=board_size,
        games_per_iteration=1,
        max_turns=1,
        rng_seed=123,
        num_visits=2,
        randomize_first_n=1,
    )

    with tempfile.TemporaryDirectory(prefix="graygo_cpp_smoke_") as tmp:
        traced_path = Path(tmp) / "model.pt"
        export(model, traced_path)
        samples = generate_selfplay_data_cpp(traced_path, cfg, device="cpu")
    _assert_sample_batch(samples, board_size)


def test_callback_gate_smoke() -> None:
    board_size = 5
    model = GrayGoNet(board_size=board_size, blocks=1, filters=16).eval()
    result = evaluate_models(
        model,
        copy.deepcopy(model).eval(),
        GateConfig(
            board_size=board_size,
            num_games=1,
            max_turns=1,
            rng_seed=123,
            num_visits=2,
            max_candidates=4,
        ),
    )
    assert 0.0 <= result.win_rate <= 1.0
    assert result.wins + result.losses + result.draws == 1


def _run_tiny_training(output_dir: str, replay_checkpoint_interval: int) -> None:
    cmd = [
        sys.executable,
        "run.py",
        "--iterations",
        "1",
        "--device",
        "cpu",
        "--board-size",
        "5",
        "--blocks",
        "1",
        "--filters",
        "16",
        "--games-per-iteration",
        "1",
        "--max-turns",
        "1",
        "--randomize-first-n",
        "1",
        "--mcts-visits",
        "1",
        "--batch-size",
        "2",
        "--train-batches",
        "1",
        "--gating-games",
        "1",
        "--gating-visits",
        "1",
        "--replay-buffer-size",
        "16",
        "--replay-checkpoint-interval",
        str(replay_checkpoint_interval),
        "--no-slack",
        "--output-dir",
        output_dir,
    ]
    subprocess.run(cmd, check=True)


def test_fresh_iteration_zero_run_without_replay_checkpoint() -> None:
    with tempfile.TemporaryDirectory(prefix="graygo_run_smoke_") as tmp:
        _run_tiny_training(tmp, replay_checkpoint_interval=0)
        out_dir = Path(tmp)
        assert (out_dir / "training_state.pt").exists()
        assert not (out_dir / "replay_buffer.pt").exists()


def test_fresh_iteration_zero_run_with_replay_checkpoint() -> None:
    with tempfile.TemporaryDirectory(prefix="graygo_run_replay_smoke_") as tmp:
        _run_tiny_training(tmp, replay_checkpoint_interval=1)
        out_dir = Path(tmp)
        assert (out_dir / "training_state.pt").exists()
        assert (out_dir / "replay_buffer.pt").exists()
        state = torch.load(out_dir / "training_state.pt", map_location="cpu", weights_only=False)
        assert state["replay_buffer_path"] == "replay_buffer.pt"
        assert state["replay_buffer_saved_this_iteration"] is True


def main() -> int:
    torch.set_num_threads(1)
    tests = [
        test_aux5_empty_board_is_finite,
        test_python_selfplay_targets,
        test_cpp_selfplay_targets,
        test_callback_gate_smoke,
        test_fresh_iteration_zero_run_without_replay_checkpoint,
        test_fresh_iteration_zero_run_with_replay_checkpoint,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

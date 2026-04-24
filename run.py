"""Training entry point for Gray Go v6.

Key changes from v5:
- 5 auxiliary heads (3 new: opponent move dist, position complexity, influence map)
- Soft gating: 48% threshold (safety check only)
- Buffer flush on strong promotion (WR > 55%)
- Buffer size: 2M (up from 1M)
- Same model architecture (12b128f), MCTS now at 600 visits

Usage:
    python run.py --iterations 0 --device cuda
    python run.py --resume --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import signal
import time
from pathlib import Path
from typing import Optional
import os
import subprocess

_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\nShutdown requested (signal {signum}). Will stop after current iteration.", flush=True)


def _seed_for(base: Optional[int], iteration: int, offset: int) -> Optional[int]:
    return None if base is None else base + iteration * 100 + offset

def _slack_notify(message: str, channel: str = "C0AC5PWNW2W", thread_ts: str = "1774994131.003049") -> None:
    """Post a status message to Slack thread."""
    token = os.environ.get("SLACK_AGENTX_TOKEN", "")
    if not token:
        return
    payload = {"channel": channel, "text": message}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    try:
        subprocess.run(
            ["curl", "-s", "-X", "POST", "https://slack.com/api/chat.postMessage",
             "-H", f"Authorization: Bearer {token}",
             "-H", "Content-Type: application/json; charset=utf-8",
             "-d", json.dumps(payload)],
            timeout=10, capture_output=True,
        )
    except Exception:
        pass  # Never let Slack notifications crash training



def parse_args():
    p = argparse.ArgumentParser(description="Gray Go v6 training")
    p.add_argument("--iterations", type=int, default=0, help="0 = run forever")
    p.add_argument("--board-size", type=int, default=9)
    p.add_argument("--blocks", type=int, default=12)
    p.add_argument("--filters", type=int, default=128)

    # Self-play
    p.add_argument("--games-per-iteration", type=int, default=300)
    p.add_argument("--max-turns", type=int, default=100)
    p.add_argument("--randomize-first-n", type=int, default=4)

    # MCTS
    p.add_argument("--mcts-visits", type=int, default=600)
    p.add_argument("--mcts-cpuct", type=float, default=1.5)
    p.add_argument("--mcts-tau", type=float, default=0.01)
    p.add_argument("--dirichlet-alpha", type=float, default=0.15)
    p.add_argument("--dirichlet-epsilon", type=float, default=0.30)

    # Temperature
    p.add_argument("--temp-high", type=float, default=1.0)
    p.add_argument("--temp-low", type=float, default=0.3)
    p.add_argument("--temp-threshold", type=int, default=15)

    # Augmentation
    p.add_argument("--aug-symmetries", type=int, default=2)
    p.add_argument("--aug-shifts", type=int, default=2)

    # Training
    p.add_argument("--replay-buffer-size", type=int, default=2_000_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--train-batches", type=int, default=0, help="0 = epoch-based")
    p.add_argument("--train-epochs", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--lr-decay", type=float, default=0.0,
                   help="CosineAnnealing T_max (iterations). 0=disabled")
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--lambda-aux1", type=float, default=0.5)
    p.add_argument("--lambda-aux2", type=float, default=0.25)
    p.add_argument("--lambda-aux3", type=float, default=0.3)
    p.add_argument("--lambda-aux4", type=float, default=0.15)
    p.add_argument("--lambda-aux5", type=float, default=0.2)

    # Gating
    p.add_argument("--gating-games", type=int, default=100)
    p.add_argument("--gating-threshold", type=float, default=0.48)
    p.add_argument("--buffer-flush-wr", type=float, default=0.55,
                   help="Flush buffer if gate win rate exceeds this")

    # Infrastructure
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=str, default="checkpoints")
    p.add_argument("--cooldown", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-from-v5", type=str, default=None,
                   help="Path to v5 checkpoint to warm-start from")
    p.add_argument("--slack-channel", type=str, default="C0AC5PWNW2W",
                   help="Slack channel ID for status updates")
    p.add_argument("--no-slack", action="store_true",
                   help="Disable Slack notifications")
    p.add_argument("--use-cpp-selfplay", action="store_true",
                   help="Use C++ self-play engine (requires traced model + mcts_engine)")
    p.add_argument("--traced-model", type=str, default=None,
                   help="Path to TorchScript traced model (default: <output-dir>/champion_traced.pt)")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch required. Install torch first.", flush=True)
        return 1

    from model import GrayGoNet
    from selfplay import SelfPlayConfig, generate_selfplay_data
    from train import TrainConfig, ReplayBuffer, train_model
    from gate import GateConfig, evaluate_models

    if args.use_cpp_selfplay:
        from selfplay_cpp import generate_selfplay_data_cpp, CPP_SELFPLAY_AVAILABLE
        if not CPP_SELFPLAY_AVAILABLE:
            print("--use-cpp-selfplay requires mcts_engine. Build with `make cpp-selfplay`.", flush=True)
            return 1
        from export_model import export as export_traced_model
    

    # Monkey-patch gate_v4 to use v6 model inference
    
    

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.", flush=True)
        device = torch.device("cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"

    run_forever = args.iterations == 0
    max_label = "inf" if run_forever else str(args.iterations)

    # Resume logic
    resume_iter = 0
    ckpt = None

    if args.resume_from_v5:
        v5_path = Path(args.resume_from_v5)
        if not v5_path.exists():
            print(f"--resume-from-v5: {v5_path} not found", flush=True)
            return 1
        # Load v5 checkpoint, extract what we can (backbone + policy/value + aux1/aux2)
        ckpt = torch.load(v5_path, map_location=device, weights_only=False)
        sd = ckpt["state_dict"]
        filters = sd["stem_conv.conv.weight"].shape[0]
        block_indices = {int(k.split(".")[1]) for k in sd if k.startswith("res_blocks.")}
        blocks = len(block_indices)
        board_size = int(round((sd["policy_fc.weight"].shape[0] - 1) ** 0.5))
        args.blocks = blocks
        args.filters = filters
        args.board_size = board_size
        print(f"Warm-starting from v5 checkpoint: {v5_path}", flush=True)
    elif args.resume:
        champion_files = sorted(output_dir.glob("champion_iter_*.pt"))
        if not champion_files:
            print(f"--resume but no checkpoints in {output_dir}", flush=True)
            return 1
        last = champion_files[-1]
        ckpt = torch.load(last, map_location=device, weights_only=False)
        sd = ckpt["state_dict"]
        filters = sd["stem_conv.conv.weight"].shape[0]
        block_indices = {int(k.split(".")[1]) for k in sd if k.startswith("res_blocks.")}
        blocks = len(block_indices)
        board_size = int(round((sd["policy_fc.weight"].shape[0] - 1) ** 0.5))
        args.blocks = blocks
        args.filters = filters
        args.board_size = board_size
        resume_iter = ckpt["metadata"]["iteration"] + 1
        print(f"Resuming from {last} (iter {resume_iter})", flush=True)

    champion = GrayGoNet(
        board_size=args.board_size,
        blocks=args.blocks,
        filters=args.filters,
    ).to(device)

    if ckpt is not None:
        sd = ckpt["state_dict"]
        # Load matching keys, ignore new aux head keys
        champion_sd = champion.state_dict()
        loaded_keys = []
        skipped_keys = []
        for k, v in sd.items():
            if k in champion_sd and champion_sd[k].shape == v.shape:
                champion_sd[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append(k)
        champion.load_state_dict(champion_sd)
        print(f"Loaded {len(loaded_keys)}/{len(champion_sd)} params, "
              f"skipped {len(skipped_keys)} (new/shape-mismatch)", flush=True)

    params = sum(p.numel() for p in champion.parameters())
    print(f"Model: {args.blocks}b{args.filters}f, board={args.board_size}, "
          f"params={params:,}", flush=True)
    print(f"Architecture: single policy, 5 aux heads (v6)", flush=True)
    print(f"MCTS: visits={args.mcts_visits}, c_puct={args.mcts_cpuct}", flush=True)
    print(f"Augmentation: 8x (color x {args.aug_symmetries} sym x {args.aug_shifts} shifts)", flush=True)
    print(f"Learning rate: {args.learning_rate}", flush=True)
    print(f"Gating: threshold={args.gating_threshold}, flush at WR>{args.buffer_flush_wr}", flush=True)
    print(f"Buffer: {args.replay_buffer_size:,} samples", flush=True)

    candidate = copy.deepcopy(champion).to(device)
    replay_buffer = ReplayBuffer(maxlen=args.replay_buffer_size)

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        num_batches=args.train_batches,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        l2_regularization=args.l2,
        replay_buffer_size=args.replay_buffer_size,
        lambda_aux1=args.lambda_aux1,
        lambda_aux2=args.lambda_aux2,
        lambda_aux3=args.lambda_aux3,
        lambda_aux4=args.lambda_aux4,
        lambda_aux5=args.lambda_aux5,
    )

    iteration = resume_iter
    while True:
        if _shutdown_requested:
            print(f"\nStopping at iteration {iteration}.", flush=True)
            break
        if not run_forever and iteration >= args.iterations:
            break

        print(f"\n{'='*60}", flush=True)
        print(f"=== Iteration {iteration + 1}/{max_label} ===", flush=True)
        print(f"{'='*60}", flush=True)
        if not args.no_slack:
            _slack_notify(f"♟️ *GrayGo v6* — Iteration {iteration + 1} starting\nModel: {args.blocks}b{args.filters}f | Buffer: {len(replay_buffer):,}")

        # ── Self-play ──
        selfplay_cfg = SelfPlayConfig(
            board_size=args.board_size,
            games_per_iteration=args.games_per_iteration,
            max_turns=args.max_turns,
            rng_seed=_seed_for(args.seed, iteration, 1),
            num_visits=args.mcts_visits,
            c_puct=args.mcts_cpuct,
            tau=args.mcts_tau,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_epsilon=args.dirichlet_epsilon,
            temp_high=args.temp_high,
            temp_low=args.temp_low,
            temp_threshold=args.temp_threshold,
            randomize_first_n=args.randomize_first_n,
            aug_symmetries=args.aug_symmetries,
            aug_shifts=args.aug_shifts,
        )

        t0 = time.time()
        if args.use_cpp_selfplay:
            traced_path = Path(args.traced_model) if args.traced_model else output_dir / "champion_traced.pt"
            # Re-export traced model each iteration (champion may have been promoted)
            export_traced_model(champion, traced_path)
            samples = generate_selfplay_data_cpp(
                traced_path, selfplay_cfg, device=args.device
            )
        else:
            samples = generate_selfplay_data(champion, selfplay_cfg)
        selfplay_time = time.time() - t0
        replay_buffer.extend(samples)
        print(f"Self-play: {len(samples)} samples in {selfplay_time:.1f}s, "
              f"buffer: {len(replay_buffer)}", flush=True)
        if not args.no_slack:
            _slack_notify(f"♟️ *GrayGo v6* Iter {iteration + 1} self-play done\n{len(samples):,} samples in {selfplay_time/3600:.1f}h | Buffer: {len(replay_buffer):,}")

        # ── Train ──
        candidate.load_state_dict(champion.state_dict())
        train_cfg.rng_seed = _seed_for(args.seed, iteration, 11)
        t0 = time.time()
        metrics = train_model(candidate, replay_buffer, train_cfg)
        train_time = time.time() - t0
        print(
            f"Train ({train_time:.1f}s): loss={metrics['loss']:.4f}, "
            f"p={metrics['policy_loss']:.4f}, "
            f"v={metrics['value_loss']:.4f}, "
            f"a1={metrics['aux1_loss']:.4f}, "
            f"a2={metrics['aux2_loss']:.4f}, "
            f"a3={metrics['aux3_loss']:.4f}, "
            f"a4={metrics['aux4_loss']:.4f}, "
            f"a5={metrics['aux5_loss']:.4f}",
            flush=True,
        )

        # ── Gate ──
        promoted = False
        gate_summary = None
        if iteration == 0 and not args.resume and not args.resume_from_v5:
            champion.load_state_dict(candidate.state_dict())
            promoted = True
            print("Promoted by default (first iteration).", flush=True)
            if not args.no_slack:
                _slack_notify(f"♟️ *GrayGo v6* Iter 1 promoted (first iteration)\nTrain loss: {metrics['loss']:.4f}")
        else:
            gate_cfg = GateConfig(
                board_size=args.board_size,
                num_games=args.gating_games,
                max_turns=args.max_turns,
                rng_seed=_seed_for(args.seed, iteration, 21),
            )
            t0 = time.time()
            result = evaluate_models(candidate, champion, gate_cfg)
            gate_time = time.time() - t0
            gate_summary = {
                "win_rate": result.win_rate,
                "wins": result.wins,
                "losses": result.losses,
                "draws": result.draws,
            }
            promoted = result.win_rate > args.gating_threshold
            print(
                f"Gate ({gate_time:.1f}s): win_rate={result.win_rate:.3f} "
                f"(W/L/D: {result.wins}/{result.losses}/{result.draws})",
                flush=True,
            )
            if not args.no_slack:
                emoji = "✅" if promoted else "❌"
                _slack_notify(f"♟️ *GrayGo v6* Iter {iteration + 1} gate result {emoji}\nWR: {result.win_rate:.1%} (W{result.wins}/L{result.losses}/D{result.draws}) | Train loss: {metrics['loss']:.4f}")
            if promoted:
                champion.load_state_dict(candidate.state_dict())
                print("Candidate PROMOTED.", flush=True)
                # Buffer flush on strong promotion
                if result.win_rate > args.buffer_flush_wr and len(replay_buffer) > 0:
                    replay_buffer.clear()
                    print(f"Buffer FLUSHED (WR {result.win_rate:.3f} > {args.buffer_flush_wr}).", flush=True)
            else:
                candidate.load_state_dict(champion.state_dict())
                print("Candidate rejected.", flush=True)

        # ── Save ──
        metadata = {
            "iteration": iteration,
            "promoted": promoted,
            "train_metrics": metrics,
            "gate": gate_summary,
            "buffer_size": len(replay_buffer),
            "config": {
                "version": "v6",
                "blocks": args.blocks,
                "filters": args.filters,
                "board_size": args.board_size,
                "mcts_visits": args.mcts_visits,
                "games_per_iteration": args.games_per_iteration,
                "max_turns": args.max_turns,
                "aug_symmetries": args.aug_symmetries,
                "aug_shifts": args.aug_shifts,
                "aug_total": 8,
                "dirichlet_alpha": args.dirichlet_alpha,
                "dirichlet_epsilon": args.dirichlet_epsilon,
                "randomize_first_n": args.randomize_first_n,
                "lambda_aux1": args.lambda_aux1,
                "lambda_aux2": args.lambda_aux2,
                "lambda_aux3": args.lambda_aux3,
                "lambda_aux4": args.lambda_aux4,
                "lambda_aux5": args.lambda_aux5,
                "learning_rate": args.learning_rate,
                "gating_threshold": args.gating_threshold,
                "buffer_flush_wr": args.buffer_flush_wr,
                "replay_buffer_size": args.replay_buffer_size,
            },
        }
        torch.save(
            {"state_dict": champion.state_dict(), "metadata": metadata},
            output_dir / f"champion_iter_{iteration:04d}.pt",
        )
        torch.save(
            {"state_dict": candidate.state_dict(), "metadata": metadata},
            output_dir / f"candidate_iter_{iteration:04d}.pt",
        )
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")

        iteration += 1

        if args.cooldown > 0 and not _shutdown_requested:
            if run_forever or iteration < args.iterations:
                print(f"Cooldown {args.cooldown}s...", flush=True)
                for _ in range(args.cooldown):
                    if _shutdown_requested:
                        break
                    time.sleep(1)

    print(f"\nDone after {iteration} iterations. Output: {output_dir}", flush=True)
    if not args.no_slack:
        _slack_notify(f"♟️ *GrayGo v6* training stopped after {iteration} iterations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

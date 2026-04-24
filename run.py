"""Training entry point for Gray Go v6.

Key changes from v5:
- 5 auxiliary heads (3 new: opponent move dist, position complexity, influence map)
- Reduced-visit joint-MCTS gating: 48% threshold
- Buffer flush on strong promotion (WR > 55%)
- Buffer size: 2M (up from 1M)
- 12b128f default model, self-play MCTS at 600 visits

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


def _load_matching_state(model, state_dict: dict) -> tuple[int, int]:
    model_state = model.state_dict()
    loaded = 0
    skipped = 0
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_state)
    return loaded, skipped


def _infer_model_shape(state_dict: dict, metadata: Optional[dict] = None) -> tuple[int, int, int]:
    filters = state_dict["stem_conv.conv.weight"].shape[0]
    block_indices = {int(k.split(".")[1]) for k in state_dict if k.startswith("res_blocks.")}
    blocks = len(block_indices)

    board_size = None
    config = (metadata or {}).get("config", {})
    if "board_size" in config:
        board_size = int(config["board_size"])
    elif "board_size" in (metadata or {}):
        board_size = int(metadata["board_size"])
    elif "policy_fc.weight" in state_dict:
        board_size = int(round((state_dict["policy_fc.weight"].shape[0] - 1) ** 0.5))
    if board_size is None:
        board_size = 9

    return blocks, filters, board_size


def _optimizer_to(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if hasattr(value, "to"):
                state[key] = value.to(device)


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

    # Training
    p.add_argument("--replay-buffer-size", type=int, default=2_000_000)
    p.add_argument("--replay-checkpoint-interval", type=int, default=1,
                   help="Save replay_buffer.pt every N completed iterations. 0 disables replay serialization")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--train-batches", type=int, default=0, help="0 = epoch-based")
    p.add_argument("--train-epochs", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--lr-decay", type=float, default=0.0,
                   help="CosineAnnealing T_max in optimizer batches. 0=disabled")
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=5.0,
                   help="Gradient clipping max norm. Use <=0 to disable clipping")
    p.add_argument("--no-online-augmentation", action="store_true",
                   help="Disable random D4 + torus shift augmentation during training")
    p.add_argument("--lambda-aux1", type=float, default=0.5)
    p.add_argument("--lambda-aux2", type=float, default=0.25)
    p.add_argument("--lambda-aux3", type=float, default=0.3)
    p.add_argument("--lambda-aux4", type=float, default=0.15)
    p.add_argument("--lambda-aux5", type=float, default=0.2)

    # Gating
    p.add_argument("--gating-games", type=int, default=100)
    p.add_argument("--gating-visits", type=int, default=64,
                   help="MCTS visits per move during promotion gate")
    p.add_argument("--gating-temperature", type=float, default=0.01,
                   help="Move sampling temperature for promotion gate")
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
    from train import (
        TrainConfig,
        ReplayBuffer,
        create_optimizer,
        create_scheduler,
        train_model,
    )
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

    if args.replay_checkpoint_interval < 0:
        print("--replay-checkpoint-interval must be >= 0", flush=True)
        return 1

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.", flush=True)
        device = torch.device("cpu")
    runtime_device = device.type

    def notify(message: str) -> None:
        if not args.no_slack:
            _slack_notify(message, channel=args.slack_channel)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    training_state_path = output_dir / "training_state.pt"
    replay_state_path = output_dir / "replay_buffer.pt"

    run_forever = args.iterations == 0
    max_label = "inf" if run_forever else str(args.iterations)

    # Resume logic
    resume_iter = 0
    ckpt = None
    training_state = None

    if args.resume_from_v5:
        v5_path = Path(args.resume_from_v5)
        if not v5_path.exists():
            print(f"--resume-from-v5: {v5_path} not found", flush=True)
            return 1
        # Load v5 checkpoint, extract what we can (backbone + policy/value + aux1/aux2)
        ckpt = torch.load(v5_path, map_location=device, weights_only=False)
        sd = ckpt["state_dict"]
        blocks, filters, board_size = _infer_model_shape(sd, ckpt.get("metadata", {}))
        args.blocks = blocks
        args.filters = filters
        args.board_size = board_size
        print(f"Warm-starting from v5 checkpoint: {v5_path}", flush=True)
    elif args.resume and training_state_path.exists():
        training_state = torch.load(training_state_path, map_location=device, weights_only=False)
        sd = training_state["champion_state_dict"]
        metadata = training_state.get("metadata", {})
        blocks, filters, board_size = _infer_model_shape(sd, metadata)
        args.blocks = blocks
        args.filters = filters
        args.board_size = board_size
        resume_iter = int(training_state.get("next_iteration", metadata.get("iteration", -1) + 1))
        ckpt = {"state_dict": sd, "metadata": metadata}
        print(f"Resuming full training state from {training_state_path} (iter {resume_iter})", flush=True)
    elif args.resume:
        champion_files = sorted(output_dir.glob("champion_iter_*.pt"))
        if not champion_files:
            print(f"--resume but no checkpoints in {output_dir}", flush=True)
            return 1
        last = champion_files[-1]
        ckpt = torch.load(last, map_location=device, weights_only=False)
        sd = ckpt["state_dict"]
        blocks, filters, board_size = _infer_model_shape(sd, ckpt.get("metadata", {}))
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
        loaded, skipped = _load_matching_state(champion, ckpt["state_dict"])
        print(f"Loaded {loaded}/{len(champion.state_dict())} params, "
              f"skipped {skipped} (new/shape-mismatch)", flush=True)

    params = sum(p.numel() for p in champion.parameters())
    print(f"Model: {args.blocks}b{args.filters}f, board={args.board_size}, "
          f"params={params:,}", flush=True)
    print(f"Architecture: single policy, 5 aux heads (v6)", flush=True)
    print(f"MCTS: visits={args.mcts_visits}, c_puct={args.mcts_cpuct}", flush=True)
    online_augmentation = not args.no_online_augmentation
    print(f"Online augmentation: {'enabled' if online_augmentation else 'disabled'} "
          f"(random D4 + torus shift per sampled position)", flush=True)
    print(f"Learning rate: {args.learning_rate}", flush=True)
    print(f"Gating: reduced-visit joint MCTS, visits={args.gating_visits}, "
          f"threshold={args.gating_threshold}, flush at WR>{args.buffer_flush_wr}", flush=True)
    print(f"Buffer: {args.replay_buffer_size:,} samples", flush=True)
    if args.replay_checkpoint_interval == 0:
        print("Replay checkpointing: disabled", flush=True)
    else:
        print(f"Replay checkpointing: every {args.replay_checkpoint_interval} iteration(s)", flush=True)

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
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else float("inf"),
        online_augmentation=online_augmentation,
        lambda_aux1=args.lambda_aux1,
        lambda_aux2=args.lambda_aux2,
        lambda_aux3=args.lambda_aux3,
        lambda_aux4=args.lambda_aux4,
        lambda_aux5=args.lambda_aux5,
    )
    optimizer = create_optimizer(candidate, train_cfg)
    scheduler = create_scheduler(optimizer, train_cfg)
    loaded_replay_buffer = False
    replay_checkpoint_available = False

    if training_state is not None:
        if "candidate_state_dict" in training_state:
            loaded, skipped = _load_matching_state(candidate, training_state["candidate_state_dict"])
            print(f"Loaded candidate state: {loaded}/{len(candidate.state_dict())} params, "
                  f"skipped {skipped}", flush=True)
        if "replay_buffer" in training_state:
            replay_buffer.load_state_dict(training_state["replay_buffer"])
            loaded_replay_buffer = True
            print(f"Loaded replay buffer: {len(replay_buffer):,} samples", flush=True)
        else:
            replay_path = training_state.get("replay_buffer_path")
            if replay_path:
                replay_file = output_dir / replay_path
                if replay_file.exists():
                    replay_buffer.load_state_dict(
                        torch.load(replay_file, map_location="cpu", weights_only=False)
                    )
                    loaded_replay_buffer = True
                    replay_checkpoint_available = True
                    print(f"Loaded replay buffer: {len(replay_buffer):,} samples from {replay_file}", flush=True)
                else:
                    print(f"Replay buffer checkpoint not found: {replay_file}", flush=True)
        if "optimizer_state_dict" in training_state:
            try:
                optimizer.load_state_dict(training_state["optimizer_state_dict"])
                _optimizer_to(optimizer, device)
                print("Loaded optimizer state.", flush=True)
            except Exception as exc:
                print(f"Could not load optimizer state; starting fresh: {exc}", flush=True)
                optimizer = create_optimizer(candidate, train_cfg)
        if scheduler is not None and training_state.get("scheduler_state_dict") is not None:
            try:
                scheduler.load_state_dict(training_state["scheduler_state_dict"])
                print("Loaded scheduler state.", flush=True)
            except Exception as exc:
                print(f"Could not load scheduler state; starting fresh: {exc}", flush=True)
                scheduler = create_scheduler(optimizer, train_cfg)

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
        notify(f"♟️ *GrayGo v6* — Iteration {iteration + 1} starting\nModel: {args.blocks}b{args.filters}f | Buffer: {len(replay_buffer):,}")

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
        )

        t0 = time.time()
        if args.use_cpp_selfplay:
            traced_path = Path(args.traced_model) if args.traced_model else output_dir / "champion_traced.pt"
            # Re-export traced model each iteration (champion may have been promoted)
            export_traced_model(champion, traced_path)
            samples = generate_selfplay_data_cpp(
                traced_path, selfplay_cfg, device=runtime_device
            )
        else:
            samples = generate_selfplay_data(champion, selfplay_cfg)
        selfplay_time = time.time() - t0
        replay_buffer.extend(samples)
        print(f"Self-play: {len(samples)} samples in {selfplay_time:.1f}s, "
              f"buffer: {len(replay_buffer)}", flush=True)
        notify(f"♟️ *GrayGo v6* Iter {iteration + 1} self-play done\n{len(samples):,} samples in {selfplay_time/3600:.1f}h | Buffer: {len(replay_buffer):,}")

        # ── Train ──
        candidate.load_state_dict(champion.state_dict())
        train_cfg.rng_seed = _seed_for(args.seed, iteration, 11)
        t0 = time.time()
        metrics = train_model(candidate, replay_buffer, train_cfg, optimizer=optimizer, scheduler=scheduler)
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
        buffer_flushed = False
        if iteration == 0 and not args.resume and not args.resume_from_v5:
            champion.load_state_dict(candidate.state_dict())
            promoted = True
            print("Promoted by default (first iteration).", flush=True)
            notify(f"♟️ *GrayGo v6* Iter 1 promoted (first iteration)\nTrain loss: {metrics['loss']:.4f}")
        else:
            gate_cfg = GateConfig(
                board_size=args.board_size,
                num_games=args.gating_games,
                max_turns=args.max_turns,
                rng_seed=_seed_for(args.seed, iteration, 21),
                num_visits=args.gating_visits,
                c_puct=args.mcts_cpuct,
                tau=args.mcts_tau,
                move_temperature=args.gating_temperature,
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
            emoji = "✅" if promoted else "❌"
            notify(f"♟️ *GrayGo v6* Iter {iteration + 1} MCTS gate result {emoji}\nWR: {result.win_rate:.1%} (W{result.wins}/L{result.losses}/D{result.draws}) | Train loss: {metrics['loss']:.4f}")
            if promoted:
                champion.load_state_dict(candidate.state_dict())
                print("Candidate PROMOTED.", flush=True)
                # Buffer flush on strong promotion
                if result.win_rate > args.buffer_flush_wr and len(replay_buffer) > 0:
                    replay_buffer.clear()
                    buffer_flushed = True
                    print(f"Buffer FLUSHED (WR {result.win_rate:.3f} > {args.buffer_flush_wr}).", flush=True)
            else:
                candidate.load_state_dict(champion.state_dict())
                optimizer = create_optimizer(candidate, train_cfg)
                scheduler = create_scheduler(optimizer, train_cfg)
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
                "online_augmentation": online_augmentation,
                "dirichlet_alpha": args.dirichlet_alpha,
                "dirichlet_epsilon": args.dirichlet_epsilon,
                "randomize_first_n": args.randomize_first_n,
                "lambda_aux1": args.lambda_aux1,
                "lambda_aux2": args.lambda_aux2,
                "lambda_aux3": args.lambda_aux3,
                "lambda_aux4": args.lambda_aux4,
                "lambda_aux5": args.lambda_aux5,
                "learning_rate": args.learning_rate,
                "grad_clip_norm": train_cfg.grad_clip_norm,
                "gating_threshold": args.gating_threshold,
                "gating_visits": args.gating_visits,
                "gating_temperature": args.gating_temperature,
                "buffer_flush_wr": args.buffer_flush_wr,
                "replay_buffer_size": args.replay_buffer_size,
                "replay_checkpoint_interval": args.replay_checkpoint_interval,
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
        save_replay = (
            args.replay_checkpoint_interval > 0
            and (
                (iteration + 1) % args.replay_checkpoint_interval == 0
                or buffer_flushed
                or (loaded_replay_buffer and not replay_checkpoint_available)
            )
        )
        if save_replay:
            torch.save(replay_buffer.state_dict(), replay_state_path)
            replay_checkpoint_available = True
        elif buffer_flushed:
            replay_checkpoint_available = False

        replay_buffer_path = (
            replay_state_path.name
            if replay_checkpoint_available
            else None
        )
        torch.save(
            {
                "champion_state_dict": champion.state_dict(),
                "candidate_state_dict": candidate.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "metadata": metadata,
                "next_iteration": iteration + 1,
                "replay_buffer_path": replay_buffer_path,
                "replay_buffer_size": len(replay_buffer),
                "replay_buffer_saved_this_iteration": save_replay,
                "replay_checkpoint_interval": args.replay_checkpoint_interval,
            },
            training_state_path,
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
    notify(f"♟️ *GrayGo v6* training stopped after {iteration} iterations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

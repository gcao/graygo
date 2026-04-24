"""Training utilities for Gray Go v6 — 5 auxiliary heads.

Loss = L_policy + L_value 
       + 0.5 * L_aux1   (stone delta, spatial)
       + 0.25 * L_aux2   (territory control, scalar)
       + 0.3 * L_aux3    (opponent action distribution)
       + 0.15 * L_aux4   (position complexity, scalar)
       + 0.2 * L_aux5    (influence map, spatial)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from selfplay import TrainingSample


def require_torch():
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch required for training.")


@dataclass
class TrainConfig:
    batch_size: int = 512
    num_batches: int = 0
    train_epochs: float = 2.0
    learning_rate: float = 5e-4
    lr_decay: float = 0.0
    l2_regularization: float = 1e-4
    replay_buffer_size: int = 2_000_000
    rng_seed: Optional[int] = None

    # Auxiliary loss weights
    lambda_aux1: float = 0.5
    lambda_aux2: float = 0.25
    lambda_aux3: float = 0.3
    lambda_aux4: float = 0.15
    lambda_aux5: float = 0.2


class ReplayBuffer:
    def __init__(self, maxlen: int = 2_000_000) -> None:
        self._data: deque[TrainingSample] = deque(maxlen=maxlen)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def add(self, sample: TrainingSample) -> None:
        self._data.append(sample)

    def extend(self, samples: list[TrainingSample]) -> None:
        self._data.extend(samples)

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> list[TrainingSample]:
        data_len = len(self._data)
        if data_len == 0:
            raise ValueError("Replay buffer empty.")
        if data_len >= batch_size:
            idx = rng.choice(data_len, size=batch_size, replace=False)
        else:
            idx = rng.choice(data_len, size=batch_size, replace=True)
        return [self._data[int(i)] for i in idx]


# ─────────────────────────────────────────────────────────────
# Online augmentation
# ─────────────────────────────────────────────────────────────

def _apply_d4(array: np.ndarray, op: int) -> np.ndarray:
    if op < 4:
        return np.rot90(array, op, axes=(-2, -1))
    flipped = np.flip(array, axis=-1)
    return np.rot90(flipped, op - 4, axes=(-2, -1))


def apply_random_symmetry(
    state: np.ndarray,
    policy: np.ndarray,
    aux1: np.ndarray,
    aux3: np.ndarray,
    aux5: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply random torus symmetry to state, policy, and spatial aux targets."""
    s = state.shape[-1]
    op = int(rng.integers(0, 8))
    dx = int(rng.integers(0, s))
    dy = int(rng.integers(0, s))

    ts = _apply_d4(state, op)
    ts = np.roll(ts, shift=(dy, dx), axis=(-2, -1))

    p_board = policy[:-1].reshape(s, s)
    p_board = _apply_d4(p_board, op)
    p_board = np.roll(p_board, shift=(dy, dx), axis=(-2, -1))
    tp = np.concatenate([p_board.reshape(-1), policy[-1:]])

    ta1 = _apply_d4(aux1, op)
    ta1 = np.roll(ta1, shift=(dy, dx), axis=(-2, -1))

    a3_board = aux3[:-1].reshape(s, s)
    a3_board = _apply_d4(a3_board, op)
    a3_board = np.roll(a3_board, shift=(dy, dx), axis=(-2, -1))
    ta3 = np.concatenate([a3_board.reshape(-1), aux3[-1:]])

    ta5 = _apply_d4(aux5, op)
    ta5 = np.roll(ta5, shift=(dy, dx), axis=(-2, -1))

    return (
        ts.astype(np.float32),
        tp.astype(np.float32),
        ta1.astype(np.float32),
        ta3.astype(np.float32),
        ta5.astype(np.float32),
    )


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train_model(model, replay_buffer: ReplayBuffer, cfg: TrainConfig) -> dict[str, float]:
    """Train model with policy + value + 5 aux losses."""
    require_torch()
    import torch

    if len(replay_buffer) == 0:
        raise ValueError("Cannot train with empty buffer.")

    rng = np.random.default_rng(cfg.rng_seed)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.l2_regularization,
    )

    # LR scheduler (CosineAnnealing if lr_decay > 0, interpreted as T_max in batches)
    scheduler = None
    if cfg.lr_decay > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg.lr_decay), eta_min=cfg.learning_rate * 0.1
        )
    model.train()
    device = next(model.parameters()).device

    if cfg.num_batches > 0:
        actual_batches = cfg.num_batches
    else:
        actual_batches = max(1, int(len(replay_buffer) * cfg.train_epochs / cfg.batch_size))

    total_loss = 0.0
    total_p_loss = 0.0
    total_v_loss = 0.0
    total_a1_loss = 0.0
    total_a2_loss = 0.0
    total_a3_loss = 0.0
    total_a4_loss = 0.0
    total_a5_loss = 0.0
    batch_losses = []
    batch_grad_norms = []

    for _ in range(actual_batches):
        batch = replay_buffer.sample_batch(cfg.batch_size, rng)

        states = np.stack([s.state for s in batch]).astype(np.float32)
        p_targets = np.stack([s.policy for s in batch]).astype(np.float32)
        v_targets = np.array([s.value_target for s in batch], dtype=np.float32)
        a1_targets = np.stack([s.aux1_target for s in batch]).astype(np.float32)
        a2_targets = np.array([s.aux2_target for s in batch], dtype=np.float32)
        a3_targets = np.stack([s.aux3_target for s in batch]).astype(np.float32)
        a4_targets = np.array([s.aux4_target for s in batch], dtype=np.float32)
        a5_targets = np.stack([s.aux5_target for s in batch]).astype(np.float32)

        for i in range(states.shape[0]):
            states[i], p_targets[i], a1_targets[i], a3_targets[i], a5_targets[i] = \
                apply_random_symmetry(
                    states[i], p_targets[i], a1_targets[i], a3_targets[i], a5_targets[i], rng
                )

        states_t = torch.from_numpy(states).to(device)
        p_t = torch.from_numpy(p_targets).to(device)
        v_t = torch.from_numpy(v_targets).to(device)
        a1_t = torch.from_numpy(a1_targets).to(device)
        a2_t = torch.from_numpy(a2_targets).to(device)
        a3_t = torch.from_numpy(a3_targets).to(device)
        a4_t = torch.from_numpy(a4_targets).to(device)
        a5_t = torch.from_numpy(a5_targets).to(device)

        pred_logits, pred_v, pred_a1, pred_a2, pred_a3, pred_a4, pred_a5 = model(states_t)

        # Policy loss
        p_log = torch.log_softmax(pred_logits, dim=1)
        p_loss = -(p_t * p_log).sum(dim=1).mean()

        # Value loss
        v_loss = torch.mean((pred_v - v_t) ** 2)

        # Aux1: stone delta (spatial MSE)
        a1_loss = torch.mean((pred_a1 - a1_t) ** 2)

        # Aux2: territory control (scalar MSE)
        a2_loss = torch.mean((pred_a2 - a2_t) ** 2)

        # Aux3: opponent action distribution, including pass
        a3_log = torch.log_softmax(pred_a3, dim=1)
        a3_loss = -(a3_t * a3_log).sum(dim=1).mean()

        # Aux4: position complexity (scalar MSE, pred is sigmoid-bounded)
        a4_loss = torch.mean((pred_a4 - a4_t) ** 2)

        # Aux5: influence map (spatial MSE)
        a5_loss = torch.mean((pred_a5 - a5_t) ** 2)

        loss = (p_loss + v_loss
                + cfg.lambda_aux1 * a1_loss
                + cfg.lambda_aux2 * a2_loss
                + cfg.lambda_aux3 * a3_loss
                + cfg.lambda_aux4 * a4_loss
                + cfg.lambda_aux5 * a5_loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        batch_grad_norms.append(float(grad_norm))

        optimizer.step()

        if cfg.lr_decay > 0 and scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item())
        total_p_loss += float(p_loss.item())
        total_v_loss += float(v_loss.item())
        total_a1_loss += float(a1_loss.item())
        total_a2_loss += float(a2_loss.item())
        total_a3_loss += float(a3_loss.item())
        total_a4_loss += float(a4_loss.item())
        total_a5_loss += float(a5_loss.item())
        batch_losses.append(float(loss.item()))

    n = float(actual_batches)

    metrics = {
        "loss": total_loss / n,
        "policy_loss": total_p_loss / n,
        "value_loss": total_v_loss / n,
        "aux1_loss": total_a1_loss / n,
        "aux2_loss": total_a2_loss / n,
        "aux3_loss": total_a3_loss / n,
        "aux4_loss": total_a4_loss / n,
        "aux5_loss": total_a5_loss / n,
        "num_batches": actual_batches,
    }

    if batch_losses:
        metrics["first_10_loss"] = sum(batch_losses[:10]) / min(10, len(batch_losses))
        metrics["last_10_loss"] = sum(batch_losses[-10:]) / min(10, len(batch_losses))

    if batch_grad_norms:
        metrics["grad_norm_mean"] = float(np.mean(batch_grad_norms))
        metrics["grad_norm_max"] = float(np.max(batch_grad_norms))

    all_vals = np.array([s.value_target for s in replay_buffer._data])
    metrics["value_target_mean"] = float(np.mean(all_vals))
    metrics["value_target_std"] = float(np.std(all_vals))

    return metrics

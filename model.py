"""Neural network for Gray Go v6 — enriched auxiliary heads.

Changes from v5 (model_v4.py is reused as base):
- 3 new auxiliary heads for implicit knowledge extraction:
  - aux3: opponent move distribution (spatial 9x9) — predicts where opponent played
  - aux4: position complexity (scalar) — entropy of MCTS visit distribution
  - aux5: influence map (spatial 9x9) — predicted territory ownership
- Same backbone: 12 blocks, 128 filters, circular convolutions
- Same policy head: single, player-relative, color-flip at inference
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for model modules.")


if TORCH_AVAILABLE:

    class CircularConv2d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
            super().__init__()
            self.pad = kernel_size // 2
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="circular")
            return self.conv(x)

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = CircularConv2d(channels, channels, kernel_size=3)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = CircularConv2d(channels, channels, kernel_size=3)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.bn2(self.conv2(out))
            return F.relu(out + residual, inplace=True)

    class GrayGoNet(nn.Module):
        """Single policy + value + 5 aux heads with player-relative encoding."""

        def __init__(self, board_size: int = 9, blocks: int = 12, filters: int = 128,
                     in_channels: int = 6) -> None:
            super().__init__()
            self.board_size = board_size
            self.action_size = board_size * board_size + 1

            # Shared backbone
            self.stem_conv = CircularConv2d(in_channels, filters, kernel_size=3)
            self.stem_bn = nn.BatchNorm2d(filters)
            self.res_blocks = nn.ModuleList([ResidualBlock(filters) for _ in range(blocks)])

            # Policy head (single — player-relative)
            self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1)
            self.policy_bn = nn.BatchNorm2d(2)
            self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)

            # Value head (from current player's perspective)
            self.value_conv = nn.Conv2d(filters, 1, kernel_size=1)
            self.value_bn = nn.BatchNorm2d(1)
            self.value_fc1 = nn.Linear(board_size * board_size, filters * 2)
            self.value_fc2 = nn.Linear(filters * 2, 1)

            # Aux1: spatial (S x S) — next-state stone delta
            self.aux1_conv1 = nn.Conv2d(filters, 32, kernel_size=1)
            self.aux1_bn = nn.BatchNorm2d(32)
            self.aux1_conv2 = nn.Conv2d(32, 1, kernel_size=1)

            # Aux2: scalar — territory control ratio
            self.aux2_conv = nn.Conv2d(filters, 1, kernel_size=1)
            self.aux2_bn = nn.BatchNorm2d(1)
            self.aux2_fc1 = nn.Linear(board_size * board_size, filters)
            self.aux2_fc2 = nn.Linear(filters, 1)

            # Aux3: spatial (S x S) — opponent move distribution (NEW)
            self.aux3_conv1 = nn.Conv2d(filters, 32, kernel_size=1)
            self.aux3_bn = nn.BatchNorm2d(32)
            self.aux3_conv2 = nn.Conv2d(32, 1, kernel_size=1)

            # Aux4: scalar — position complexity (NEW)
            self.aux4_conv = nn.Conv2d(filters, 1, kernel_size=1)
            self.aux4_bn = nn.BatchNorm2d(1)
            self.aux4_fc1 = nn.Linear(board_size * board_size, 64)
            self.aux4_fc2 = nn.Linear(64, 1)

            # Aux5: spatial (S x S) — influence map (NEW)
            self.aux5_conv1 = nn.Conv2d(filters, 32, kernel_size=1)
            self.aux5_bn = nn.BatchNorm2d(32)
            self.aux5_conv2 = nn.Conv2d(32, 1, kernel_size=1)

        def forward(self, x: torch.Tensor):
            """Returns (policy_logits, value, aux1, aux2, aux3, aux4, aux5)."""
            # Shared backbone
            x = F.relu(self.stem_bn(self.stem_conv(x)), inplace=True)
            for block in self.res_blocks:
                x = block(x)

            # Policy
            p = F.relu(self.policy_bn(self.policy_conv(x)), inplace=True)
            p = p.view(p.size(0), -1)
            policy_logits = self.policy_fc(p)

            # Value
            v = F.relu(self.value_bn(self.value_conv(x)), inplace=True)
            v = v.view(v.size(0), -1)
            v = F.relu(self.value_fc1(v), inplace=True)
            value = torch.tanh(self.value_fc2(v)).squeeze(-1)

            # Aux1: spatial stone delta
            a1 = F.relu(self.aux1_bn(self.aux1_conv1(x)), inplace=True)
            a1 = self.aux1_conv2(a1).squeeze(1)

            # Aux2: scalar territory control
            a2 = F.relu(self.aux2_bn(self.aux2_conv(x)), inplace=True)
            a2 = a2.view(a2.size(0), -1)
            a2 = F.relu(self.aux2_fc1(a2), inplace=True)
            aux2 = self.aux2_fc2(a2).squeeze(-1)

            # Aux3: spatial opponent move distribution
            a3 = F.relu(self.aux3_bn(self.aux3_conv1(x)), inplace=True)
            a3 = self.aux3_conv2(a3).squeeze(1)

            # Aux4: scalar position complexity
            a4 = F.relu(self.aux4_bn(self.aux4_conv(x)), inplace=True)
            a4 = a4.view(a4.size(0), -1)
            a4 = F.relu(self.aux4_fc1(a4), inplace=True)
            aux4 = torch.sigmoid(self.aux4_fc2(a4)).squeeze(-1)  # [0,1] range

            # Aux5: spatial influence map
            a5 = F.relu(self.aux5_bn(self.aux5_conv1(x)), inplace=True)
            a5 = self.aux5_conv2(a5).squeeze(1)

            return policy_logits, value, a1, aux2, a3, aux4, a5


# ─────────────────────────────────────────────────────────────
# Encoding (unchanged from v4)
# ─────────────────────────────────────────────────────────────

def _get_constants():
    try:
        from graygo_engine import BLACK, WHITE, GRAY, EMPTY, BLACK_PLAYER, WHITE_PLAYER
        return BLACK, WHITE, GRAY, EMPTY, BLACK_PLAYER, WHITE_PLAYER
    except ImportError:
        from engine import BLACK, WHITE, GRAY, EMPTY, BLACK_PLAYER, WHITE_PLAYER
        return BLACK, WHITE, GRAY, EMPTY, BLACK_PLAYER, WHITE_PLAYER

_BLACK, _WHITE, _GRAY, _EMPTY, _BLACK_PLAYER, _WHITE_PLAYER = _get_constants()


def encode_player_relative(state, player: int, board_size: int = 9) -> np.ndarray:
    """Encode board state from a player's perspective (6 channels)."""
    s = board_size
    planes = np.zeros((6, s, s), dtype=np.float32)

    try:
        from graygo_engine import BLACK, WHITE, GRAY, EMPTY, BLACK_PLAYER, WHITE_PLAYER
        board = state.get_board_numpy()
        forbidden = state.get_forbidden_points()
        my_forbidden = forbidden[player]
        opp_forbidden = forbidden[1 - player]
    except ImportError:
        from engine import BLACK, WHITE, GRAY, EMPTY, BLACK_PLAYER, WHITE_PLAYER
        board = state.board.grid
        my_forbidden = state.forbidden_points[player]
        opp_forbidden = state.forbidden_points[1 - player]

    if player == BLACK_PLAYER:
        my_color, opp_color = BLACK, WHITE
    else:
        my_color, opp_color = WHITE, BLACK

    planes[0] = (board == my_color).astype(np.float32)
    planes[1] = (board == opp_color).astype(np.float32)
    planes[2] = (board == GRAY).astype(np.float32)
    planes[3] = (board == EMPTY).astype(np.float32)
    for action in my_forbidden:
        planes[4, action // s, action % s] = 1.0
    for action in opp_forbidden:
        planes[5, action // s, action % s] = 1.0

    return planes


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

def predict(model, state_encoded: np.ndarray):
    """Single forward pass -> (policy, value, aux1-aux5)."""
    require_torch()
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(state_encoded.astype(np.float32)).unsqueeze(0).to(device)
        logits, val, a1, a2, a3, a4, a5 = model(tensor)
        policy = torch.softmax(logits, dim=1)[0].cpu().numpy()
        value = float(val[0].item())
    return policy, value


def predict_batch(model, states: np.ndarray):
    """Batched forward pass."""
    require_torch()
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(states.astype(np.float32)).to(device)
        logits, vals, a1s, a2s, a3s, a4s, a5s = model(tensor)
        policies = torch.softmax(logits, dim=1).cpu().numpy()
        values = vals.cpu().numpy()
    return policies, values


def evaluate_position(model, state, board_size: int = 9):
    """Color-flip inference: run model from both perspectives."""
    from engine import BLACK_PLAYER, WHITE_PLAYER

    black_encoded = encode_player_relative(state, BLACK_PLAYER, board_size)
    white_encoded = encode_player_relative(state, WHITE_PLAYER, board_size)

    bp, bv = predict(model, black_encoded)
    wp, wv = predict(model, white_encoded)

    value = (bv - wv) / 2.0
    return bp, wp, value


def evaluate_position_batch(model, states_black: np.ndarray, states_white: np.ndarray):
    """Batched color-flip inference."""
    bp_all, bv_all = predict_batch(model, states_black)
    wp_all, wv_all = predict_batch(model, states_white)
    values = (bv_all - wv_all) / 2.0
    return bp_all, wp_all, values

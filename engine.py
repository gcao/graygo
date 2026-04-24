"""Core Gray Go game engine for self-play training.

The implementation follows the project specification:
- Toroidal board topology.
- Simultaneous moves with collision-to-gray behavior.
- Two-stage capture resolution order.
- Per-player suicidal-move forbidden points.
- Chinese-style scoring with gray-faction split rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

EMPTY = 0
BLACK = 1
WHITE = 2
GRAY = 3

BLACK_PLAYER = 0
WHITE_PLAYER = 1

PLAYER_COLORS = (BLACK, WHITE)


def action_count(size: int) -> int:
    """Number of policy actions (all board points + pass)."""
    return size * size + 1


def pass_action(size: int) -> int:
    """Action index for pass."""
    return size * size


def is_pass(action: int, size: int) -> bool:
    return action == pass_action(size)


def xy_to_action(x: int, y: int, size: int) -> int:
    return y * size + x


def action_to_xy(action: int, size: int) -> tuple[int, int]:
    if action < 0 or action >= size * size:
        raise ValueError(f"Board action out of range: {action}")
    return action % size, action // size


def territory_player_shares(border_colors: Iterable[int]) -> tuple[float, float]:
    """Convert bordering factions into black/white ownership shares.

    Shares are faction-equal first, then gray ownership is split evenly
    between black and white players.
    """
    colors = set(border_colors)
    if not colors:
        return 0.5, 0.5

    weight = 1.0 / len(colors)
    w_black = weight if BLACK in colors else 0.0
    w_white = weight if WHITE in colors else 0.0
    w_gray = weight if GRAY in colors else 0.0

    black_share = w_black + 0.5 * w_gray
    white_share = w_white + 0.5 * w_gray
    return black_share, white_share


@dataclass
class RoundResolution:
    board_changed: bool
    collision_point: Optional[tuple[int, int]]
    captured_counts: dict[int, int]
    turn_number: int
    game_over: bool


@dataclass
class Board:
    size: int = 9
    grid: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)

    def copy(self) -> "Board":
        result = Board(self.size)
        result.grid = self.grid.copy()
        return result

    def get(self, x: int, y: int) -> int:
        return int(self.grid[y % self.size, x % self.size])

    def set(self, x: int, y: int, value: int) -> None:
        self.grid[y % self.size, x % self.size] = value

    def get_action(self, action: int) -> int:
        x, y = action_to_xy(action, self.size)
        return self.get(x, y)

    def set_action(self, action: int, value: int) -> None:
        x, y = action_to_xy(action, self.size)
        self.set(x, y, value)

    def neighbors(self, x: int, y: int) -> tuple[tuple[int, int], ...]:
        s = self.size
        return (
            ((x - 1) % s, y),
            ((x + 1) % s, y),
            (x, (y - 1) % s),
            (x, (y + 1) % s),
        )

    def _group_and_liberties(
        self,
        start_action: int,
        visited: set[int],
    ) -> tuple[list[int], set[int]]:
        color = self.get_action(start_action)
        if color == EMPTY:
            return [], set()

        group: list[int] = []
        liberties: set[int] = set()
        stack = [start_action]

        while stack:
            action = stack.pop()
            if action in visited:
                continue
            if self.get_action(action) != color:
                continue

            visited.add(action)
            group.append(action)
            x, y = action_to_xy(action, self.size)

            for nx, ny in self.neighbors(x, y):
                na = xy_to_action(nx, ny, self.size)
                neighbor_color = self.get(nx, ny)
                if neighbor_color == EMPTY:
                    liberties.add(na)
                elif neighbor_color == color and na not in visited:
                    stack.append(na)

        return group, liberties

    def dead_groups(self) -> list[list[int]]:
        dead: list[list[int]] = []
        visited: set[int] = set()
        board_points = self.size * self.size

        for action in range(board_points):
            color = self.get_action(action)
            if color == EMPTY or action in visited:
                continue
            group, liberties = self._group_and_liberties(action, visited)
            if group and not liberties:
                dead.append(group)

        return dead

    def remove_groups(self, groups: list[list[int]]) -> dict[int, int]:
        removed = {BLACK: 0, WHITE: 0, GRAY: 0}
        for group in groups:
            if not group:
                continue
            color = self.get_action(group[0])
            if color == EMPTY:
                continue
            removed[color] += len(group)
            for action in group:
                self.set_action(action, EMPTY)
        return removed

    def score(self) -> tuple[float, float]:
        black_score = 0.0
        white_score = 0.0
        board_points = self.size * self.size

        for action in range(board_points):
            color = self.get_action(action)
            if color == BLACK:
                black_score += 1.0
            elif color == WHITE:
                white_score += 1.0
            elif color == GRAY:
                black_score += 0.5
                white_score += 0.5

        visited: set[int] = set()
        for action in range(board_points):
            if action in visited or self.get_action(action) != EMPTY:
                continue

            region: list[int] = []
            borders: set[int] = set()
            stack = [action]

            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                if self.get_action(cur) != EMPTY:
                    continue

                visited.add(cur)
                region.append(cur)
                x, y = action_to_xy(cur, self.size)
                for nx, ny in self.neighbors(x, y):
                    na = xy_to_action(nx, ny, self.size)
                    color = self.get(nx, ny)
                    if color == EMPTY and na not in visited:
                        stack.append(na)
                    elif color != EMPTY:
                        borders.add(color)

            black_share, white_share = territory_player_shares(borders)
            black_score += len(region) * black_share
            white_score += len(region) * white_share

        return black_score, white_score


@dataclass
class GameState:
    size: int = 9
    board: Board = field(init=False)
    turn_number: int = 0
    forbidden_points: list[set[int]] = field(default_factory=lambda: [set(), set()])
    consecutive_double_passes: int = 0
    game_over: bool = False
    # Ko history: last 2 entries of (board_state, black_action, white_action)
    ko_history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.board = Board(self.size)

    def copy(self) -> "GameState":
        result = GameState(self.size)
        result.board = self.board.copy()
        result.turn_number = self.turn_number
        result.forbidden_points = [
            set(self.forbidden_points[BLACK_PLAYER]),
            set(self.forbidden_points[WHITE_PLAYER]),
        ]
        result.consecutive_double_passes = self.consecutive_double_passes
        result.game_over = self.game_over
        result.ko_history = [(b.copy(), ba, wa) for b, ba, wa in self.ko_history]
        return result

    def legal_actions(self, player: int) -> np.ndarray:
        if player not in (BLACK_PLAYER, WHITE_PLAYER):
            raise ValueError(f"Invalid player index: {player}")

        total_actions = action_count(self.size)
        mask = np.zeros(total_actions, dtype=bool)
        board_points = self.size * self.size

        occupied = self.board.grid.reshape(-1) != EMPTY
        mask[:board_points] = ~occupied

        if self.forbidden_points[player]:
            mask[list(self.forbidden_points[player])] = False

        mask[pass_action(self.size)] = True
        return mask

    def is_legal_action(self, player: int, action: int) -> bool:
        if is_pass(action, self.size):
            return True
        if action < 0 or action >= self.size * self.size:
            return False
        if self.board.get_action(action) != EMPTY:
            return False
        if action in self.forbidden_points[player]:
            return False
        return True

    def _resolve_captures(self, newly_placed: set[int]) -> dict[int, int]:
        total_removed = {BLACK: 0, WHITE: 0, GRAY: 0}

        # Step 3/4: remove dead groups that do not contain newly placed stones.
        dead_groups = self.board.dead_groups()
        first_pass = [group for group in dead_groups if newly_placed.isdisjoint(group)]
        removed = self.board.remove_groups(first_pass)
        for color, count in removed.items():
            total_removed[color] += count

        # Step 5: recheck and remove remaining dead groups simultaneously.
        dead_groups = self.board.dead_groups()
        removed = self.board.remove_groups(dead_groups)
        for color, count in removed.items():
            total_removed[color] += count

        return total_removed

    def step(self, black_action: int, white_action: int) -> RoundResolution:
        if self.game_over:
            raise RuntimeError("Game is already over.")
        if not self.is_legal_action(BLACK_PLAYER, black_action):
            raise ValueError(f"Illegal black action: {black_action}")
        if not self.is_legal_action(WHITE_PLAYER, white_action):
            raise ValueError(f"Illegal white action: {white_action}")

        both_pass = is_pass(black_action, self.size) and is_pass(white_action, self.size)
        if both_pass:
            self.consecutive_double_passes += 1
        else:
            self.consecutive_double_passes = 0

        board_before = self.board.grid.copy()
        collision_point: Optional[tuple[int, int]] = None
        newly_placed: set[int] = set()

        if not both_pass:
            black_is_pass = is_pass(black_action, self.size)
            white_is_pass = is_pass(white_action, self.size)

            if not black_is_pass and not white_is_pass and black_action == white_action:
                x, y = action_to_xy(black_action, self.size)
                self.board.set(x, y, GRAY)
                collision_point = (x, y)
                newly_placed.add(black_action)
            else:
                if not black_is_pass:
                    x, y = action_to_xy(black_action, self.size)
                    self.board.set(x, y, BLACK)
                    newly_placed.add(black_action)
                if not white_is_pass:
                    x, y = action_to_xy(white_action, self.size)
                    self.board.set(x, y, WHITE)
                    newly_placed.add(white_action)

            captured_counts = self._resolve_captures(newly_placed)
        else:
            captured_counts = {BLACK: 0, WHITE: 0, GRAY: 0}

        board_changed = not np.array_equal(board_before, self.board.grid)

        if board_changed:
            self.forbidden_points = [set(), set()]

            # Ko detection: if current board matches a previous state in ko_history,
            # the moves from that round become forbidden (but pass is always allowed)
            for hist_board, hist_black, hist_white in self.ko_history:
                if np.array_equal(self.board.grid, hist_board):
                    if not is_pass(hist_black, self.size):
                        self.forbidden_points[BLACK_PLAYER].add(hist_black)
                    if not is_pass(hist_white, self.size):
                        self.forbidden_points[WHITE_PLAYER].add(hist_white)
                    break
        else:
            if not is_pass(black_action, self.size):
                self.forbidden_points[BLACK_PLAYER].add(black_action)
            if not is_pass(white_action, self.size):
                self.forbidden_points[WHITE_PLAYER].add(white_action)

        # Update ko history: save pre-turn board state + moves (keep last 2)
        self.ko_history.append((board_before, black_action, white_action))
        if len(self.ko_history) > 2:
            self.ko_history.pop(0)

        self.turn_number += 1
        if both_pass and self.consecutive_double_passes >= 2:
            self.game_over = True

        return RoundResolution(
            board_changed=board_changed,
            collision_point=collision_point,
            captured_counts=captured_counts,
            turn_number=self.turn_number,
            game_over=self.game_over,
        )

    def score(self) -> tuple[float, float]:
        return self.board.score()

    def winner_color(self) -> int:
        black_score, white_score = self.score()
        if black_score > white_score:
            return BLACK
        if white_score > black_score:
            return WHITE
        return EMPTY

    def winner_player(self) -> Optional[int]:
        winner = self.winner_color()
        if winner == BLACK:
            return BLACK_PLAYER
        if winner == WHITE:
            return WHITE_PLAYER
        return None


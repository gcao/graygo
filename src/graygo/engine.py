"""Immutable reference implementation of docs/design.md §1.

Actions are row-major intersection indices, with size**2 denoting pass.
No random state or opponent commitments are hidden inside the resolver.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from numbers import Integral
import json

EMPTY, BLACK, WHITE, GRAY = range(4)
RULE_VERSION = "graygo/1"


@dataclass(frozen=True)
class Rules:
    size: int = 9
    max_rounds: int | None = None
    collision: str = "gray"

    def __post_init__(self):
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 3:
            raise ValueError("size must be an integer >= 3 (four distinct neighbors)")
        if self.max_rounds is None:
            if self.size != 9:
                raise ValueError("non-9x9 research games require an explicit round limit")
            object.__setattr__(self, "max_rounds", 200)
        if isinstance(self.max_rounds, bool) or not isinstance(self.max_rounds, int) or self.max_rounds < 1:
            raise ValueError("max_rounds must be a positive integer")
        if self.collision not in ("gray", "remove"):
            raise ValueError("collision must be 'gray' or the labeled 'remove' ablation")

    @property
    def points(self) -> int:
        return self.size * self.size

    @property
    def pass_action(self) -> int:
        return self.points

    def action(self, row: int, col: int) -> int:
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError("coordinate outside board")
        return row * self.size + col


def _colors(values):
    """Normalize integral colors; ordinary Python ints avoid expensive ABC checks."""
    values = tuple(values)
    normalized = None
    for index, value in enumerate(values):
        if type(value) is int:
            color = value
        elif not isinstance(value, bool) and isinstance(value, Integral):
            color = int(value)
            if normalized is None:
                normalized = list(values[:index])
        else:
            raise ValueError("integer colors 0..3 required")
        if not 0 <= color <= 3:
            raise ValueError("integer colors 0..3 required")
        if normalized is not None:
            normalized.append(color)
    return values if normalized is None else tuple(normalized)


def _board(values, rules):
    values = tuple(values)
    if len(values) != rules.points:
        raise ValueError("board must contain size**2 integer colors 0..3")
    try:
        return _colors(values)
    except ValueError:
        raise ValueError("board must contain size**2 integer colors 0..3") from None


def _action(action, rules):
    if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action <= rules.points:
        raise ValueError("invalid action index")
    return int(action)


@dataclass(frozen=True)
class RoundRecord:
    before: tuple[int, ...]
    black: int
    white: int

    def __post_init__(self):
        # Copy and normalize values so a record cannot alias a caller's array.
        try:
            values = _colors(self.before)
        except ValueError:
            raise ValueError("invalid historical board colors") from None
        object.__setattr__(self, "before", values)
        for field in ("black", "white"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError("invalid historical action")
            object.__setattr__(self, field, int(value))


@dataclass(frozen=True)
class State:
    rules: Rules = Rules()
    board: tuple[int, ...] = ()
    forbidden_black: frozenset[int] = frozenset()
    forbidden_white: frozenset[int] = frozenset()
    history: tuple[RoundRecord, ...] = ()  # oldest first
    pass_streak: int = 0
    rounds_played: int = 0

    def __post_init__(self):
        # () is the convenient empty-board initializer; malformed supplied boards fail.
        raw = (EMPTY,) * self.rules.points if isinstance(self.board, tuple) and not self.board else self.board
        object.__setattr__(self, "board", _board(raw, self.rules))
        for name in ("forbidden_black", "forbidden_white"):
            points = frozenset(_action(p, self.rules) for p in getattr(self, name))
            if any(_action(p, self.rules) == self.rules.pass_action for p in points):
                raise ValueError("pass cannot be forbidden")
            object.__setattr__(self, name, points)
        history = tuple(self.history)
        if len(history) > 2:
            raise ValueError("only two previous completed round records may be saved")
        for record in history:
            if not isinstance(record, RoundRecord):
                raise ValueError("history must contain RoundRecords")
            _board(record.before, self.rules)
            _action(record.black, self.rules)
            _action(record.white, self.rules)
        object.__setattr__(self, "history", history)
        if type(self.pass_streak) is not int or not 0 <= self.pass_streak <= 2:
            raise ValueError("invalid mutual-pass count")
        if type(self.rounds_played) is not int or not 0 <= self.rounds_played <= self.rules.max_rounds:
            raise ValueError("invalid round count")
        if len(history) != min(self.rounds_played, 2) or self.pass_streak > self.rounds_played:
            raise ValueError("history must describe the last two completed rounds")
        trailing_passes = 0
        for record in reversed(history):
            if record.black != self.rules.pass_action or record.white != self.rules.pass_action:
                break
            trailing_passes += 1
        if self.pass_streak != trailing_passes:
            raise ValueError("mutual-pass count disagrees with history")
        if any(not group.liberties for group in groups(self.board, self.rules.size)):
            raise ValueError("a resolved starting state cannot contain zero-liberty groups")

    @property
    def done(self):
        return self.pass_streak == 2 or self.rounds_played == self.rules.max_rounds

    @property
    def terminal_reason(self):
        if self.pass_streak == 2:
            return "mutual_pass"
        if self.rounds_played == self.rules.max_rounds:
            return "round_limit"
        return None

    def forbidden(self, player):
        if player == BLACK:
            return self.forbidden_black
        if player == WHITE:
            return self.forbidden_white
        raise ValueError("player must be BLACK or WHITE")

    def legal_actions(self, player):
        blocked = self.forbidden(player)
        if self.done:
            return ()
        return tuple(p for p, color in enumerate(self.board) if color == EMPTY and p not in blocked) + (self.rules.pass_action,)

    def to_dict(self):
        return {
            "rule_version": RULE_VERSION,
            "rules": {"size": self.rules.size, "max_rounds": self.rules.max_rounds, "collision": self.rules.collision},
            "board": list(self.board),
            "forbidden_black": sorted(self.forbidden_black),
            "forbidden_white": sorted(self.forbidden_white),
            "history": [{"before": list(h.before), "black": h.black, "white": h.white} for h in self.history],
            "pass_streak": self.pass_streak, "rounds_played": self.rounds_played,
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("state must be a JSON object")
        if value.get("rule_version") != RULE_VERSION:
            raise ValueError("unsupported rule version")
        expected = {"rule_version", "rules", "board", "forbidden_black", "forbidden_white", "history", "pass_streak", "rounds_played"}
        if set(value) != expected:
            raise ValueError("state schema fields do not match version 1")
        rules = Rules(**value["rules"])
        return cls(rules, _board(value["board"], rules),
                   frozenset(value["forbidden_black"]), frozenset(value["forbidden_white"]),
                   tuple(RoundRecord(tuple(h["before"]), h["black"], h["white"]) for h in value["history"]),
                   value["pass_streak"], value["rounds_played"])

    def canonical_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Group:
    color: int
    stones: frozenset[int]
    liberties: frozenset[int]


@lru_cache(maxsize=16)
def neighbors(size):
    return tuple((((r - 1) % size) * size + c, ((r + 1) % size) * size + c,
                  r * size + (c - 1) % size, r * size + (c + 1) % size)
                 for r in range(size) for c in range(size))


def groups(board, size):
    adjacent = neighbors(size)
    seen = set()
    result = []
    for point, color in enumerate(board):
        if color == EMPTY or point in seen:
            continue
        stones, liberties, stack = {point}, set(), [point]
        seen.add(point)
        while stack:
            for other in adjacent[stack.pop()]:
                if board[other] == EMPTY:
                    liberties.add(other)
                elif board[other] == color and other not in seen:
                    seen.add(other)
                    stones.add(other)
                    stack.append(other)
        result.append(Group(color, frozenset(stones), frozenset(liberties)))
    return tuple(result)


@dataclass(frozen=True)
class Score:
    black_units: int
    white_units: int

    @property
    def black(self):
        return Fraction(self.black_units, 4)

    @property
    def white(self):
        return Fraction(self.white_units, 4)

    @property
    def utility_black(self):
        return (self.black_units > self.white_units) - (self.black_units < self.white_units)


def score(board, size):
    board = _board(board, Rules(size, 1))
    adjacent, seen = neighbors(size), set()
    black = 0
    region_share = {0: 2, 1: 4, 2: 0, 3: 2, 4: 2, 5: 3, 6: 1, 7: 2}
    for point, color in enumerate(board):
        if color:
            black += {BLACK: 4, WHITE: 0, GRAY: 2}[color]
        elif point not in seen:
            stack, area, border = [point], 0, 0
            seen.add(point)
            while stack:
                area += 1
                for other in adjacent[stack.pop()]:
                    if board[other]:
                        border |= 1 << (board[other] - 1)
                    elif other not in seen:
                        seen.add(other)
                        stack.append(other)
            black += area * region_share[border]
    return Score(black, 4 * size * size - black)


@dataclass(frozen=True)
class Transition:
    state: State
    black: int
    white: int
    collision: int | None
    captured: tuple[Group, ...]
    suicidal: tuple[Group, ...]
    board_changed: bool
    reward_black: int


def resolve(state, black, white):
    """Validate both commitments against one pre-round state, then resolve once."""
    if state.done:
        raise ValueError("cannot play after termination")
    rules = state.rules
    black, white = _action(black, rules), _action(white, rules)
    if black not in state.legal_actions(BLACK) or white not in state.legal_actions(WHITE):
        raise ValueError("occupied or player-forbidden placement")
    board, placed = list(state.board), set()
    collision = black if black == white and black != rules.pass_action else None
    if collision is not None:
        if rules.collision == "gray":
            board[collision] = GRAY
            placed.add(collision)
    else:
        for action, color in ((black, BLACK), (white, WHITE)):
            if action != rules.pass_action:
                board[action] = color
                placed.add(action)
    captured = tuple(g for g in groups(board, rules.size) if not g.liberties and not g.stones & placed)
    for group in captured:
        for p in group.stones:
            board[p] = EMPTY
    suicidal = tuple(g for g in groups(board, rules.size) if not g.liberties)
    for group in suicidal:
        for p in group.stones:
            board[p] = EMPTY
    board = tuple(board)
    changed = board != state.board
    matching = tuple(h for h in state.history if h.before == board)
    fb, fw = set(state.forbidden_black), set(state.forbidden_white)
    if changed and not matching:
        fb.clear()
        fw.clear()
    if not changed:
        if black != rules.pass_action:
            fb.add(black)
        if white != rules.pass_action:
            fw.add(white)
    for h in matching:
        if h.black != rules.pass_action:
            fb.add(h.black)
        if h.white != rules.pass_action:
            fw.add(h.white)
    next_state = State(rules, board, frozenset(fb), frozenset(fw),
                       (state.history + (RoundRecord(state.board, black, white),))[-2:],
                       state.pass_streak + 1 if black == white == rules.pass_action else 0,
                       state.rounds_played + 1)
    reward = score(board, rules.size).utility_black if next_state.done else 0
    return Transition(next_state, black, white, collision, captured, suicidal, changed, reward)


def transform_action(action, rules, *, shift=(0, 0), rotation=0, reflect=False):
    action = _action(action, rules)
    if action == rules.pass_action:
        return action
    n = rules.size
    row, col = divmod(action, n)
    if reflect:
        col = n - 1 - col
    for _ in range(rotation % 4):
        row, col = col, n - 1 - row
    return ((row + shift[0]) % n) * n + (col + shift[1]) % n


def transform(state, *, shift=(0, 0), rotation=0, reflect=False, swap=False):
    """Transform the complete state, including ko records and player ownership."""
    def action(p):
        return transform_action(p, state.rules, shift=shift, rotation=rotation, reflect=reflect)

    def board(values):
        out = [0] * state.rules.points
        for p, color in enumerate(values):
            out[action(p)] = {BLACK: WHITE, WHITE: BLACK}.get(color, color) if swap else color
        return tuple(out)

    fb, fw = state.forbidden_black, state.forbidden_white
    if swap:
        fb, fw = fw, fb
    records = tuple(RoundRecord(board(h.before), action(h.white if swap else h.black),
                                action(h.black if swap else h.white)) for h in state.history)
    return State(state.rules, board(state.board), frozenset(map(action, fb)), frozenset(map(action, fw)),
                 records, state.pass_streak, state.rounds_played)

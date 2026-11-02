import itertools
import json
import random
import unittest
from graygo.engine import (BLACK as B, WHITE as W, GRAY as G, EMPTY as E,
                           Rules, State, RoundRecord, neighbors, groups, resolve, score,
                           transform, transform_action)


def position(cells, size=9, limit=None):
    rules = Rules(size, limit)
    board = [E] * rules.points
    for (r, c), color in cells.items():
        board[r * size + c] = color
    return State(rules, tuple(board))


def rings():
    cells = {}
    for center, color in (((2, 2), W), ((6, 6), B)):
        r, c = center
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cells[r + dr, c + dc] = color
    return position(cells)


def ko_position():
    cells = {(3, 3): B, (2, 3): W, (4, 3): W, (3, 2): W,
             (2, 4): B, (4, 4): B, (3, 5): B}
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cells[6 + dr, 6 + dc] = W
    return position(cells)


class EngineTests(unittest.TestCase):
    def test_topology(self):
        for n in (3, 5, 9):
            self.assertTrue(all(len(set(adj)) == 4 for adj in neighbors(n)))
            self.assertEqual(set(neighbors(n)[0]), {n - 1, n * (n - 1), 1, n})
        s = position({(0, 0): B, (0, 8): B})
        self.assertEqual(len(groups(s.board, 9)), 1)

    def test_both_moves_and_collision(self):
        s = State()
        t = resolve(s, 0, 1)
        self.assertEqual(t.state.board[:2], (B, W))
        self.assertEqual(s.board, (E,) * 81)
        t = resolve(s, 0, 0)
        self.assertEqual(t.state.board[0], G)
        self.assertEqual(t.collision, 0)
        self.assertEqual(t.reward_black, 0)
        self.assertEqual(t.state.history, (RoundRecord(s.board, 0, 0),))

    def test_simultaneous_suicide_is_not_sequential_rescue(self):
        s = position({(1, 2): W, (3, 2): W, (2, 1): W,
                      (1, 3): B, (3, 3): B, (2, 4): B})
        x, y = 2 * 9 + 2, 2 * 9 + 3
        t = resolve(s, x, y)
        self.assertEqual(t.state.board, s.board)
        self.assertEqual(len(t.suicidal), 2)
        self.assertEqual(t.captured, ())
        self.assertEqual(t.state.forbidden_black, {x})
        self.assertEqual(t.state.forbidden_white, {y})

    def test_capture_precedes_suicide_owner_fixture(self):
        s = position({(1, 2): W, (1, 3): B, (2, 1): W, (2, 3): W, (2, 4): B,
                      (3, 1): W, (3, 2): B, (3, 3): B, (3, 4): W, (4, 3): W})
        t = resolve(s, 2 * 9 + 2, 4 * 9 + 2)
        self.assertEqual(t.state.board[2 * 9 + 3], E)
        self.assertEqual(t.state.board[2 * 9 + 2], B)
        self.assertEqual(len(t.captured), 1)
        self.assertEqual(t.suicidal, ())

    def test_mutual_old_groups_capture_together(self):
        s = position({(3, 2): W, (3, 3): B, (3, 4): W, (3, 5): B,
                      (4, 3): W, (4, 4): B})
        t = resolve(s, 2 * 9 + 4, 2 * 9 + 3)
        self.assertEqual(t.state.board[3 * 9 + 3:3 * 9 + 5], (E, E))
        self.assertEqual(len(t.captured), 2)
        self.assertEqual(t.suicidal, ())

    def test_gray_captures_and_is_captured(self):
        # White's last liberty is filled by a gray collision; the newborn gray is rescued.
        s = position({(3, 3): W, (2, 3): B, (4, 3): B, (3, 2): B,
                      (2, 4): W, (4, 4): W, (3, 5): W})
        p = 3 * 9 + 4
        t = resolve(s, p, p)
        self.assertEqual(t.state.board[p], G)
        self.assertEqual(t.state.board[3 * 9 + 3], E)
        s = position({(3, 3): G, (2, 3): W, (4, 3): W, (3, 2): W})
        t = resolve(s, 81, p)
        self.assertEqual(t.state.board[3 * 9 + 3], E)
        self.assertEqual(t.captured[0].color, G)

    def test_suicide_accumulates_and_new_board_clears(self):
        s = rings()
        x, y = 2 * 9 + 2, 6 * 9 + 6
        t = resolve(s, x, 81).state
        self.assertEqual(t.forbidden_black, {x})
        with self.assertRaises(ValueError):
            resolve(t, x, 81)
        u = resolve(t, 81, y).state
        self.assertEqual(u.board, s.board)
        self.assertEqual(u.forbidden_black, {x})
        self.assertEqual(u.forbidden_white, {y})
        v = resolve(u, 0, 81).state
        self.assertFalse(v.forbidden_black | v.forbidden_white)

    def test_disappearing_collision_forbids_both(self):
        s = rings()
        x = 20
        t = resolve(s, x, x)
        self.assertEqual(t.state.board, s.board)
        self.assertEqual(t.state.forbidden_black, {x})
        self.assertEqual(t.state.forbidden_white, {x})

    def test_outgoing_move_ko_and_retaining_old_ban(self):
        s = ko_position()
        x, y, z = 3 * 9 + 3, 3 * 9 + 4, 6 * 9 + 6
        first = resolve(s, 81, y).state
        self.assertEqual(first.board[x], E)
        middle = resolve(first, z, 81).state  # no change, black now forbidden at z
        last = resolve(middle, x, 81).state
        self.assertEqual(last.board, s.board)
        self.assertEqual(last.forbidden_white, {y})
        self.assertEqual(last.forbidden_black, {z})
        with self.assertRaises(ValueError):
            resolve(last, 81, y)

    def test_oldest_ko_record_expires(self):
        s = ko_position()
        x, y, z = 30, 31, 60
        first = resolve(s, 81, y).state
        second = resolve(first, z, 81).state
        third = resolve(second, 81, 81).state
        final = resolve(third, x, 81).state
        self.assertEqual(final.board, s.board)
        self.assertFalse(final.forbidden_black | final.forbidden_white)
        self.assertEqual(len(final.history), 2)

    def test_pass_and_final_round(self):
        s = resolve(State(), 81, 81).state
        self.assertFalse(s.done)
        s = resolve(s, 81, 81).state
        self.assertTrue(s.done)
        self.assertEqual(s.terminal_reason, "mutual_pass")
        self.assertEqual(s.legal_actions(B), ())
        with self.assertRaises(ValueError):
            resolve(s, 81, 81)
        # A disappearing placement resets the streak rather than counting as pass.
        s = resolve(rings(), 81, 81).state
        self.assertEqual(resolve(s, 20, 81).state.pass_streak, 0)
        s = State()
        while not s.done:
            s = resolve(s, s.legal_actions(B)[0], 81).state
        self.assertEqual(s.rounds_played, 200)
        self.assertEqual(s.terminal_reason, "round_limit")
        one = Rules(3, 1)
        t = resolve(State(one), 0, one.pass_action)
        self.assertTrue(t.state.done)
        self.assertEqual(t.reward_black, 1)
        self.assertEqual(score(t.state.board, 3).black, 9)

    def test_legal_validation_and_roundtrip(self):
        with self.assertRaises(ValueError):
            Rules(2)
        with self.assertRaises(ValueError):
            Rules(5)
        for action in (-1, 82, True, "pass"):
            with self.assertRaises(ValueError):
                resolve(State(), action, 81)
        s = resolve(State(), 0, 1).state
        with self.assertRaises(ValueError):
            resolve(s, 0, 81)
        self.assertEqual(State.from_dict(json.loads(s.canonical_json())), s)
        self.assertEqual(hash(State.from_dict(s.to_dict())), hash(s))
        with self.assertRaises(ValueError):
            State(forbidden_black={81})
        # Two visually identical boards with different restrictions/history are different states.
        a = resolve(rings(), 20, 81).state
        self.assertNotEqual(a, rings())

    def test_collision_ablation(self):
        s = State(Rules(9, collision="remove"))
        t = resolve(s, 0, 0)
        self.assertEqual(t.state.board, s.board)
        self.assertEqual(t.state.forbidden_black, {0})
        self.assertEqual(t.state.forbidden_white, {0})

    def test_scoring_all_boundary_sets_and_exact_units(self):
        for subset in ((), (B,), (W,), (G,), (B,W), (B,G), (W,G), (B,W,G)):
            board = [subset[0] if subset else E] * 25
            if subset:
                center = 12
                board[center] = E
                for i, p in enumerate(neighbors(5)[center]):
                    board[p] = subset[i % len(subset)]
                living = sum({E:0, B:4, W:0, G:2}[v] for v in board)
                expected = {(B,):4,(W,):0,(G,):2,(B,W):2,(B,G):3,(W,G):1,(B,W,G):2}[subset]
                self.assertEqual(score(board, 5).black_units, living + expected)
            sc = score(board, 5)
            self.assertEqual(sc.black_units + sc.white_units, 100)
        self.assertEqual(score([E]*81, 9).black, score([E]*81, 9).white)

    def test_complete_state_symmetry_commutes_with_resolution(self):
        rng = random.Random(9482)
        states = [ko_position(), rings(), State()]
        s = State(Rules(5, 40))
        for _ in range(20):
            if s.done:
                break
            s = resolve(s, rng.choice(s.legal_actions(B)), rng.choice(s.legal_actions(W))).state
            states.append(s)
        for s in states:
            a, b = rng.choice(s.legal_actions(B)), rng.choice(s.legal_actions(W))
            target = resolve(s, a, b).state
            for rot, mirror, swap in itertools.product(range(4), (False,True), (False,True)):
                kw = dict(shift=(1, -2), rotation=rot, reflect=mirror)
                ts = transform(s, swap=swap, **kw)
                ta, tb = transform_action(a, s.rules, **kw), transform_action(b, s.rules, **kw)
                actual = resolve(ts, tb if swap else ta, ta if swap else tb).state
                self.assertEqual(actual, transform(target, swap=swap, **kw))
                sc, tc = score(s.board, s.rules.size), score(ts.board, ts.rules.size)
                self.assertEqual(tc.black_units, sc.white_units if swap else sc.black_units)

    def test_history_does_not_alias_mutable_inputs(self):
        board=list(State().board)
        record=RoundRecord(board,81,81)
        board[0]=B
        self.assertEqual(record.before[0],E)
        snapshot=State().to_dict()
        snapshot["board"]=[]
        with self.assertRaises(ValueError):State.from_dict(snapshot)

    def test_complete_history_and_pass_metadata_are_required(self):
        with self.assertRaises(ValueError):State(rounds_played=2)
        with self.assertRaises(ValueError):State(rounds_played=1,history=(RoundRecord(State().board,81,81),),pass_streak=0)

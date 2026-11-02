import unittest
import numpy as np
from graygo.engine import BLACK,WHITE,Rules,State
from graygo.solving import solve_matrix,value_gaps,FiniteSolver,SolveBudgetExceeded,markov_best_response


class SolveTests(unittest.TestCase):
    def test_known_matrix_games(self):
        for a,expected in (([[1,-1],[-1,1]],0),([[1,-1],[-1,.5]],-1/7),
                            ([[0,-1,1],[1,0,-1],[-1,1,0]],0),([[-1,-1],[1,1]],1)):
            s=solve_matrix(a)
            self.assertAlmostEqual(s.value,expected,places=8)
            self.assertLess(s.residual_gap,1e-8)
            self.assertLess(max(value_gaps(a,s.row,s.column,s.value)),1e-8)
        s=solve_matrix([[1,-1],[-1,.5]])
        np.testing.assert_allclose(s.row,[3/7,4/7],atol=1e-8)
        np.testing.assert_allclose(s.column,[3/7,4/7],atol=1e-8)

    def test_actual_gray_go_one_round_game(self):
        state=State(Rules(3,1))
        solver=FiniteSolver()
        value,black,white,solution=solver.solve(state)
        self.assertAlmostEqual(value,0)
        self.assertEqual(len(black),10)
        self.assertEqual(len(white),10)
        self.assertLess(solution.row[-1],1e-8)
        self.assertLess(solution.column[-1],1e-8)
        # Best responses optimize all available moves and follow actual resolver outcomes.
        br_b=markov_best_response(state,lambda s:solution.column,BLACK)
        br_w=markov_best_response(state,lambda s:solution.row,WHITE)
        self.assertAlmostEqual(br_b,0)
        self.assertAlmostEqual(br_w,0)
        with self.assertRaises(SolveBudgetExceeded):FiniteSolver(max_nodes=1).solve(state)

    def test_reject_invalid_values_and_policies(self):
        with self.assertRaises(ValueError):solve_matrix([[float('nan')]])
        with self.assertRaises(ValueError):value_gaps([[1,-1]],[-1],[.5,.5])


    def test_reachable_mixed_two_round_fixture(self):
        import json
        from pathlib import Path
        from graygo.engine import resolve
        fixture=json.loads((Path(__file__).parent/"fixtures"/"mixed_subgame.json").read_text())
        root=State.from_dict(fixture["state"])
        replay=State(root.rules)
        for b,w in fixture["prefix"]:replay=resolve(replay,b,w).state
        self.assertEqual(root,replay)
        solver=FiniteSolver()
        value,bs,ws,solution=solver.solve(root)
        self.assertAlmostEqual(value,-.25,places=8)
        np.testing.assert_allclose(solution.row,[.25,.25,.5],atol=1e-8)
        np.testing.assert_allclose(solution.column,[.25,.25,.5],atol=1e-8)
        a=np.array([[solver.solve(resolve(root,b,w).state)[0] for w in ws] for b in bs])
        np.testing.assert_allclose(a,np.diag([-1.,-1.,-.5]),atol=1e-8)
        self.assertLess(a.min(1).max(),a.max(0).min())
        self.assertAlmostEqual(markov_best_response(root,lambda s:solver.policy(s,WHITE),BLACK),value)
        self.assertAlmostEqual(markov_best_response(root,lambda s:solver.policy(s,BLACK),WHITE),value)
        self.assertLess(solver.numerical_value_error_bound(root),1e-7)

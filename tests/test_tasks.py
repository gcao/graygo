from collections import Counter
from copy import deepcopy
import unittest
import numpy as np
from graygo.engine import BLACK,WHITE
from graygo.public_game import PublicGameSolver,evaluate_markov,markov_response,markov_gaps
from graygo.solving import SolveBudgetExceeded
from graygo.tasks import DelayedGame,DelayedState,BranchingGame,BranchState,BalancedCoverage


class TaskTests(unittest.TestCase):
    def test_forced_chains_all_declared_horizons_roles_and_value_targets(self):
        game=DelayedGame(equivalent_actions=1,codebook_seed=12)
        for h in (50,100,200):
            for role in (BLACK,WHITE):
                root=game.root(h,role);solver=PublicGameSolver(game)
                sign=1 if role==BLACK else -1
                self.assertAlmostEqual(solver.solve(root)[0],sign)
                uniform=lambda s:np.ones(len(game.actions(s,BLACK)))/len(game.actions(s,BLACK))
                uniform_w=lambda s:np.ones(len(game.actions(s,WHITE)))/len(game.actions(s,WHITE))
                values=evaluate_markov(game,root,uniform,uniform_w)
                self.assertEqual(values[root].value,0.)
                own=values[root].black_q if role==BLACK else values[root].white_q
                self.assertEqual(own,(1.,-1.))
                for branch in (0,1):
                    state=game.transition(root,branch if role==BLACK else 0,branch if role==WHITE else 0)
                    self.assertEqual(values[state].value,sign*(1 if branch==0 else -1))
                best=lambda s:solver.policy(s,BLACK)
                best_w=lambda s:solver.policy(s,WHITE)
                self.assertEqual(markov_gaps(game,root,best,best_w,sign),(0.,0.))

    def test_40_action_stress_has_same_transition_for_all_joint_actions(self):
        game=DelayedGame(horizons=(3,),equivalent_actions=40)
        root=game.root(3);state=game.transition(root,0,0)
        self.assertEqual(len(game.actions(state,BLACK)),40);self.assertEqual(len(game.actions(state,WHITE)),40)
        next_states={game.transition(state,b,w) for b in range(40) for w in range(40)}
        self.assertEqual(len(next_states),1)
        solved=PublicGameSolver(game).solve(root)
        self.assertEqual(solved[0],1.)

    def test_opaque_codebook_and_no_branch_feature_or_outcome_range(self):
        game=DelayedGame(horizons=(3,),equivalent_actions=40,codebook_seed=7)
        other=DelayedGame(horizons=(3,),equivalent_actions=40,codebook_seed=8)
        self.assertNotEqual(game.encoding.digest,other.encoding.digest)
        self.assertEqual(len(set(game.encoding.ids.values())),game.encoding.features)
        root=game.root(3)
        favorable,unfavorable=game.transition(root,0,0),game.transition(root,1,0)
        a,b=game.observation(favorable,BLACK),game.observation(unfavorable,BLACK)
        self.assertEqual(int(a.spatial.sum()),1);self.assertEqual(int(b.spatial.sum()),1)
        self.assertFalse(np.array_equal(a.spatial,b.spatial))
        np.testing.assert_array_equal(a.global_features,b.global_features)
        np.testing.assert_array_equal(a.legal,b.legal)
        self.assertEqual(a.global_features.shape,(1,))
        game.spec.pack(a)
        state=favorable
        while not game.terminal(state):state=game.transition(state,0,0)
        terminal=game.observation(state,BLACK)
        self.assertFalse(terminal.legal.any());self.assertEqual(int(terminal.spatial.sum()),0)
        self.assertEqual(game.utility(state),1.)

    def test_branching_exact_values_role_swap_and_root_action_values(self):
        game=BranchingGame(codebook_seed=6)
        for role in (BLACK,WHITE):
            root=game.root(role);solver=PublicGameSolver(game)
            self.assertAlmostEqual(solver.solve(root)[0],-1/7 if role==BLACK else 1/7)
            for branch,value in ((0,-1/7),(1,-7/22)):
                node=game.transition(root,branch if role==BLACK else 0,branch if role==WHITE else 0)
                self.assertAlmostEqual(solver.solve(node)[0],value if role==BLACK else -value)
            self.assertAlmostEqual(solver.policy(root,role)[0],1.)
            black=lambda s:solver.policy(s,BLACK);white=lambda s:solver.policy(s,WHITE)
            self.assertTrue(max(markov_gaps(game,root,black,white,solver.solve(root)[0]))<1e-12)
        root=game.root(BLACK)
        black=lambda s:np.ones(len(game.actions(s,BLACK)))/len(game.actions(s,BLACK))
        white=lambda s:np.ones(len(game.actions(s,WHITE)))/len(game.actions(s,WHITE))
        values=evaluate_markov(game,root,black,white)
        self.assertEqual(values[root].black_q,(-.125,-.3125));self.assertEqual(values[root].value,-.21875)
        gaps=markov_gaps(game,root,black,white,-1/7)
        np.testing.assert_allclose(gaps,[.3125-1/7,1/7],atol=1e-12,rtol=0)

    def test_later_action_matters_for_off_policy_targets(self):
        game=BranchingGame();root=game.root();later=game.transition(root,0,0)
        # A behavior suffix (b=0,w=1) returns -1. A target suffix (b=0,w=0) returns +1.
        self.assertEqual(game.utility(game.transition(later,0,1)),-1.)
        self.assertEqual(game.utility(game.transition(later,0,0)),1.)
        first=lambda s:np.array([1.,0.]) if len(game.actions(s,BLACK))==2 else np.ones(1)
        first_w=lambda s:np.array([1.,0.]) if len(game.actions(s,WHITE))==2 else np.ones(1)
        values=evaluate_markov(game,root,first,first_w)
        self.assertEqual(values[root].black_q[0],1.)

    def test_coverage_balances_fresh_rounds_and_8_1_1_blocks(self):
        coverage=BalancedCoverage(seed=3);rounds=Counter();groups=Counter()
        for block in range(len(coverage.groups)):
            episodes=[coverage.next() for _ in range(10)]
            self.assertEqual(Counter(e.forced_root for e in episodes),Counter({None:8,0:1,1:1}))
            self.assertEqual(len({(e.horizon,e.controller) for e in episodes}),1)
            for e in episodes:
                rounds[e.horizon,e.controller]+=e.horizon;groups[e.horizon,e.controller]+=1
        self.assertEqual(set(rounds.values()),{2000})
        self.assertEqual(len(rounds),6);self.assertEqual(coverage.completed,140)
        state=coverage.state_dict();copy=BalancedCoverage(seed=99);copy.load_state_dict(state)
        self.assertEqual([coverage.next() for _ in range(20)],[copy.next() for _ in range(20)])

    def test_coverage_mid_block_exact_resume_and_corruption_rejection(self):
        coverage=BalancedCoverage(seed=3)
        for _ in range(17):coverage.next()
        state=deepcopy(coverage.state_dict());restored=BalancedCoverage(seed=9);restored.load_state_dict(state)
        self.assertEqual([coverage.next() for _ in range(170)],[restored.next() for _ in range(170)])
        bad=deepcopy(state);bad['episodes'].append(bad['episodes'][0])
        with self.assertRaises(ValueError):restored.load_state_dict(bad)

    def test_fail_closed_budgets_invalid_policies_and_forged_states(self):
        game=BranchingGame();root=game.root()
        with self.assertRaises(SolveBudgetExceeded):PublicGameSolver(game,max_nodes=1).solve(root)
        with self.assertRaises(SolveBudgetExceeded):evaluate_markov(game,root,lambda s:[.5,.5],lambda s:[1.],max_nodes=1)
        with self.assertRaises(ValueError):markov_response(game,root,lambda s:[np.nan],BLACK)
        with self.assertRaises(ValueError):DelayedState(3,BLACK,1,-1)
        with self.assertRaises(ValueError):BranchState(BLACK,2,0,-1,0)
        with self.assertRaises(ValueError):game.transition(root,0,1)
        with self.assertRaises(ValueError):game.utility(root)

if __name__=='__main__':unittest.main()

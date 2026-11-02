from pathlib import Path
import json
import unittest
import numpy as np
from graygo.engine import BLACK,WHITE,State
from graygo.solving import FiniteSolver,SolveBudgetExceeded
from graygo.strategies import GrayGoGame,episode_best_response,evaluate_episode_mixtures,episode_value_gaps,condition_episode_mixture


class RepeatBitGame:
    @staticmethod
    def actions(state,player):return (0,1)
    @staticmethod
    def terminal(state):return state[0]==2
    @staticmethod
    def utility(state):return 1. if state[1]==state[2] else -1.
    @staticmethod
    def transition(state,b,w):return (state[0]+1,b,w)


def constant(bit):
    return lambda state,history:np.array([1.,0.]) if bit==0 else np.array([0.,1.])


class StrategyTests(unittest.TestCase):
    def test_episode_identity_is_learned_from_past_actions(self):
        game=RepeatBitGame();root=(0,None,None)
        mixture=episode_best_response(game,root,[constant(0),constant(1)],[.5,.5],BLACK)
        restarted=episode_best_response(game,root,[lambda s,h:[.5,.5]],[1.],BLACK)
        # At round two an actual episode component is known from the first reveal;
        # a fresh per-state mixture remains unpredictable.
        self.assertEqual(mixture.black_value,1.)
        self.assertEqual(restarted.black_value,0.)

    def test_independent_component_draws_are_not_correlated(self):
        game=RepeatBitGame();root=(0,None,None);components=[constant(0),constant(1)]
        result=evaluate_episode_mixtures(game,root,components,[.5,.5],components,[.5,.5])
        self.assertEqual(result.black_value,0.)
        def copy_last(state,history):
            return [1.,0.] if not history or history[-1][1]==0 else [0.,1.]
        result=evaluate_episode_mixtures(game,root,[copy_last],[1.],components,[.5,.5])
        self.assertEqual(result.black_value,1.)
        self.assertEqual(episode_best_response(game,root,components,[.5,.5],WHITE).black_value,-1.)

    def test_real_gray_go_equilibrium_and_complete_best_response(self):
        fixture=json.loads((Path(__file__).parent/'fixtures'/'mixed_subgame.json').read_text())
        state=State.from_dict(fixture['state']);solver=FiniteSolver();value=solver.solve(state)[0]
        black=lambda s,h:solver.policy(s,BLACK)
        white=lambda s,h:solver.policy(s,WHITE)
        result=evaluate_episode_mixtures(GrayGoGame(),state,[black,black],[.2,.8],[white],[1.])
        self.assertAlmostEqual(result.black_value,-.25)
        gaps=episode_value_gaps(GrayGoGame(),state,value,[black],[1.],[white,white],[.3,.7])
        self.assertLess(max(gaps['row_gap'],gaps['column_gap']),1e-7)

    def test_invalid_probabilities_and_budget_never_return_a_solution(self):
        game=RepeatBitGame();root=(0,None,None)
        with self.assertRaises(ValueError):episode_best_response(game,root,[constant(0)],[.2],BLACK)
        with self.assertRaises(ValueError):episode_best_response(game,root,[lambda s,h:[2.,-1.]],[1.],BLACK)
        with self.assertRaises(SolveBudgetExceeded):episode_best_response(game,root,[constant(0)],[1.],BLACK,max_nodes=1)
        def unreachable(s,h):raise AssertionError('zero-prior component evaluated')
        result=episode_best_response(game,root,[constant(0),unreachable],[1.,0.],BLACK)
        self.assertEqual(result.black_value,1.)


    def test_prefix_conditions_component_belief_without_restarting_it(self):
        game=RepeatBitGame();components=[constant(0),constant(1)]
        state,belief,history=condition_episode_mixture(game,(0,None,None),components,[.5,.5],WHITE,[(0,1)])
        np.testing.assert_array_equal(belief,[0.,1.])
        self.assertEqual(history,((0,1),))
        result=episode_best_response(game,state,components,belief,BLACK,history=history)
        self.assertEqual(result.black_value,1.)
        with self.assertRaises(ValueError):condition_episode_mixture(game,(0,None,None),[constant(0)],[1.],WHITE,[(0,1)])

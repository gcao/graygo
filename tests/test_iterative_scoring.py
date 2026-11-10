import json
from pathlib import Path
import unittest
import numpy as np
from graygo.engine import Rules,State,score,territory_labels,resolve,transform,neighbors
from graygo.rule_state import RuleStateCodec,RuleStateSpec
from graygo.strategies import GrayGoGame


def cells(key):return tuple('+BWG'.index(c) for c in key.replace('/',''))


class IterativeScoringTests(unittest.TestCase):
    def test_shared_examples_and_physical_board_is_unchanged(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/iterative-scoring.json').read_text())
        for case in fixture['cases']:
            with self.subTest(case=case['name']):
                original=cells(case['board_key']);n=len(case['board_key'].split('/'))
                self.assertEqual(territory_labels(original,n),cells(case['labels_key']))
                result=score(original,n,scoring='iterative')
                self.assertEqual(float(result.black),case['black']);self.assertEqual(float(result.white),case['white'])
                self.assertEqual(float(score(original,n).black),case['legacy_black'])
                self.assertEqual(original,cells(case['board_key']))

    def test_marking_is_symmetric_idempotent_and_conserves_area(self):
        rng=np.random.default_rng(1110);n=5
        for _ in range(100):
            board=tuple(map(int,rng.integers(0,4,n*n)))
            labels=territory_labels(board,n)
            self.assertEqual(territory_labels(labels,n),labels)
            self.assertTrue(all(old==0 or old==new for old,new in zip(board,labels)))
            swapped=tuple(3-c if c in (1,2) else c for c in board)
            expected=tuple(3-c if c in (1,2) else c for c in labels)
            self.assertEqual(territory_labels(swapped,n),expected)
            # A toroidal shift changes coordinates, not marking order or score.
            shift=lambda b:tuple(b[((r-1)%n)*n+(c-2)%n] for r in range(n) for c in range(n))
            self.assertEqual(territory_labels(shift(board),n),shift(labels))
            self.assertEqual(score(board,n,scoring='iterative'),score(shift(board),n,scoring='iterative'))
            value=score(board,n,scoring='iterative');opposite=score(swapped,n,scoring='iterative')
            self.assertEqual(value.black_units+value.white_units,4*n*n)
            self.assertEqual(value.black_units,opposite.white_units)

    def test_v3_serialization_and_replay_are_separate_from_v2(self):
        rules=Rules(5,2,ko='full-history',scoring='iterative')
        self.assertEqual(rules.rule_version,'graygo/3')
        state=State(rules);state=resolve(state,0,rules.pass_action).state
        self.assertEqual(State.from_dict(state.to_dict()),state)
        spec=RuleStateSpec(rules,ko_hints=True)
        self.assertEqual(RuleStateSpec.from_dict(spec.to_dict()),spec)
        packet=RuleStateCodec(rules).pack(state,1)
        with self.assertRaises(ValueError):RuleStateCodec(Rules(5,2,ko='full-history')).unpack(packet)
        self.assertNotIn('scoring',Rules(ko='full-history').to_dict())
        self.assertEqual(Rules().rule_version,'graygo/1')
        with self.assertRaises(ValueError):Rules(scoring='iterative')
        with self.assertRaises(ValueError):score((0,)*9,3,scoring='unsupported')

    def test_terminal_rewards_follow_the_selected_scoring_rule(self):
        board=cells('WBBW+/W+++W/+WG++/++++W/+BBBB')
        state=State(Rules(5,1,ko='full-history',scoring='iterative'),board)
        old=score(board,5);new=score(board,5,scoring='iterative')
        self.assertEqual(old.black_units,50)
        self.assertEqual(new.black_units,42)
        outcome=resolve(state,25,25)
        self.assertTrue(outcome.state.done)
        self.assertEqual(outcome.reward_black,new.utility_black)
        self.assertEqual(GrayGoGame.utility(outcome.state),new.utility_black)
        self.assertNotEqual(outcome.reward_black,old.utility_black)


    def test_scalar_and_batched_evaluation_use_v3_terminal_scores(self):
        from graygo.evaluation import evaluate_pair
        from graygo.evaluation_batch import PairRequest,evaluate_pairs_batched
        class PassPolicy:
            sha256='0'*64
            def begin_episode(self,seed):return self
            def action(self,state,player):return state.rules.pass_action
        policy=PassPolicy()
        root=State(Rules(5,1,ko='full-history',scoring='iterative'),cells('WBBW+/W+++W/+WG++/++++W/+BBBB'))
        scalar=evaluate_pair(policy,policy,root,unit_seed=10)
        batch=evaluate_pairs_batched([PairRequest(policy,policy,root,10)])[0]
        self.assertEqual(batch,scalar)
        for game in batch['subject']:
            self.assertEqual(game['black_score_units'],42)
            self.assertEqual(game['white_score_units'],58)
            self.assertEqual(game['black_utility'],-1)

    def test_v3_territory_metrics_count_actual_empty_points_and_virtual_marks(self):
        from graygo.board_collection import BoardCollector
        collector=BoardCollector(Rules(5,2,ko='full-history',scoring='iterative'),2,1)
        board=cells('B++++/+++++/++G++/+++++/+++++')
        collector._territory(board)
        self.assertEqual(collector.metrics['terminal_empty_points'],23)
        self.assertEqual(collector.metrics['terminal_marked_points'],19)
        self.assertEqual(collector.metrics['fractional_bias_empty_points'],4)

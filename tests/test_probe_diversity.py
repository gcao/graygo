from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from graygo.engine import BLACK,WHITE,GRAY,Rules,State,groups,resolve,transform,transform_action
from graygo.probe_benchmark import ProbeBenchmark,ProbeFamily,DIVERSE_GENERATOR_VERSION,strategy_family_id
from graygo.probe_certification import CertifiedRoot,certify_one_round,qualify_pair
from graygo.probe_diversity import multigroup_proposal,construct_diverse_benchmark
from graygo.probe_setup import reachable_multigroup_pair

FIXTURE=Path(__file__).parent/'fixtures/diverse-prototypes-v1.json'


class ProbeDiversityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.prototypes=json.loads(FIXTURE.read_text())['prototypes']

    def test_all_eleven_independent_prototypes_have_real_default_game_traces_and_exact_ranges(self):
        ids=set();mixed=gray=0
        for data in self.prototypes:
            roots=reachable_multigroup_pair(data['before']['board'],data['after']['board'])
            certified=[]
            for (root,trace),name in zip(roots,('before','after')):
                self.assertEqual(root.rules,Rules());self.assertEqual(root.rounds_played,199)
                self.assertFalse(root.done);self.assertFalse(root.forbidden_black|root.forbidden_white)
                self.assertEqual(root.pass_streak,0);self.assertEqual(len(trace),199)
                # Replay directly through the resolver, without the setup
                # builder's replay cache or any assigned synthetic metadata.
                state=State()
                for black,white in trace:state=resolve(state,black,white).state
                self.assertEqual(state,root)
                proof=certify_one_round(root)
                self.assertEqual([[str(v) for v in row] for row in proof.certificate.payoffs],data[name+'_matrix'])
                self.assertEqual([str(proof.certificate.lower),str(proof.certificate.upper)],data[name+'_range'])
                with patch('graygo.probe_certification.solve_matrix',side_effect=AssertionError('proof verification must be exact')):
                    self.assertEqual(CertifiedRoot.from_dict(proof.to_dict()),proof)
                certified.append(proof)
                gray+=GRAY in root.board
                mixed+=any(sum(p>0 for p in distribution)>1 for distribution in proof.certificate.equilibrium)
            a,b=certified;qualification=qualify_pair(a,b)
            self.assertTrue(qualification['eligible']);self.assertGreaterEqual(qualification['separation'],Fraction(1,10))
            identity=strategy_family_id(a,b);self.assertEqual(identity,data['strategy_family_id']);ids.add(identity)
        self.assertEqual(len(ids),11);self.assertGreater(mixed,0);self.assertGreater(gray,0)

    def test_padding_and_final_gray_stone_are_actual_resolved_moves(self):
        for faction in (BLACK,WHITE,GRAY):
            with self.subTest(faction=faction):
                dense=[faction]*81;dense[0]=0;sparse=dense.copy();sparse[1]=0
                for root,trace in reachable_multigroup_pair(dense,sparse):
                    state=State()
                    for black,white in trace[:196]:state=resolve(state,black,white).state
                    board=state.board
                    after_suicide=resolve(state,*trace[196]).state
                    self.assertEqual(after_suicide.board,board)
                    self.assertEqual(len(after_suicide.forbidden_black|after_suicide.forbidden_white),1)
                    after_pass=resolve(after_suicide,*trace[197]).state
                    if faction==GRAY:
                        self.assertNotEqual(after_pass.board,board)
                        self.assertFalse(after_pass.forbidden_black|after_pass.forbidden_white)
                    else:
                        self.assertEqual(trace[197],(81,81))
                        self.assertEqual(after_pass.board,board);self.assertEqual(after_pass.pass_streak,1)
                    self.assertEqual(resolve(after_pass,*trace[198]).state,root)
                    if faction==GRAY:
                        self.assertEqual(trace[-1][0],trace[-1][1])
                        self.assertTrue(all(b==w or b==81 or w==81 for b,w in trace))

    def test_unstable_or_sparse_boards_cannot_get_fake_round_counters(self):
        with self.assertRaisesRegex(ValueError,'zero-liberty|stable groups'):
            reachable_multigroup_pair([BLACK]*81,[BLACK]*81)
        board=[0]*81;board[0]=BLACK
        with self.assertRaisesRegex(ValueError,'protected ring'):
            reachable_multigroup_pair(board,board)
        with self.assertRaises(ValueError):reachable_multigroup_pair([0]*80,[0]*81)

    def test_proposals_are_seeded_single_cell_changes_with_stable_groups(self):
        first=np.random.default_rng(127);second=np.random.default_rng(127);operations=set();valid=0
        for _ in range(100):
            a=multigroup_proposal(first);b=multigroup_proposal(second);self.assertEqual(a,b)
            proposal,reason,operation=a;operations.add(operation)
            if proposal is None:
                self.assertEqual(reason,'unstable_recolor');continue
            valid+=1;before,after,point=proposal
            self.assertEqual(before.rules,Rules(9,1));self.assertEqual(after.rules,before.rules)
            self.assertEqual([p for p,(x,y) in enumerate(zip(before.board,after.board)) if x!=y],[point])
            self.assertTrue(1<=before.board.count(0)<=4);self.assertTrue(1<=after.board.count(0)<=5)
            self.assertTrue(all(g.liberties for state in (before,after) for g in groups(state.board,9)))
        self.assertEqual(operations,{'remove','recolor'});self.assertGreater(valid,20)

    def test_seed_reproduction_partial_coverage_and_fixed_alternating_directions(self):
        first,a=construct_diverse_benchmark(974214,attempt_cap=200,time_cap_seconds=30)
        second,b=construct_diverse_benchmark(974214,attempt_cap=200,time_cap_seconds=30)
        self.assertIsNone(first);self.assertIsNone(second)
        self.assertEqual(a['status'],'attempt_limited');self.assertEqual(a['attempts'],200)
        self.assertEqual(a['partial_families'],b['partial_families']);self.assertEqual(a['counts'],b['counts'])
        self.assertEqual(a['distinct_strategic_families'],2);self.assertFalse(a['benchmark_complete'])
        families=[ProbeFamily.from_dict(row) for row in a['partial_families']]
        self.assertEqual([f.direction for f in families],[1,-1])
        self.assertEqual(len({f.strategy_id for f in families}),len(families))
        self.assertFalse(a['candidate_policy_used'])

    def test_geometric_copies_cannot_fill_independent_family_slots(self):
        data=self.prototypes[1];base=[State.from_dict(data[name]) for name in ('before','after')]
        proposals=[]
        for i in range(8):
            params={'rotation':i%4,'reflect':i>=4,'shift':(i,2*i)}
            roots=[transform(root,**params) for root in base]
            point=transform_action(data['changed_point'],base[0].rules,**params)
            proposals.append(((roots[0],roots[1],point),None,'remove'))
        with patch('graygo.probe_diversity.multigroup_proposal',side_effect=proposals):
            benchmark,report=construct_diverse_benchmark(5,attempt_cap=8,time_cap_seconds=30)
        self.assertIsNone(benchmark);self.assertEqual(report['distinct_strategic_families'],1)
        self.assertEqual(report['counts']['repeated_strategy'],7);self.assertFalse(report['benchmark_complete'])

    def test_new_generator_metadata_cannot_promote_the_historical_one_class_catalog(self):
        historical=ProbeBenchmark.from_dict(json.loads((FIXTURE.parent/'collision-benchmark-v1.json').read_text()))
        modern=replace(historical,generator=DIVERSE_GENERATOR_VERSION)
        self.assertEqual(ProbeBenchmark.from_dict(modern.to_dict()).to_dict(),modern.to_dict())
        self.assertEqual(modern.strategic_family_count,1);self.assertFalse(modern.family_coverage_complete)
        with self.assertRaisesRegex(ValueError,'unknown collision probe generator'):
            replace(historical,generator='unregistered')

    def test_interrupt_and_budget_checks_cannot_produce_partial_success(self):
        benchmark,report=construct_diverse_benchmark(5,attempt_cap=10,time_cap_seconds=30,stop_requested=lambda:True)
        self.assertIsNone(benchmark);self.assertEqual(report['status'],'interrupted');self.assertEqual(report['attempts'],0)
        with patch('graygo.probe_diversity.time.monotonic',side_effect=[0.,2.,2.]):
            benchmark,report=construct_diverse_benchmark(5,attempt_cap=10,time_cap_seconds=1)
        self.assertIsNone(benchmark);self.assertEqual(report['status'],'budget_limited');self.assertFalse(report['benchmark_complete'])
        for time_cap in (0,-1,True,float('inf')):
            with self.assertRaises(ValueError):construct_diverse_benchmark(5,time_cap_seconds=time_cap)


if __name__=='__main__':unittest.main()

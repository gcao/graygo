from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from graygo.engine import Rules,State,groups,transform
from graygo.probe_benchmark import ProbeBenchmark,ProbeFamily,MULTICELL_GENERATOR_VERSION
from graygo.probe_diversity import multicell_proposal,construct_multicell_benchmark


class MulticellProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark,cls.prefix=construct_multicell_benchmark(974215,attempt_cap=500,time_cap_seconds=30)
        cls.families=tuple(ProbeFamily.from_dict(p) for p in cls.prefix['partial_families'])

    def test_exact_two_cells_stable_groups_and_full_action_budget(self):
        first=np.random.default_rng(923);second=np.random.default_rng(923);valid=0;operations=set()
        for _ in range(150):
            a=multicell_proposal(first);self.assertEqual(a,multicell_proposal(second))
            candidate,reason,operation=a
            if candidate is None:continue
            before,after,point=candidate;valid+=1;operations.add(operation)
            self.assertIsNone(point)
            self.assertEqual(sum(x!=y for x,y in zip(before.board,after.board)),2)
            self.assertLessEqual(max(len(before.legal_actions(1)),len(after.legal_actions(2))),6)
            self.assertTrue(all(group.liberties for root in (before,after) for group in groups(root.board,9)))
        self.assertGreater(valid,20);self.assertIn('double_remove',operations);self.assertIn('remove_recolor',operations)

    def test_second_removal_cannot_exceed_five_empty_points(self):
        board=[1]*81
        for point in (0,2,4,6):board[point]=0
        middle=board.copy();middle[10]=0
        candidate=(State(Rules(9,1),board),State(Rules(9,1),middle),10)
        valid=0
        with patch('graygo.probe_diversity.multigroup_proposal',return_value=(candidate,None,'remove')):
            for seed in range(30):
                proposal,reason,operation=multicell_proposal(np.random.default_rng(seed))
                self.assertEqual(operation,'remove_recolor')
                if proposal is not None:
                    valid+=1;self.assertEqual(proposal[1].board.count(0),5)
        self.assertGreater(valid,0)

    def test_fixed_prefix_reproduces_all_proofs_and_explicit_changed_points(self):
        benchmark,report=construct_multicell_benchmark(974215,attempt_cap=500,time_cap_seconds=30)
        self.assertIsNone(benchmark);self.assertIsNone(self.benchmark)
        self.assertEqual(report['partial_families'],self.prefix['partial_families'])
        self.assertEqual(report['controlled_changes'],self.prefix['controlled_changes'])
        self.assertEqual(report['counts'],self.prefix['counts'])
        self.assertEqual([f.attempt for f in self.families],[18,54,63,138,354])
        self.assertEqual([f.direction for f in self.families],[1,-1,1,-1,1])
        self.assertEqual(len({f.strategy_id for f in self.families}),5)
        self.assertFalse(report['benchmark_complete'])
        self.assertEqual(report['generator'],MULTICELL_GENERATOR_VERSION)
        for family in self.families:
            self.assertIsNone(family.changed_point);self.assertEqual(len(family.changed_points),2)
            self.assertEqual(report['controlled_changes'][family.identity],list(family.changed_points))
            self.assertEqual(ProbeFamily.from_dict(family.to_dict()),family)

    def test_geometric_duplicates_do_not_manufacture_more_families(self):
        family=self.families[0];before=State(Rules(9,1),family.before.state.board);after=State(Rules(9,1),family.after.state.board)
        changes=family.changed_points;removals=sum(0 in (before.board[p],after.board[p]) for p in changes)
        operation=('double_recolor','remove_recolor','double_remove')[removals]
        proposals=[]
        for i in range(8):
            params={'rotation':i%4,'reflect':i>=4,'shift':(i,2*i)}
            proposals.append(((transform(before,**params),transform(after,**params),None),None,operation))
        with patch('graygo.probe_diversity.multicell_proposal',side_effect=proposals):
            benchmark,report=construct_multicell_benchmark(1,attempt_cap=8,time_cap_seconds=30)
        self.assertIsNone(benchmark);self.assertEqual(report['distinct_strategic_families'],1)
        self.assertEqual(report['counts']['repeated_strategy'],7);self.assertFalse(report['benchmark_complete'])

    def test_version_label_cannot_relabel_single_cell_cases_or_relax_coverage(self):
        fixture=Path(__file__).parent/'fixtures/collision-benchmark-v1.json'
        original=ProbeBenchmark.from_dict(json.loads(fixture.read_text()))
        with self.assertRaisesRegex(ValueError,'exactly two controlled cells'):
            replace(original,generator=MULTICELL_GENERATOR_VERSION)
        with self.assertRaisesRegex(ValueError,'complete 32-case'):
            ProbeBenchmark(974215,self.families,MULTICELL_GENERATOR_VERSION)

    def test_new_prefix_adds_complete_game_classes_beyond_the_audited_one_cell_catalog(self):
        path=Path(__file__).resolve().parents[1]/'docs/verification/probe-diversity-v1-construction/report.json'
        previous={f['strategy_family_id'] for f in json.loads(path.read_text())['partial_families']}
        self.assertEqual(len(previous),24)
        self.assertTrue({f.strategy_id for f in self.families}-previous)

    def test_interrupted_and_time_limited_work_stays_incomplete(self):
        benchmark,report=construct_multicell_benchmark(1,attempt_cap=10,time_cap_seconds=30,stop_requested=lambda:True)
        self.assertIsNone(benchmark);self.assertEqual(report['attempts'],0);self.assertEqual(report['status'],'interrupted')
        self.assertEqual(report['controlled_changes'],{});self.assertFalse(report['benchmark_complete'])
        with patch('graygo.probe_diversity.time.monotonic',side_effect=[0.,2.,2.]):
            benchmark,report=construct_multicell_benchmark(1,attempt_cap=10,time_cap_seconds=1)
        self.assertIsNone(benchmark);self.assertEqual(report['status'],'budget_limited');self.assertFalse(report['benchmark_complete'])


if __name__=='__main__':unittest.main()

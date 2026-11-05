from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from graygo.board_execution import BoardExecutor,BoardExecutionPlan
from graygo.board_journal import BoardJournalPlan
from graygo.board_runtime import BoardRuntimeConfig
from graygo.confidence import canonical_sha
from graygo.engine import Rules
from tools.champion_gate import GatePlan,IndependentGate
from tools.champion_loop import ChampionLoop,LoopPlan
from tools.continuous_training import create,run,next_endpoint,stage_alpha
from tools.learning_loop import worker
from test_learning_loop import exact,normalized


def config():
    return BoardRuntimeConfig(rules=Rules(3,5),envs=2,capacity=256,batch=4,learning_start=4,
        target_every=8,width=8,blocks=1,norm_groups=2)


def parent_run(base):
    parent=base/'parent';publication=base/'champion.json'
    with BoardExecutor(parent,config(),BoardJournalPlan('practice',8,(0,8)),BoardExecutionPlan(120),reserve_bytes=0) as executor:
        with ChampionLoop(executor,LoopPlan((8,)),IndependentGate(GatePlan(1),reserve_bytes=0),publish_manifest=publication) as loop:loop.run()
    return parent,publication


class Clock:
    def __init__(self):self.now=0.
    def monotonic(self):return self.now
    def wall(self):return 1800000000.+self.now


class ContinuousTrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_unlimited_executor_continues_after_an_hour_and_resumes_finite_json_clocks(self):
        with tempfile.TemporaryDirectory() as directory:
            clock=Clock();journal=BoardJournalPlan('loop',32,(0,16,32));plan=BoardExecutionPlan(None,checkpoint_every=16)
            kwargs=dict(reserve_bytes=0,monotonic=clock.monotonic,wall_time=clock.wall)
            with BoardExecutor(directory,config(),journal,plan,**kwargs) as executor:
                original=executor.runtime.advance
                def advance(*args,**values):
                    result=original(*args,**values);clock.now+=5000;return result
                executor.runtime.advance=advance
                result=executor.run(16)
                self.assertEqual(result['records'],16);self.assertNotEqual(result['status'],'budget_limited')
                self.assertGreater(result['clock']['observed_seconds'],3600)
            with BoardExecutor(directory,config(),journal,plan,resume=True,**kwargs) as executor:
                self.assertEqual(executor.run()['records'],32)
            body=json.loads((Path(directory)/'execution.json').read_text())['body']
            self.assertIsNone(body['contract']['execution']['time_ceiling_seconds'])
            self.assertEqual(body['progress']['records'],32)
            json.dumps(body,allow_nan=False)

    def test_unlimited_scope_does_not_relax_diagnostic_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'explicit experimental loop'):
                BoardExecutor(directory,config(),BoardJournalPlan('diagnostic',32,(0,32)),BoardExecutionPlan(None))

    def test_unlimited_unclean_recovery_keeps_clock_accounting_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            clock=Clock();journal=BoardJournalPlan('loop',32,(0,16,32));plan=BoardExecutionPlan(None)
            kwargs=dict(reserve_bytes=0,monotonic=clock.monotonic,wall_time=clock.wall)
            with BoardExecutor(directory,config(),journal,plan,**kwargs) as executor:executor.run(16)
            path=Path(directory)/'execution.json';wrapper=json.loads(path.read_text())
            wrapper['body']['clock']['inflight_since']=clock.wall();wrapper['sha256']=canonical_sha(wrapper['body'])
            path.write_text(json.dumps(wrapper));clock.now+=7200
            with BoardExecutor(directory,config(),journal,plan,resume=True,**kwargs) as executor:
                self.assertEqual(executor.run()['records'],32)
                self.assertEqual(executor.clock['unclean_recoveries'],1)
                self.assertEqual(executor.clock['recovery_charge_seconds'],7200)
                json.dumps(executor.report('complete'),allow_nan=False)

    def test_unlimited_evaluation_allowance_is_finite_and_earned(self):
        with tempfile.TemporaryDirectory() as directory:
            clock=Clock()
            with BoardExecutor(directory,config(),BoardJournalPlan('loop',32,(0,32)),BoardExecutionPlan(None),
                reserve_bytes=0,monotonic=clock.monotonic,wall_time=clock.wall) as executor:
                with ChampionLoop(executor,LoopPlan((32,)),IndependentGate(GatePlan(1),reserve_bytes=0)) as loop:
                    self.assertEqual(loop._evaluation_remaining(),0)
                    clock.now=1000;self.assertEqual(loop._evaluation_remaining(),100)
                    loop.state['evaluation_seconds']=80;self.assertEqual(loop._evaluation_remaining(),20)
                    clock.now=2000;self.assertEqual(loop._evaluation_remaining(),120)

    def test_two_real_segments_preserve_the_same_learner_as_uninterrupted_training(self):
        checkout=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);parent,publication=parent_run(base);campaign=base/'campaign'
            create(campaign,parent,checkout,publication,project=base,segment_records=16,block=8,reserve_bytes=0)
            state=run(campaign,max_segments=2,reserve_bytes=0)
            self.assertEqual(state['completed_segments'],2);self.assertIsNone(state['current_stage'])
            child=Path(state['parent_run'])
            data=torch.load(child/'resume/000000000032/runtime.pt',map_location='cpu',weights_only=True)
            self.assertEqual(data['backend']['counters']['fresh_records'],32)
            for index,stage in enumerate(sorted((campaign/'stages').iterdir())):
                plan=json.loads((stage/'plan.json').read_text())['body']
                self.assertIsNone(plan['remaining_seconds'])
                self.assertEqual(plan['gate_alpha'],stage_alpha(index))
            with BoardExecutor(base/'plain',config(),BoardJournalPlan('loop',32,(0,8,16,24,32)),
                               BoardExecutionPlan(None),reserve_bytes=0) as executor:
                executor.run()
                exact(normalized(executor.runtime.state_dict()),normalized(data))
            receipt=json.loads((base/'runs/latest-training-loop.json').read_text())
            self.assertTrue(receipt['unlimited']);self.assertIsNone(receipt['deadline_utc'])

    def test_interrupted_segment_restarts_from_its_existing_checkpoint(self):
        checkout=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);parent,publication=parent_run(base);campaign=base/'campaign'
            create(campaign,parent,checkout,publication,project=base,segment_records=16,block=8,reserve_bytes=0)
            def interrupted(stage,**kwargs):
                kwargs['stop_state'].append(True)
                return worker(stage,**kwargs)
            with patch('tools.continuous_training.worker',side_effect=interrupted):
                first=run(campaign,max_segments=1,reserve_bytes=0)
            self.assertEqual(first['status'],'interrupted')
            state=run(campaign,max_segments=1,reserve_bytes=0)
            self.assertEqual(state['completed_segments'],1)
            status=json.loads((campaign/'stages/000000/status.json').read_text())
            self.assertEqual(status['total_records'],16)

    def test_rollovers_have_no_record_ceiling_and_a_summable_promotion_allowance(self):
        self.assertEqual(next_endpoint(20000000,20000000),40000000)
        self.assertEqual(next_endpoint(80000000,20000000),100000000)
        self.assertLess(sum(stage_alpha(i) for i in range(10000)),.05)
        self.assertAlmostEqual(sum(stage_alpha(i) for i in range(10000)),.05*(1-1/10001))
        with self.assertRaises(ValueError):GatePlan(1,alpha=.1)


if __name__=='__main__':unittest.main()

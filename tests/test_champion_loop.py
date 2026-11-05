from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import torch
from graygo.board_execution import BoardExecutor,BoardExecutionPlan
from graygo.board_journal import BoardJournalPlan
from graygo.board_runtime import BoardRuntimeConfig
from graygo.confidence import canonical_sha
from graygo.engine import Rules
from graygo.evaluation import BaselinePolicy,FrozenNeuralPolicy
from tools.champion_gate import GatePlan,IndependentGate
from tools.champion_loop import ChampionLoop,LoopPlan,parse_args
from tools.champion_screen import IndependentScreen,ScreenPlan


class FixtureGate:
    def __init__(self,promote=False):
        self.plan=GatePlan(2,(4,));self.sha256=canonical_sha({'fixture':promote});self.promote=promote
        self.protected_authorizer=None;self.calls=[]
    def evaluate(self,candidate,champion,pool,root,**kwargs):
        self.calls.append((candidate.sha256,champion.sha256,tuple(pool)))
        return {'status':'complete','numeric_pass':self.promote,'promotion_permitted':False,
                'experimental_promotion_permitted':self.promote,'fixture_only':True}


def config():
    return BoardRuntimeConfig(rules=Rules(3,5),envs=4,seed=3,batch=4,learning_start=4,capacity=256,
        target_every=8,width=8,blocks=1,norm_groups=2,reservoir_open=0,handoff_end=8)


def execute(root,resume=False):
    return BoardExecutor(root,config(),BoardJournalPlan('diagnostic',256,(0,8,128,256)),
        BoardExecutionPlan(120,checkpoint_every=128,checkpoint_headroom_bytes=65536),resume=resume,reserve_bytes=0)


def normalized(value):
    if isinstance(value,dict):return {k:normalized(v) for k,v in value.items() if k!='timings'}
    return value


def exact(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


class ChampionGateTests(unittest.TestCase):
    def test_obviously_losing_first_look_stops_without_a_larger_match(self):
        plan=GatePlan(4,(500,2000))
        result=plan.numeric(0,-np.ones(250),{'random':-np.ones(250)},{'random':np.ones(250)},500)
        self.assertTrue(result['stop_for_futility']);self.assertFalse(result['numeric_pass'])

    def test_old_48_percent_and_all_draw_promotions_are_rejected(self):
        plan=GatePlan(2,(2000,));n=1000
        for head in (np.zeros(n),np.full(n,-.03)):
            result=plan.numeric(0,head,{'random':np.ones(n)},{'random':np.zeros(n)},2000)
            self.assertFalse(result['numeric_pass'])
            self.assertLess(result['head_win_score']['lower'],.5)

    def test_credible_improvement_can_pass_but_historical_regression_blocks_it(self):
        plan=GatePlan(2,(2000,));n=1000
        result=plan.numeric(0,np.ones(n),{'random':np.ones(n),'historical':np.ones(n)},
                            {'random':np.zeros(n),'historical':np.zeros(n)},2000)
        self.assertTrue(result['numeric_pass']);self.assertEqual(result['alpha_per_round'],.025)
        result=plan.numeric(0,np.ones(n),{'random':np.ones(n),'historical':np.full(n,-.2)},
                            {'random':np.zeros(n),'historical':np.zeros(n)},2000)
        self.assertFalse(result['criteria']['no_historical_regression'])
        self.assertFalse(result['numeric_pass'])


class ChampionLoopTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_evaluation_quota_survives_resume_and_does_not_block_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            gate=FixtureGate()
            plan=LoopPlan((128,256),continue_after_evaluation_budget=True)
            with execute(directory) as learner:
                learner.journal.diagnostic_opening('pass')
                original=gate.evaluate
                def slow_gate(*args,**kwargs):
                    decision=original(*args,**kwargs)
                    learner.base_observed+=15. # Simulated completed atomic evaluation work.
                    return decision
                gate.evaluate=slow_gate
                with ChampionLoop(learner,plan,gate) as loop:
                    loop.run(until_round=1)
                    self.assertGreaterEqual(loop.state['evaluation_seconds'],15.)
            gate=FixtureGate()
            with execute(directory,True) as learner:
                with ChampionLoop(learner,plan,gate,resume=True) as loop:
                    result=loop.run()
                    self.assertEqual(result['learner_records'],256)
                    self.assertEqual(gate.calls,[])
                    self.assertEqual(loop.state['rounds'][-1]['decision']['reason'],'evaluation_budget_exhausted')
                    self.assertFalse(any(r['promoted'] for r in loop.state['rounds']))

    def test_short_gate_budget_can_reject_and_continue_training(self):
        class ShortGate(FixtureGate):
            def evaluate(self,*args,**kwargs):
                super().evaluate(*args,**kwargs)
                return {'status':'budget_limited','numeric_pass':False,'promotion_permitted':False}
        with tempfile.TemporaryDirectory() as directory:
            gate=ShortGate()
            with execute(directory) as learner:
                learner.journal.diagnostic_opening('pass')
                with ChampionLoop(learner,LoopPlan((128,256),continue_after_evaluation_budget=True),gate) as loop:
                    self.assertEqual(loop.run()['learner_records'],256)
                    self.assertEqual(len(gate.calls),2)
                    self.assertTrue(all(r['decision']['reason']=='evaluation_budget_exhausted' for r in loop.state['rounds']))

    def test_inflight_evaluation_is_charged_on_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=LoopPlan((128,256))
            with execute(directory) as learner:
                learner.journal.diagnostic_opening('pass')
                with ChampionLoop(learner,plan,FixtureGate()) as loop:
                    loop.run(until_round=1)
                    loop.state['evaluation_inflight']=learner._spent();loop._save()
                    learner.base_observed+=15.
            gate=FixtureGate()
            with execute(directory,True) as learner:
                with ChampionLoop(learner,plan,gate,resume=True) as loop:
                    self.assertIsNone(loop.state['evaluation_inflight'])
                    self.assertGreaterEqual(loop.state['evaluation_seconds'],15.)
                    loop.run();self.assertEqual(gate.calls,[])

    def test_real_q_screen_works_before_average_initialization_without_mutating_training(self):
        with tempfile.TemporaryDirectory() as directory:
            with execute(directory) as learner:
                expected=deepcopy(learner.runtime.state_dict())
                screen=IndependentScreen(ScreenPlan(4,30),reserve_bytes=0)
                result=screen.evaluate(learner.root/'weights/000000000000',average_ready=False,
                    root=learner.root/'screen',time_cap_seconds=30)
                self.assertEqual(result['status'],'complete')
                self.assertEqual(result['heads']['average']['status'],'uninitialized')
                self.assertFalse(result['average_eligible']);self.assertFalse(result['promotion_permitted'])
                for report in result['heads']['greedy_q']['opponents'].values():self.assertEqual(report['games'],4)
                exact(normalized(expected),normalized(learner.runtime.state_dict()))

    def test_cli_preserves_primary_schedule_and_requires_explicit_early_diagnostic(self):
        base=['--run-dir','unused','--time-cap-seconds','60']
        args=parse_args(base+['--practice'])
        self.assertEqual((args.records,args.reservoir_open,args.handoff_end),(5000000,5000000,7000000))
        with self.assertRaisesRegex(ValueError,'early-opening-diagnostic'):
            parse_args(base+['--diagnostic','--reservoir-open','0','--handoff-end','1000'])
        with self.assertRaisesRegex(ValueError,'early-opening-diagnostic'):
            parse_args(base+['--practice','--early-opening-diagnostic','--reservoir-open','0','--handoff-end','1000'])
        args=parse_args(base+['--diagnostic','--early-opening-diagnostic','--reservoir-open','0','--handoff-end','1000'])
        self.assertEqual(args.reservoir_open,0)
        with self.assertRaisesRegex(ValueError,'one hour'):parse_args(['--run-dir','unused','--practice','--time-cap-seconds','7200'])
        with self.assertRaisesRegex(ValueError,'opening'):
            BoardJournalPlan('practice',5000002,(0,5000002))

    def test_rejected_candidates_preserve_the_entire_persistent_learner(self):
        with tempfile.TemporaryDirectory() as directory:
            with execute(Path(directory)/'plain') as plain:
                plain.journal.diagnostic_opening('pass');plain.run()
                expected=deepcopy(plain.runtime.state_dict())
                replay={k:v.copy() for k,v in plain.runtime.replay.arrays.items()}
                reservoir={k:v.copy() for k,v in plain.runtime.reservoir.arrays.items()}
            gate=FixtureGate()
            with execute(Path(directory)/'loop') as learner:
                learner.journal.diagnostic_opening('pass')
                with ChampionLoop(learner,LoopPlan((128,256)),gate) as loop:
                    initial=loop.state['champion']['weights_sha256']
                    result=loop.run()
                    self.assertEqual(result['rounds'],2);self.assertEqual(result['learner_records'],256)
                    self.assertEqual(loop.state['champion']['weights_sha256'],initial)
                    self.assertGreater(learner.runtime.backend.counters.supervised_updates,0)
                    self.assertTrue(all(not r['learner_reinitialized'] and not r['buffers_flushed'] for r in loop.state['rounds']))
                    self.assertFalse(any(r['promoted'] for r in loop.state['rounds']))
                exact(normalized(expected),normalized(learner.runtime.state_dict()))
                for k,v in replay.items():np.testing.assert_array_equal(v,learner.runtime.replay.arrays[k])
                for k,v in reservoir.items():np.testing.assert_array_equal(v,learner.runtime.reservoir.arrays[k])

    def test_promoted_fixture_only_changes_local_release_and_history_not_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            gate=FixtureGate(True)
            with execute(directory) as learner:
                learner.journal.diagnostic_opening('pass')
                with ChampionLoop(learner,LoopPlan((128,256)),gate) as loop:
                    result=loop.run();self.assertEqual(result['rounds'],2)
                    self.assertTrue(any(r['promoted'] for r in loop.state['rounds']))
                    self.assertTrue(loop.state['history'])
                    release=json.loads((Path(directory)/'league/champion.json').read_text())
                    self.assertTrue(release['body']['experimental'])
                    self.assertFalse(release['body']['formal_training_or_promotion_grant'])
                    self.assertTrue(Path(loop.state['champion']['bundle']).is_file())
                    self.assertTrue(any('historical-0' in labels for _,_,labels in gate.calls))

    def test_round_resume_keeps_optimizer_buffers_and_original_champion_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'resumed';gate=FixtureGate(True)
            with execute(root) as learner:
                learner.journal.diagnostic_opening('pass')
                with ChampionLoop(learner,LoopPlan((128,256)),gate) as loop:
                    self.assertEqual(loop.run(until_round=1)['learner_records'],128)
            with execute(root,True) as learner:
                with ChampionLoop(learner,LoopPlan((128,256)),gate,resume=True) as loop:
                    actual=loop.run();state=deepcopy(learner.runtime.state_dict())
                    self.assertEqual(actual['rounds'],2)
            with execute(Path(directory)/'full') as learner:
                learner.journal.diagnostic_opening('pass')
                with ChampionLoop(learner,LoopPlan((128,256)),FixtureGate(True)) as loop:loop.run()
                exact(normalized(state),normalized(learner.runtime.state_dict()))

    def test_incomplete_gate_resumes_evaluation_without_replaying_training(self):
        class DeferredGate(FixtureGate):
            def __init__(self):super().__init__();self.deferred=True
            def evaluate(self,*args,**kwargs):
                if self.deferred:
                    self.deferred=False
                    return {'status':'budget_limited','numeric_pass':False,'promotion_permitted':False}
                return super().evaluate(*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            gate=DeferredGate()
            with execute(directory) as learner:
                learner.journal.diagnostic_opening('pass')
                with ChampionLoop(learner,LoopPlan((128,256)),gate) as loop:
                    first=loop.run(until_round=1)
                    self.assertEqual(first['status'],'budget_limited');self.assertEqual(first['learner_records'],128)
                    state=deepcopy(learner.runtime.state_dict())
                    second=loop.run(until_round=1)
                    self.assertEqual(second['rounds'],1);self.assertEqual(second['learner_records'],128)
                    exact(normalized(state),normalized(learner.runtime.state_dict()))

    def test_external_publication_is_atomic_and_follows_the_committed_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            pointer=Path(directory)/'published/champion.json'
            with execute(Path(directory)/'run') as learner:
                learner.journal.diagnostic_opening('pass')
                with ChampionLoop(learner,LoopPlan((128,256)),FixtureGate(True),publish_manifest=pointer) as loop:
                    loop.run(until_round=1)
                    published=json.loads(pointer.read_text())
                    self.assertEqual(published['sha256'],canonical_sha(published['body']))
                    self.assertEqual(published['body']['champion'],loop.state['champion'])
                    self.assertEqual(published['body']['loop_contract_sha256'],canonical_sha(loop.contract))
            changed=json.loads(pointer.read_text());changed['body']['loop_contract_sha256']='0'*64
            changed['sha256']=canonical_sha(changed['body']);pointer.write_text(json.dumps(changed))
            with execute(Path(directory)/'run',True) as learner:
                with self.assertRaisesRegex(ValueError,'another loop'):
                    ChampionLoop(learner,LoopPlan((128,256)),FixtureGate(True),publish_manifest=pointer,resume=True)

    def test_real_independent_gate_uses_complete_color_pairs_and_never_promotes_draws(self):
        with tempfile.TemporaryDirectory() as directory:
            with execute(Path(directory)/'learner') as learner:
                candidate=FrozenNeuralPolicy.from_archive(learner.root/'weights/000000000000',label='candidate')
                gate=IndependentGate(GatePlan(1,(4,)),reserve_bytes=0)
                decision=gate.evaluate(candidate,candidate,{'random':BaselinePolicy('random')},
                    Path(directory)/'gate',round_index=0,time_cap_seconds=30)
                self.assertFalse(decision['promotion_permitted'])
                self.assertEqual(decision['status'],'complete')
                self.assertTrue((Path(directory)/'gate/head/evidence.sqlite').is_file())


if __name__=='__main__':unittest.main()

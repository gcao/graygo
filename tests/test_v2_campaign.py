import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from graygo.engine import Rules,State,resolve,solo_repeat_hints,transform
from graygo.encoding import encode,KO_ENCODING_VERSION
from graygo.rule_state import RuleObservation,RuleStateSpec,RuleStateCodec
from graygo.native_decode import build_native
from graygo.models import ModelConfig,PolicyNet,QNet,transfer_board_weights
from graygo.betting import log_e,estimate
from graygo.board_runtime import BoardRuntimeConfig,BoardNFSPRuntime
from tools.v2_gate import V2GatePlan

class V2CampaignTests(unittest.TestCase):
    def states(self):
        rng=np.random.default_rng(1102026);s=State(Rules(5,40,ko='full-history'))
        for i in range(20):
            s=resolve(s,int(rng.choice(s.legal_actions(1))),int(rng.choice(s.legal_actions(2)))).state
            if i in (0,5,12,19):yield s

    def test_conditional_hints_match_independent_reference_for_every_legal_placement(self):
        for s in self.states():
            hints=solo_repeat_hints(s)
            self.assertIs(hints,solo_repeat_hints(s))
            for player in (1,2):
                expected=set()
                for move in s.legal_actions(player):
                    if move==s.rules.pass_action:continue
                    t=resolve(s,move,s.rules.pass_action) if player==1 else resolve(s,s.rules.pass_action,move)
                    if t.repetition_rejected:expected.add(move)
                self.assertEqual(hints[player-1],frozenset(expected))
                self.assertEqual(encode(s,player,ko_hints=True).spatial.shape,(32,5,5))
            # Hints never narrow the legal mask.
            np.testing.assert_array_equal(encode(s,1).legal,encode(s,1,ko_hints=True).legal)

    def test_native_and_python_replay_regenerate_identical_packed_hints(self):
        with tempfile.TemporaryDirectory() as directory:
            library=build_native(directory)['library'];rules=Rules(5,40,ko='full-history')
            python=RuleStateSpec(rules,ko_hints=True);native=RuleStateSpec(rules,library,ko_hints=True)
            states=list(self.states());rows=np.stack([np.frombuffer(python.codec.pack(s,p),np.uint8) for s in states for p in (1,2)])
            empty=np.empty((len(rows),0),np.float32);empty_bits=np.empty((len(rows),0),np.uint8)
            actual=native.unpack(rows,empty,empty_bits);reference=python.unpack(rows,empty,empty_bits)
            for a,b in zip(actual,reference):np.testing.assert_array_equal(a,b)
            for i,(s,p) in enumerate((s,p) for s in states for p in (1,2)):
                np.testing.assert_array_equal(actual[0][i],encode(s,p,ko_hints=True).spatial)
            self.assertEqual(RuleStateSpec.from_dict(native.to_dict()),native)
            self.assertEqual(python.version,KO_ENCODING_VERSION)
            with self.assertRaises(ValueError):RuleStateCodec(rules).unpack(rows[0])
            with self.assertRaises(ValueError):python.pack(RuleObservation(states[0],1))
            valid=np.ones(len(rows),bool);valid[-1]=False;rows[-1]=255
            x,g,m=native.unpack(rows,empty,empty_bits,valid=valid)
            self.assertFalse(x[-1].any());self.assertFalse(m[-1].any())

    def test_added_inputs_preserve_existing_model_function_and_global_positions(self):
        torch.set_num_threads(1);torch.manual_seed(19);s=next(self.states())
        for cls in (PolicyNet,QNet):
            original=cls(ModelConfig(8,1,2));new=cls(ModelConfig(8,1,2),spatial_channels=32)
            transfer_board_weights(new,original.state_dict())
            stem=new.encoder.stem.weight;old=original.encoder.stem.weight
            self.assertTrue(torch.equal(stem[:,:30],old[:,:30]));self.assertFalse(bool(stem[:,30:32].any()))
            self.assertTrue(torch.equal(stem[:,32:],old[:,30:]))
            a,b=encode(s,1),encode(s,1,ko_hints=True)
            with torch.no_grad():
                expected=original(torch.tensor(a.spatial[None].copy()),torch.tensor(a.global_features[None].copy()))
                actual=new(torch.tensor(b.spatial[None].copy()),torch.tensor(b.global_features[None].copy()))
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_hint_runtime_checkpoint_resumes_full_history_and_fresh_schema(self):
        torch.set_num_threads(1)
        cfg=BoardRuntimeConfig(rules=Rules(3,10,ko='full-history'),history_features=True,envs=2,width=8,blocks=1,norm_groups=2,
            capacity=64,batch=4,learning_start=4,target_every=8,reservoir_open=0,handoff_end=2,epsilon_offset_records=80_000_000)
        self.assertEqual(cfg.schedule().epsilon(0),.01)
        with tempfile.TemporaryDirectory() as directory:
            a=BoardNFSPRuntime(cfg);a.run_profile(8,time_cap_seconds=30);checkpoint=a.save(directory,reserve_bytes=0)
            b=BoardNFSPRuntime(cfg);b.restore(checkpoint)
            for x,y in zip(a.collector.active,b.collector.active):
                self.assertEqual(None if x is None else x.state,None if y is None else y.state)
            a.run_profile(16,time_cap_seconds=30);b.run_profile(16,time_cap_seconds=30)
            for key,value in a.backend.q.state_dict().items():self.assertTrue(torch.equal(value,b.backend.q.state_dict()[key]))
            with self.assertRaises(ValueError):BoardNFSPRuntime(replace(cfg,history_features=False)).restore(checkpoint)

    def test_fixed_bet_mixture_has_unit_expectation_under_null(self):
        for threshold in (0.,-.15):
            # IID two-point population with mean exactly the null boundary.
            prob=(threshold+1)/2;expected=0.
            for bits in range(16):
                values=np.array([1 if bits&(1<<i) else -1 for i in range(4)])
                likelihood=np.prod(np.where(values==1,prob,1-prob))
                expected+=likelihood*np.exp(log_e(values,threshold))
            self.assertAlmostEqual(expected,1.,places=12)
        with self.assertRaises(ValueError):log_e([np.nan],0)
        with self.assertRaises(ValueError):log_e([2],0)
        bound=estimate([1]*100,alpha=.01)
        self.assertGreater(bound['lower'],0);self.assertLessEqual(bound['lower'],bound['mean'])

    def test_gate_spends_only_actual_attempts_and_preserves_historical_guard(self):
        plan=V2GatePlan(looks=(2000,))
        self.assertAlmostEqual(sum(plan.family(i,['old']).alpha for i in range(10000)),.05-.05/10001)
        good=plan.numeric(0,[1]*1000,{'old':[1]*1000},{'old':[1]*1000},2000)
        self.assertTrue(good['numeric_pass'])
        bad=plan.numeric(0,[1]*1000,{'old':[-1]*1000},{'old':[1]*1000},2000)
        self.assertFalse(bad['numeric_pass']);self.assertFalse(bad['criteria']['no_historical_regression'])
        self.assertFalse(plan.numeric(0,[-1]*1000,{'old':[1]*1000},{'old':[1]*1000},2000)['numeric_pass'])

    def test_parallel_v2_gate_round_trips_real_frozen_hint_models(self):
        from graygo.manifest import source_digest,write_report
        from tools.v2_models import archive_runtime,describe,export
        from tools.parallel_gate import ParallelGate
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=BoardRuntimeConfig(rules=Rules(ko='full-history'),history_features=True,
                width=8,blocks=1,norm_groups=2,envs=2,capacity=64,batch=4,learning_start=4,reservoir_open=0,handoff_end=2)
            runtime=BoardNFSPRuntime(cfg)
            write_report(root/'run.json',{'source_sha256':source_digest(),'rules':cfg.rules.to_dict(),'config':cfg.plan()},reserve_bytes=0)
            archive=archive_runtime(runtime,root,0);descriptor=export(describe(archive,'tiny-v2'),root/'releases')
            gate=ParallelGate(V2GatePlan(looks=(4,)),workers=2,reserve_bytes=0)
            decision=gate.evaluate_descriptors(descriptor,descriptor,{'historical-'+str(i):descriptor for i in range(3)},root/'gate',round_index=0,time_cap_seconds=180)
            self.assertEqual(decision['status'],'complete');self.assertFalse(decision['numeric_pass'])
            self.assertEqual(decision['actual_attempt'],1)
            replay=gate.evaluate_descriptors(descriptor,descriptor,{'historical-'+str(i):descriptor for i in range(3)},root/'gate',round_index=0,time_cap_seconds=180)
            self.assertEqual(replay,decision)

    def test_async_result_cannot_publish_against_a_stale_incumbent(self):
        from tools.v2_campaign import save,finish_job
        from graygo.confidence import canonical_sha
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'job';folder.mkdir()
            spec={'mode':'gate','candidate':{'policy_sha256':'candidate'},'champion':{'policy_sha256':'old'}}
            save(folder/'job.json',spec)
            save(folder/'result.json',{'job_sha256':canonical_sha(spec),'mode':'gate','candidate':'candidate','champion':'old',
                'decision':{'status':'complete'}})
            state={'champion':{'policy_sha256':'new'},'job':{'root':str(folder),'mode':'gate','candidate':spec['candidate'],'champion_sha256':'old'}}
            with self.assertRaisesRegex(ValueError,'stale-incumbent'):finish_job(root,{},state)

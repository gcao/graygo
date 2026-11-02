from dataclasses import replace
import json
from pathlib import Path
import tempfile
import contextlib
import io
import unittest
import numpy as np
import torch
from graygo.matrix_training import MatrixActorCritic,MatrixPPOConfig,MatrixPPOTrainer,preflight_summary


class MatrixTrainingTests(unittest.TestCase):
    def setUp(self):
        self.output=io.StringIO()
        self.redirect=contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(lambda:self.redirect.__exit__(None,None,None))

    def config(self,**kwargs):
        return MatrixPPOConfig(records=4096,rollout=256,eval_every=512,hidden=16,epochs=2,minibatch=64,**kwargs)

    def test_rollout_is_on_policy_legal_and_role_balanced(self):
        with tempfile.TemporaryDirectory() as root:
            trainer=MatrixPPOTrainer(self.config(),root,reserve_bytes=0)
            self.assertEqual(trainer.manifest["ppo_base_settings"]["target_kl"],.02)
            self.assertEqual(trainer.manifest["optimizer"]["defaults"]["lr"],3e-4)
            batch,policy,counts=trainer.collect(256)
            batch.validate()
            self.assertTrue((batch.returns.abs()<=1).all())
            self.assertEqual(counts,{"current":256,"history":0})
            from unittest.mock import patch
            fixed=torch.tensor([[0],[0],[1],[1]])
            with patch("graygo.matrix_training.torch.multinomial",side_effect=[fixed,fixed]):
                controlled,_,_=trainer.collect(4)
            torch.testing.assert_close(controlled.returns,torch.tensor([1.,-1.,.5,-.5]))
            with torch.no_grad():
                logits,_=trainer.model(**batch.inputs)
                from graygo.models import masked_log_probs
                logp=masked_log_probs(logits,batch.legal).gather(1,batch.actions[:,None]).squeeze(1)
            torch.testing.assert_close(logp,batch.old_log_probs,atol=0,rtol=0)
            before={k:v.clone() for k,v in trainer.model.state_dict().items()}
            # Collecting opponent moves cannot alter learner parameters.
            trainer.collect(256)
            for k,v in trainer.model.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)

    def test_exact_resume_preserves_training_and_evaluation(self):
        with tempfile.TemporaryDirectory() as root:
            full=MatrixPPOTrainer(self.config(),Path(root)/"full",reserve_bytes=0)
            expected=full.run()
            partial=MatrixPPOTrainer(self.config(),Path(root)/"split",reserve_bytes=0)
            self.assertEqual(partial.run(until_records=2048)["status"],"segment_paused")
            resumed=MatrixPPOTrainer(self.config(),Path(root)/"split",resume=True,reserve_bytes=0)
            actual=resumed.run()
            for k,v in full.model.state_dict().items():torch.testing.assert_close(v,resumed.model.state_dict()[k],atol=0,rtol=0)
            self.assertEqual(expected["records"],4096)
            self.assertEqual(actual["per_role_records"],[2048,2048])
            self.assertEqual(actual["optimizer_steps"],expected["optimizer_steps"])
            self.assertEqual(actual["current_opponent_records"],expected["current_opponent_records"])
            self.assertEqual(actual["historical_opponent_records"],expected["historical_opponent_records"])
            for a,b in zip(actual["evaluations"],expected["evaluations"]):
                for field in ("records","latest_gaps","mixture_gaps","policies","mixture_components"):
                    self.assertEqual(a[field],b[field])
            self.assertGreater(actual["historical_opponent_records"],0)
            self.assertEqual(len(resumed.store._bundles()),1)
            self.assertFalse(actual["formal_protocol"])

    def test_critic_bucket_does_not_enter_actor(self):
        model=MatrixActorCritic(3,16)
        x=torch.eye(2)
        a,_=model(x,torch.tensor([[1.,0.],[1.,0.]]))
        b,_=model(x,torch.tensor([[0.,1.],[0.,1.]]))
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_stop_at_boundary_saves_a_resumable_run(self):
        with tempfile.TemporaryDirectory() as root:
            trainer=MatrixPPOTrainer(self.config(),root,reserve_bytes=0)
            result=trainer.run(stop_requested=lambda:"interrupted")
            self.assertEqual(result["status"],"interrupted")
            self.assertEqual(result["records"],0)
            resumed=MatrixPPOTrainer(self.config(),root,resume=True,reserve_bytes=0)
            self.assertEqual(resumed.records,0)
            with self.assertRaises(ValueError):MatrixPPOTrainer(replace(self.config(),entropy="fixed-0.1"),root,resume=True,reserve_bytes=0)
            with self.assertRaises(FileExistsError):MatrixPPOTrainer(self.config(),root,reserve_bytes=0)

    def test_schedule_and_gate_need_consistent_representation(self):
        c=MatrixPPOConfig()
        self.assertAlmostEqual(c.entropy_at(0),.01)
        self.assertAlmostEqual(c.entropy_at(1_000_000),.001)
        self.assertAlmostEqual(replace(c,entropy="fixed-0.1").entropy_at(500),.1)
        results=[]
        for entropy in ("taper","fixed-0.1"):
            for task in ("asymmetric","rps"):
                for seed in (0,1,2):
                    results.append({"task":task,"entropy_protocol":entropy,"seed":seed,"status":"complete",
                                    "has_five_final_evaluations":True,"formal_protocol":True,
                                    "latest_passed":task=="asymmetric","mixture_passed":task=="rps"})
        self.assertEqual(preflight_summary(results)["route"],"nfsp")
        for r in results:
            if r["entropy_protocol"]=="taper":r["mixture_passed"]=True
        self.assertEqual(preflight_summary(results)["route"],"ppo")
        self.assertEqual(preflight_summary(results[:-1])["route"],"pending")
        results[0]["formal_protocol"]=False
        self.assertEqual(preflight_summary(results)["route"],"pending")


    def test_tabular_reference_against_known_mixed_value(self):
        from graygo.reference_learning import fictitious_play
        result=fictitious_play([[1.,-1.],[-1.,.5]],iterations=10000,seed=0)
        self.assertAlmostEqual(result["value"],-1/7)
        self.assertLess(max(result["gaps"]),.03)
        np.testing.assert_allclose(result["row"],[3/7,4/7],atol=.02)

    def test_action_dimension_mask_for_pure82(self):
        with tempfile.TemporaryDirectory() as root:
            trainer=MatrixPPOTrainer(self.config(task="pure82"),root,reserve_bytes=0)
            data,_,_=trainer.collect(256)
            self.assertEqual(data.legal.shape,(256,82))
            data.validate()
            self.assertTrue((data.returns.abs()<=1).all())


    def test_suite_reuses_verified_cases_and_never_grades_a_short_diagnostic(self):
        from graygo.preflight import run_ppo_preflight
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as root:
            with patch("graygo.preflight.matrix_reference_controls",return_value={"passed":True,"control":"orchestration-test-only"}):
                first=run_ppo_preflight(root,records=128,reserve_bytes=0)
            self.assertEqual(first["trials"],12)
            self.assertFalse(first["complete"])
            self.assertEqual(first["route"],"pending")
            with patch("graygo.matrix_training.MatrixPPOTrainer.collect",side_effect=AssertionError("completed case restarted")):
                again=run_ppo_preflight(root,records=128,reserve_bytes=0)
            self.assertEqual(again["results"],first["results"])
            with self.assertRaises(ValueError):run_ppo_preflight(root,records=256,reserve_bytes=0)

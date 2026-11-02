from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from graygo.confidence import ConfidenceFamily
from graygo.engine import BLACK,WHITE,Rules,State,resolve
from graygo.evaluation import BaselinePolicy,EpisodeMixture,FrozenNeuralPolicy,EvaluationPlan,EvaluationRun,evaluate_pair
from graygo.evaluation_batch import PairRequest,commit_evaluation_actions,evaluate_pairs_batched
from graygo.evaluation_evidence import FrozenEvaluationEvidence
from graygo.models import ModelConfig,PolicyNet,QNet


class BatchedEvaluationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def roots(self,size=3,limit=8):
        roots=[]
        for rule in ('gray','remove'):
            empty=State(Rules(size,limit,rule));roots.extend((empty,resolve(empty,0,1).state))
        return roots

    def policy(self,representation='average',batch=8,device='cpu'):
        torch.manual_seed(72)
        model=(QNet if representation=='greedy_q' else PolicyNet)(ModelConfig(8,1,2)).to(device)
        return FrozenNeuralPolicy(model,representation=representation,inference_batch_size=batch)

    def test_baseline_games_match_scalar_with_colors_incumbents_and_mixtures(self):
        random=BaselinePolicy('random');tactical=BaselinePolicy('tactical');mixture=EpisodeMixture((random,tactical),(.4,.6))
        requests=[PairRequest(mixture,tactical,root,100+i,random) for i,root in enumerate(self.roots())]
        reference=[evaluate_pair(r.subject,r.opponent,r.root,unit_seed=r.unit_seed,incumbent=r.incumbent) for r in requests]
        self.assertEqual(evaluate_pairs_batched(requests),reference)
        self.assertEqual(evaluate_pairs_batched(list(reversed(requests))),list(reversed(reference)))

    def test_private_actor_sampling_and_global_rng_are_unchanged(self):
        policy=self.policy();roots=self.roots();seeds=list(range(8));root_pairs=[(roots[i%len(roots)],i%2+1) for i in seeds]
        scalar=[policy.begin_episode(seed) for seed in seeds];batched=[policy.begin_episode(seed) for seed in seeds]
        torch_rng=torch.get_rng_state().clone();numpy_rng=np.random.get_state()
        expected=[actor.action(state,role) for actor,(state,role) in zip(scalar,root_pairs)]
        actual=commit_evaluation_actions([(actor,state,role) for actor,(state,role) in zip(batched,root_pairs)])
        self.assertEqual(actual,expected)
        for a,b in zip(scalar,batched):self.assertEqual(a.rng.bit_generator.state,b.rng.bit_generator.state)
        torch.testing.assert_close(torch.get_rng_state(),torch_rng,atol=0,rtol=0)
        self.assertEqual(numpy_rng[0],np.random.get_state()[0]);np.testing.assert_array_equal(numpy_rng[1],np.random.get_state()[1])

    def test_fixed_shape_single_probe_matches_grouped_rows_and_reordering(self):
        for representation in ('average','greedy_q'):
            policy=self.policy(representation);requests=[(root,role) for root in self.roots() for role in (BLACK,WHITE)]
            expected=np.stack([policy.initial_distribution(root,role) for root,role in requests])
            actual=np.stack(policy._action_weights_many(requests));actual/=actual.sum(-1,keepdims=True)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
            reverse=np.stack(policy._action_weights_many(list(reversed(requests))));reverse/=reverse.sum(-1,keepdims=True)
            np.testing.assert_array_equal(reverse,actual[::-1])

    def test_real_neural_games_and_episode_mixtures_match_scalar_reference(self):
        policy=self.policy();q=self.policy('greedy_q');mixture=EpisodeMixture((policy,q),(.5,.5))
        requests=[PairRequest(mixture,policy,root,81+i,q) for i,root in enumerate(self.roots())]
        reference=[evaluate_pair(r.subject,r.opponent,r.root,unit_seed=r.unit_seed,incumbent=r.incumbent) for r in requests]
        self.assertEqual(evaluate_pairs_batched(requests),reference)
        self.assertEqual(evaluate_pairs_batched(requests[:1])+evaluate_pairs_batched(requests[1:]),reference)

    def test_full_model_grouping_uses_fewer_actual_forwards(self):
        policy=self.policy(batch=64);roots=self.roots();calls=[]
        handle=policy._model.register_forward_pre_hook(lambda model,args:calls.append(args[0].shape[0]))
        try:
            actors=[policy.begin_episode(i) for i in range(8)]
            commit_evaluation_actions([(actor,roots[i%4],i%2+1) for i,actor in enumerate(actors)])
            self.assertLess(len(calls),len(actors));self.assertTrue(all(size==64 for size in calls))
        finally:handle.remove()

    def test_inference_shape_is_frozen_and_unsupported_models_are_rejected(self):
        p=self.policy();other=FrozenNeuralPolicy(p._model,inference_batch_size=4)
        self.assertNotEqual(p.sha256,other.sha256);p.inference_batch_size=4
        with self.assertRaisesRegex(ValueError,'inference layout'):p.assert_frozen()
        class CrossRow(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.zeros(()))
            def forward(self,x,g):return x.mean(0)+self.weight
        with self.assertRaisesRegex(ValueError,'independent-row'):FrozenNeuralPolicy(CrossRow(),inference_batch_size=8)
        for value in (0,65,True):
            with self.assertRaises(ValueError):self.policy(batch=value)

    def test_terminal_mixed_shape_illegal_actions_and_empty_groups_rejected(self):
        p=self.policy()
        for requests in ([],[PairRequest(p,p,State(Rules(3,1)),i) for i in range(33)]):
            with self.assertRaises(ValueError):evaluate_pairs_batched(requests)
        terminal=resolve(State(Rules(3,1)),0,1).state
        with self.assertRaises(ValueError):PairRequest(p,p,terminal,0)
        with self.assertRaises(ValueError):p._action_weights_many([(State(Rules(3,2)),BLACK),(State(Rules(9,2)),WHITE)])
        class Bad:
            def action(self,state,player):return -1
        with self.assertRaisesRegex(ValueError,'illegal'):commit_evaluation_actions([(Bad(),State(Rules(3,2)),BLACK)])

    def test_persistent_batched_resume_keeps_scored_prefix_and_plan_bound(self):
        p=self.policy();opponent=BaselinePolicy('random');family=ConfidenceFamily('batch',('u',),(4,8))
        plan=EvaluationPlan('batch','p','u',Rules(3,5),5,family.looks,family.sha256,pair_batch_size=3)
        with tempfile.TemporaryDirectory() as directory:
            full=EvaluationRun(plan,Path(directory)/'full',p,opponent,reserve_bytes=0);full.run(8,time_cap_seconds=30)
            part=EvaluationRun(plan,Path(directory)/'part',p,opponent,reserve_bytes=0);part.run(4,time_cap_seconds=30)
            resumed=EvaluationRun(plan,Path(directory)/'part',p,opponent,resume=True,reserve_bytes=0);resumed.run(8,time_cap_seconds=30)
            self.assertEqual(full.records(),resumed.records())
            self.assertEqual(FrozenEvaluationEvidence.load(resumed.root).available_games,8)
            with self.assertRaises(ValueError):EvaluationRun(replace(plan,pair_batch_size=1),resumed.root,p,opponent,resume=True,reserve_bytes=0)

    def test_group_failure_commits_no_half_pairs_and_stop_preserves_empty_prefix(self):
        p=self.policy();family=ConfidenceFamily('fail',('u',),(4,8));plan=EvaluationPlan('fail','p','u',Rules(3,5),8,family.looks,family.sha256,pair_batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            run=EvaluationRun(plan,directory,p,p,reserve_bytes=0)
            self.assertEqual(run.run(8,time_cap_seconds=30,stop_requested=lambda:True)['status'],'interrupted')
            with patch('graygo.evaluation_batch.commit_evaluation_actions',side_effect=RuntimeError('injected group failure')):
                with self.assertRaisesRegex(RuntimeError,'group failure'):run.run(8,time_cap_seconds=30)
            self.assertEqual(run.records(),[])
            run.run(4,time_cap_seconds=30);self.assertEqual(len(run.records()),2)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_fixed_shape_batch_marginals_actions_and_trajectories_reproduce(self):
        from graygo.board_runtime import configure_runtime
        configure_runtime('cuda',True);p=self.policy(device='cuda');requests=[(root,role) for root in self.roots() for role in (BLACK,WHITE)]
        expected=np.stack([p.initial_distribution(root,role) for root,role in requests])
        actual=np.stack(p._action_weights_many(requests));actual/=actual.sum(-1,keepdims=True)
        np.testing.assert_array_equal(actual,expected)
        games=[PairRequest(p,p,root,91+i) for i,root in enumerate(self.roots())]
        scalar=[evaluate_pair(r.subject,r.opponent,r.root,unit_seed=r.unit_seed) for r in games]
        self.assertEqual(evaluate_pairs_batched(games),scalar)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_full_network_nine_board_probe_marginals_are_independent_of_batch_members(self):
        from graygo.board_runtime import configure_runtime
        configure_runtime('cuda',True);torch.manual_seed(311);roots=[State(Rules())];rng=np.random.default_rng(33)
        for _ in range(7):
            state=roots[-1];roots.append(resolve(state,int(rng.choice(state.legal_actions(BLACK))),int(rng.choice(state.legal_actions(WHITE)))).state)
        requests=[(root,role) for root in roots for role in (BLACK,WHITE)]
        for representation,kind in [('average',PolicyNet),('greedy_q',QNet)]:
            policy=FrozenNeuralPolicy(kind().to('cuda'),representation=representation,inference_batch_size=32)
            before=torch.cuda.get_rng_state().clone()
            expected=np.stack([policy.initial_distribution(state,role) for state,role in requests])
            for order in (requests,list(reversed(requests))):
                actual=np.stack(policy._action_weights_many(order));actual/=actual.sum(-1,keepdims=True)
                np.testing.assert_array_equal(actual,expected if order is requests else expected[::-1])
            torch.testing.assert_close(torch.cuda.get_rng_state(),before,atol=0,rtol=0)


if __name__=='__main__':unittest.main()

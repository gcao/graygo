from copy import deepcopy
from dataclasses import asdict
import io
import unittest
import numpy as np
import torch
from torch import nn
from graygo.buffers import ObservationSpec,ReplayBuffer,ReservoirBuffer
from graygo.encoding import Observation
from graygo.nfsp import NFSPConfig,NFSPCadence,OneStepBatch,one_step_targets,update_q,update_average,NFSPBackend
from graygo.task_models import CompactQ,CompactPolicy,TabularQ


class Table(nn.Module):
    def __init__(self,values):
        super().__init__();self.values=nn.Parameter(torch.tensor(values,dtype=torch.float32))
    def forward(self,spatial,globals):return spatial@self.values


def fixture():
    inputs={'spatial':torch.tensor([[1.,0.],[0.,1.]]),'globals':torch.empty(2,0)}
    following={'spatial':torch.tensor([[0.,1.],[0.,0.]]),'globals':torch.empty(2,0)}
    return OneStepBatch(inputs,torch.tensor([[True,True,False]]*2),torch.tensor([0,1]),torch.tensor([0.,-1.]),
                        following,torch.tensor([[True,True,False],[False,False,False]]),torch.tensor([False,True]))


class NFSPTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(5)

    def test_one_step_uses_online_policy_and_frozen_values(self):
        batch=fixture();online=Table([[0.,0.,1.],[.9,.1,1.]]);target=Table([[0.,0.,1.],[.2,.8,1.]])
        labels,stats=one_step_targets(online,target,batch,.1)
        # pi=(.95,.05), evaluated by target=(.2,.8): .23. The illegal high action is excluded.
        torch.testing.assert_close(labels,torch.tensor([.23,-1.]))
        self.assertFalse(labels.requires_grad);self.assertEqual(stats['projected_targets'],0)
        self.assertIsNone(target.values.grad);self.assertIsNone(online.values.grad)
        # No stored suffix is required to retain a nonterminal bootstrap.
        batch.terminated[0]=True;batch.next_legal[0]=False
        terminal,_=one_step_targets(online,target,batch,.1)
        torch.testing.assert_close(terminal,torch.tensor([0.,-1.]))

    def test_legal_ties_and_per_start_epsilon(self):
        batch=fixture();online=Table([[0.,0.,1.],[.5,.5,1.]]);target=Table([[0.,0.,1.],[.2,.8,1.]])
        labels,_=one_step_targets(online,target,batch,torch.tensor([0.,1.]))
        torch.testing.assert_close(labels,torch.tensor([.5,-1.]))
        online.values.data[1,0]=.9
        labels,_=one_step_targets(online,target,batch,torch.tensor([1.,0.]))
        torch.testing.assert_close(labels,torch.tensor([.5,-1.]))

    def test_q_update_changes_executed_action_not_greedy_label(self):
        batch=fixture();online=Table([[0.,0.,0.],[.4,.1,0.]]);target=deepcopy(online).requires_grad_(False)
        optimizer=torch.optim.SGD(online.parameters(),lr=.2)
        before=online.values.detach().clone();frozen=target.values.detach().clone()
        stats=update_q(online,target,optimizer,batch,epsilon=.1,max_grad_norm=10)
        self.assertGreater(online.values[0,0],before[0,0]);self.assertLess(online.values[1,1],before[1,1])
        self.assertEqual(online.values[1,0],before[1,0]);torch.testing.assert_close(target.values,frozen)
        self.assertEqual(stats['targets'],2);self.assertTrue(np.isfinite(stats['loss']))

    def test_frozen_supervised_ties_and_mask_have_finite_gradient(self):
        policy=Table([[0.,0.,0.]])
        data={'spatial':np.ones((1,1)),'globals':np.empty((1,0)),
              'legal':np.array([[True,True,False]]),'targets':np.array([[.25,.75,0.]])}
        optimizer=torch.optim.SGD(policy.parameters(),lr=.2)
        update_average(policy,optimizer,data,device='cpu')
        self.assertTrue(torch.isfinite(policy.values).all());self.assertGreater(policy.values[0,1],policy.values[0,0])
        self.assertEqual(float(policy.values[0,2].detach()),0.)
        bad=deepcopy(data);bad['targets'][0]=[.2,.7,.1]
        with self.assertRaises(ValueError):update_average(policy,optimizer,bad,device='cpu')

    def test_strict_fresh_cadence_reuse_does_not_accelerate_target(self):
        for reuse in (1,4):
            cfg=NFSPConfig(batch=128,learning_start=4096,q_reuse=reuse)
            cadence=NFSPCadence();qevents=refreshes=0
            for _ in range(100):
                q,sl,target=cadence.advance(128,128,cfg)
                qevents+=q*reuse;refreshes+=target
                self.assertTrue(sl)
            self.assertEqual(qevents,69*reuse);self.assertEqual(refreshes,1)
            self.assertEqual(cadence.fresh_q,12800)
            with self.assertRaises(ValueError):cadence.advance(129,129,cfg)
            self.assertEqual(cadence.fresh_q,12800)

    def make_backend(self,reuse=1):
        config=NFSPConfig(batch=4,learning_start=4,q_reuse=reuse,target_every=8)
        return NFSPBackend(CompactQ(2,0,2,8),CompactPolicy(2,0,2,8),config)

    def test_actual_replay_reservoir_optimizer_and_exact_resume(self):
        spec=ObservationSpec((2,),0,2,'test/nfsp/1')
        replay=ReplayBuffer(32,spec);reservoir=ReservoirBuffer(32,spec)
        rng=np.random.default_rng(17);backend=self.make_backend(4)
        terminal=Observation(np.zeros(2),np.empty(0),np.zeros(2,dtype=bool))
        def add_block(start):
            for record in range(start,start+4):
                role=record%2;obs=Observation(np.eye(2)[role],np.empty(0),np.ones(2,dtype=bool))
                replay.add(obs,role,1 if role==0 else -1,terminal,True,behavior_prob=.5,stream=record,step=0)
                reservoir.offer(obs,np.array([True,False]),record=record,policy_version=0,rng=rng,best_response_mode=True)
        add_block(0)
        frozen=deepcopy(backend.target.state_dict())
        report=backend.learn_after_fresh(4,4,replay,reservoir,rng,epsilon=.1)
        self.assertEqual(len(report['q']),4);self.assertTrue(backend.average_ready);self.assertFalse(report['target_refreshed'])
        for name,data in frozen.items():torch.testing.assert_close(backend.target.state_dict()[name],data)
        saved=io.BytesIO();torch.save(backend.state_dict(),saved);saved.seek(0)
        restored=self.make_backend(4);restored.load_state_dict(torch.load(saved,weights_only=True))
        rng_state=deepcopy(rng.bit_generator.state)
        add_block(4);post_collection=deepcopy(rng.bit_generator.state)
        report=backend.learn_after_fresh(4,4,replay,reservoir,rng,epsilon=.07)
        resumed_rng=np.random.default_rng();resumed_rng.bit_generator.state=post_collection
        resumed=restored.learn_after_fresh(4,4,replay,reservoir,resumed_rng,epsilon=.07)
        self.assertEqual(report,resumed);self.assertEqual(asdict(backend.counters),asdict(restored.counters))
        self.assertEqual(backend.counters.q_loss_targets,32);self.assertEqual(backend.counters.target_refreshes,1)
        for name,data in backend.q.state_dict().items():
            torch.testing.assert_close(data,restored.q.state_dict()[name],rtol=0,atol=0)
            torch.testing.assert_close(data,backend.target.state_dict()[name],rtol=0,atol=0)
        self.assertEqual(rng.bit_generator.state,resumed_rng.bit_generator.state)

    def test_supervised_clock_skips_unavailable_reservoir_without_backlog(self):
        backend=self.make_backend();spec=ObservationSpec((2,),0,2,'test/nfsp/1')
        replay=ReplayBuffer(32,spec);reservoir=ReservoirBuffer(32,spec,opening_record=100)
        terminal=Observation(np.zeros(2),np.empty(0),np.zeros(2,dtype=bool))
        source=Observation(np.array([1,0]),np.empty(0),np.ones(2,dtype=bool));rng=np.random.default_rng(2)
        for block in range(2):
            for record in range(block*4,block*4+4):replay.add(source,0,1,terminal,True,behavior_prob=.5,stream=record,step=0)
            backend.learn_after_fresh(4,4,replay,reservoir,rng,epsilon=.1)
        self.assertEqual(backend.counters.q_updates,2);self.assertEqual(backend.counters.supervised_updates,0)
        self.assertFalse(backend.average_ready)

    def test_invalid_rewards_masks_values_and_state_are_rejected(self):
        q=Table([[0.,0.,0.],[.1,.2,0.]])
        for mutation in ('mask','nan','shape','shaping'):
            batch=fixture()
            if mutation=='mask':batch.next_legal[1,0]=True
            if mutation=='nan':batch.rewards[1]=torch.nan
            if mutation=='shape':batch.actions[0]=2
            if mutation=='shaping':batch.rewards[0]=.1
            with self.assertRaises(ValueError):one_step_targets(q,q,batch,.1)
        bad=deepcopy(q);bad.values.data[1,0]=1.5
        with self.assertRaises(ValueError):one_step_targets(bad,q,fixture(),.1)
        with self.assertRaises(ValueError):one_step_targets(q,q,fixture(),float('nan'))
        with self.assertRaises(ValueError):one_step_targets(q,q,fixture(),torch.tensor([.1,float('nan')]))
        backend=self.make_backend();state=deepcopy(backend.state_dict());state['counters']['target_refreshes']=1
        with self.assertRaises(ValueError):backend.load_state_dict(state)

    def test_tabular_direct_estimates_project_and_reject_non_onehot(self):
        q=TabularQ(2,3);q.values.data[0,1]=2;q.project_()
        torch.testing.assert_close(q(torch.eye(2),torch.empty(2,0)),q.values)
        self.assertEqual(float(q.values[0,1].detach()),1.)
        with self.assertRaises(ValueError):q(torch.zeros(2,2),torch.empty(2,0))

    def test_independent_board_policy_and_full_encoder_update(self):
        from graygo.models import QNet,PolicyNet
        from graygo.engine import State,Rules
        from graygo.encoding import encode
        state=State(Rules())
        obs=encode(state,1)
        q=QNet();policy=PolicyNet();backend=NFSPBackend(q,policy)
        x=torch.tensor(obs.spatial[None]);g=torch.tensor(obs.global_features[None]);legal=torch.tensor(obs.legal[None])
        zero=torch.zeros_like(x);zg=torch.zeros_like(g)
        batch=OneStepBatch({'spatial':x,'globals':g},legal,torch.tensor([81]),torch.tensor([1.]),
                           {'spatial':zero,'globals':zg},torch.zeros_like(legal),torch.tensor([True]))
        stats=update_q(q,backend.target,backend.q_optimizer,batch,epsilon=.1)
        self.assertTrue(np.isfinite(stats['loss']));self.assertIsNone(next(policy.parameters()).grad)
        greedy=np.zeros((1,82),np.float32);greedy[0,81]=1
        update_average(policy,backend.average_optimizer,{'spatial':obs.spatial[None],
                        'globals':obs.global_features[None],'legal':obs.legal[None],'targets':greedy},device='cpu')
        self.assertTrue(torch.isfinite(policy(x,g)).all())

if __name__=='__main__':unittest.main()

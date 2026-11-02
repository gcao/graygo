import math
import unittest
import torch
from torch import nn
from graygo.ppo import PPOBatch,PPOConfig,generalized_advantages,update_ppo
from graygo.models import masked_log_probs


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=nn.Parameter(torch.zeros(3))
        self.critic=nn.Parameter(torch.zeros(()))
    def forward(self,features):
        return self.actor.expand(len(features),-1),self.critic.expand(len(features))


def batch(actor=True,targets=0.):
    return PPOBatch({"features":torch.zeros(4,1)},torch.tensor([0,1,0,1]),
                    torch.tensor([[True,True,False]]*4),torch.full((4,),-math.log(2)),
                    torch.tensor([1.,-1.,1.,-1.]),torch.full((4,),float(targets)),
                    torch.full((4,),actor,dtype=torch.bool),torch.ones(4,dtype=torch.bool))


class PPOTests(unittest.TestCase):
    def test_terminal_and_chunk_bootstrap(self):
        rewards=torch.tensor([[0.],[1.]])
        values=torch.tensor([[.1],[.2]])
        adv,returns=generalized_advantages(rewards,values,torch.tensor([[False],[True]]),torch.tensor([99.]),lam=1)
        torch.testing.assert_close(returns,torch.ones(2,1))
        torch.testing.assert_close(adv,torch.tensor([[.9],[.8]]))
        _,returns=generalized_advantages(torch.zeros(1,1),torch.tensor([[.2]]),torch.tensor([[False]]),torch.tensor([.8]))
        torch.testing.assert_close(returns,torch.tensor([[.8]]))

    def test_no_future_episode_leak(self):
        _,returns=generalized_advantages(torch.tensor([[1.],[100.]]),torch.zeros(2,1),torch.tensor([[True],[True]]),torch.zeros(1),lam=1)
        self.assertEqual(returns[0].item(),1)

    def test_policy_direction_mask_and_finite_gradient(self):
        model=TinyModel();optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        stats=update_ppo(model,optimizer,batch(),PPOConfig(epochs=1,minibatch=4,value_weight=0,entropy_weight=0))
        self.assertEqual(stats["updates"],1)
        self.assertGreater(model.actor[0].item(),model.actor[1].item())
        self.assertEqual(model.actor[2].item(),0)
        self.assertTrue(torch.isfinite(model.actor).all())

    def test_value_only_has_no_actor_head_loss(self):
        model=TinyModel();before=model.actor.detach().clone()
        update_ppo(model,torch.optim.SGD(model.parameters(),lr=.1),batch(False,1.),PPOConfig(epochs=1,minibatch=4))
        torch.testing.assert_close(model.actor,before,atol=0,rtol=0)
        self.assertGreater(model.critic.item(),0)

    def test_kl_stops_before_oversized_update(self):
        model=TinyModel();data=batch();data.old_log_probs.fill_(-10)
        before=model.actor.detach().clone()
        stats=update_ppo(model,torch.optim.SGD(model.parameters(),lr=.1),data,PPOConfig())
        self.assertTrue(stats["stopped_for_kl"])
        self.assertEqual(stats["updates"],0)
        torch.testing.assert_close(model.actor,before,atol=0,rtol=0)

    def test_illegal_or_nonfinite_batch_rejected(self):
        data=batch();data.actions[0]=2
        with self.assertRaises(ValueError):data.validate()
        data=batch();data.advantages[0]=float('nan')
        with self.assertRaises(ValueError):data.validate()


    def test_fully_excluded_records_do_not_step_optimizer(self):
        model=TinyModel();data=batch(False);data.value_mask.zero_()
        stats=update_ppo(model,torch.optim.Adam(model.parameters()),data)
        self.assertEqual(stats["updates"],0)
        torch.testing.assert_close(model.actor,torch.zeros(3),atol=0,rtol=0)

import random
import unittest
import numpy as np
from graygo.engine import BLACK as B, WHITE as W, GRAY as G, Rules, State, resolve, transform, transform_action
from graygo.encoding import encode
from graygo.agents import uniform, tactical


class EncodingTests(unittest.TestCase):
    def test_missing_history_and_perspective(self):
        s=State()
        a=encode(s,B)
        self.assertEqual(a.spatial.shape,(30,9,9))
        self.assertEqual(a.global_features.shape,(14,))
        self.assertEqual(a.spatial[3].sum(),81)
        self.assertEqual(a.spatial[16:28].sum(),0)
        self.assertTrue(a.legal.all())
        self.assertFalse(a.spatial.flags.writeable)
        s=resolve(s,0,1).state
        a,b=encode(s,B),encode(s,W)
        np.testing.assert_array_equal(a.spatial[0],b.spatial[1])
        self.assertEqual(tuple(a.global_features[:2]),(0,1))
        self.assertEqual(a.spatial[24:26].sum(),0)
        self.assertEqual(a.spatial[26,0,0],1)
        self.assertEqual(a.spatial[27,0,1],1)
        swapped=encode(transform(s,swap=True),W)
        np.testing.assert_array_equal(a.spatial,swapped.spatial)
        np.testing.assert_array_equal(a.global_features,swapped.global_features)

    def test_countdown_and_terminal_mask(self):
        s=State(Rules(3,2))
        s=resolve(s,0,1).state
        a=encode(s,B)
        self.assertEqual(a.global_features[7],0.5)
        self.assertTrue(a.global_features[8:].all())
        s=resolve(s,9,9).state
        self.assertTrue(s.done)
        self.assertFalse(encode(s,B).legal.any())

    def test_baselines_are_legal_reproducible_and_use_same_state(self):
        for agent in (uniform,tactical):
            rng1,rng2=random.Random(123),random.Random(123)
            self.assertEqual([agent(State(),B,rng1) for _ in range(10)],
                             [agent(State(),B,rng2) for _ in range(10)])
            s=State(Rules(3,10))
            rng=random.Random(7)
            while not s.done:
                a,b=agent(s,B,rng),agent(s,W,rng)
                self.assertIn(a,s.legal_actions(B))
                self.assertIn(b,s.legal_actions(W))
                s=resolve(s,a,b).state


try:
    import torch
    from graygo.models import PolicyValueNet,QNet,ModelConfig,masked_log_probs,greedy_distribution
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'training extra not installed')
class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(917)
        self.cfg=ModelConfig(width=8,blocks=1,norm_groups=2)

    def inputs(self,state):
        o=encode(state,B)
        return torch.tensor(o.spatial[None]),torch.tensor(o.global_features[None]),torch.tensor(o.legal[None])

    def test_shape_mask_and_backward(self):
        model=PolicyValueNet(self.cfg)
        s=resolve(State(),0,1).state
        x,g,mask=self.inputs(s)
        logits,value=model(x,g)
        logp=masked_log_probs(logits,mask)
        self.assertEqual(logits.shape,(1,82))
        self.assertEqual(logp.exp()[0,0].item(),0)
        self.assertAlmostEqual(logp.exp().sum().item(),1,places=6)
        loss=-logp[0,2]+value.square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        q=QNet(self.cfg)(x,g)
        self.assertTrue((q.abs()<=1).all())
        with self.assertRaises(ValueError):masked_log_probs(logits,torch.zeros_like(mask))
        tied=greedy_distribution(torch.zeros_like(logits),mask)
        self.assertEqual(tied[0,0].item(),0)
        self.assertAlmostEqual(tied[0,2].item(),1/80,places=6)

    def test_translation_and_empty_board_symmetry(self):
        model=PolicyValueNet(self.cfg).eval()
        s=resolve(State(),10,25).state
        x,g,_=self.inputs(s)
        y,h,_=self.inputs(transform(s,shift=(2,-3)))
        with torch.no_grad():
            a,v=model(x,g);b,w=model(y,h)
        expected=torch.roll(a[:,:81].reshape(1,9,9),(2,-3),(-2,-1)).flatten(1)
        torch.testing.assert_close(b[:,:81],expected,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(b[:,81],a[:,81],atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(v,w,atol=2e-6,rtol=2e-6)
        x,g,_=self.inputs(State())
        with torch.no_grad():a,_=model(x,g)
        torch.testing.assert_close(a[:,:81],a[:,0:1].expand(-1,81),atol=2e-6,rtol=2e-6)

    def test_critic_metadata_never_enters_actor(self):
        model=PolicyValueNet(self.cfg,critic_buckets=2)
        x,g,_=self.inputs(State())
        a,_=model(x,g,torch.tensor([[1.,0.]]))
        b,_=model(x,g,torch.tensor([[0.,1.]]))
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(a,model.policy_logits(x,g),atol=0,rtol=0)

    def test_numpy_input_board_is_copied(self):
        values=np.zeros(81,dtype=np.uint8)
        state=State(board=values)
        values[0]=B
        self.assertEqual(state.board[0],0)


class TacticalFixtureTests(unittest.TestCase):
    def position(self,color,neighbors_color):
        board=[0]*81
        board[30]=color
        for p in (21,39,29):board[p]=neighbors_color
        return State(board=tuple(board))

    def test_capture_black_white_and_gray_priorities(self):
        for color in (W,G):
            self.assertEqual(tactical(self.position(color,B),B,random.Random(1)),31)
        self.assertEqual(tactical(self.position(B,W),B,random.Random(1)),31)


@unittest.skipIf(torch is None,'training extra not installed')
class EpsilonPolicyTests(unittest.TestCase):
    def test_marginal_probability_includes_exploration_hits(self):
        from graygo.models import epsilon_greedy_distribution
        q=torch.tensor([[2.,2.,1.],[3.,1.,100.]])
        legal=torch.tensor([[True,True,True],[True,True,False]])
        result=epsilon_greedy_distribution(q,legal,torch.tensor([.2,.2]))
        torch.testing.assert_close(result,torch.tensor([[.46666667,.46666667,.06666667],[.9,.1,0.]]))
        torch.testing.assert_close(result.sum(-1),torch.ones(2))
        self.assertAlmostEqual(result[1,0].item(),.8+.2/2,places=6)
        torch.testing.assert_close(epsilon_greedy_distribution(q,legal,1),legal.float()/legal.sum(-1,keepdim=True))
        with self.assertRaises(ValueError):epsilon_greedy_distribution(q,legal,float('nan'))
        with self.assertRaises(ValueError):epsilon_greedy_distribution(q,legal,-.1)

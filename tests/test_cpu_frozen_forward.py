from copy import deepcopy
import gc
import unittest
import numpy as np
import torch
from graygo.cpu_frozen_forward import CPU3x3ForwardCache
from graygo.models import QNet,PolicyNet,ModelConfig
from graygo.nfsp_collection import FrozenPolicies,EpisodeActor
from graygo.schedules import EpisodeChoice
from graygo.encoding import encode
from graygo.engine import State,Rules,BLACK,WHITE,resolve


class CPUFrozenForwardTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(17)
    def model(self,kind=QNet):return kind(ModelConfig(8,1,2)).eval().requires_grad_(False)
    def inputs(self,size=3,batch=1):return {'spatial':torch.randn(batch,30,size,size),'globals':torch.randn(batch,14)}

    def test_hits_are_bit_exact_owned_and_do_not_advance_rng(self):
        for kind in (QNet,PolicyNet):
            model=self.model(kind);cache=CPU3x3ForwardCache();inputs=self.inputs();rng=torch.get_rng_state().clone()
            with torch.no_grad():
                expected=model(**inputs);first=cache.forward(model,**inputs);first.add_(7);second=cache.forward(model,**inputs);second.zero_();third=cache.forward(model,**inputs)
            torch.testing.assert_close(third,expected,atol=0,rtol=0);torch.testing.assert_close(torch.get_rng_state(),rng,atol=0,rtol=0);self.assertEqual(cache.stats()['hits'],2);self.assertEqual(cache.stats()['misses'],1)

    def test_changed_inputs_parameters_and_explicit_restore_recompute(self):
        model=self.model();cache=CPU3x3ForwardCache();inputs=self.inputs()
        with torch.no_grad():
            cache.forward(model,**inputs);inputs['globals'].add_(.01);torch.testing.assert_close(cache.forward(model,**inputs),model(**inputs),atol=0,rtol=0)
            next(model.parameters()).add_(.1);torch.testing.assert_close(cache.forward(model,**inputs),model(**inputs),atol=0,rtol=0)
        self.assertEqual(cache.stats()['misses'],3);self.assertEqual(cache.stats()['invalidations'],1)
        snapshot=FrozenPolicies(self.model(),self.model(PolicyNet),version=3,average_ready=True);saved=deepcopy(snapshot.state_dict())
        with torch.no_grad():
            original=snapshot.forward('q',**inputs);next(snapshot.q.parameters()).add_(.2);snapshot.forward('q',**inputs);snapshot.load_state_dict(saved);torch.testing.assert_close(snapshot.forward('q',**inputs),original,atol=0,rtol=0)
        self.assertEqual(set(snapshot.state_dict()),{'q','average','version','average_ready'})

    def test_gradients_hooks_other_shapes_and_autocast_keep_eager_behavior(self):
        model=self.model();cache=CPU3x3ForwardCache();inputs=self.inputs();inputs['spatial'].requires_grad_(True);value=cache.forward(model,**inputs);value.sum().backward();self.assertIsNotNone(inputs['spatial'].grad)
        calls=[];hook=model.register_forward_hook(lambda *args:calls.append(1))
        with torch.no_grad():cache.forward(model,**inputs);cache.forward(model,**inputs)
        hook.remove();self.assertEqual(len(calls),2)
        with torch.no_grad():
            for args in (self.inputs(size=9),self.inputs(batch=2)):
                torch.testing.assert_close(cache.forward(model,**args),model(**args),atol=0,rtol=0)
            cache.forward(model,**inputs)
            with torch.autocast('cpu',dtype=torch.bfloat16):torch.testing.assert_close(cache.forward(model,**inputs),model(**inputs),atol=0,rtol=0)
        self.assertEqual(cache.stats()['hits'],0);self.assertGreaterEqual(cache.stats()['fallbacks'],6)

    def test_lru_and_weak_model_ownership_bound_memory(self):
        cache=CPU3x3ForwardCache(max_entries_per_model=2);model=self.model()
        with torch.no_grad():
            for _ in range(4):cache.forward(model,**self.inputs())
        self.assertEqual(cache.stats()['resident_entries'],2);self.assertEqual(cache.stats()['evictions'],2);del model;gc.collect();self.assertEqual(cache.stats()['resident_models'],0)

    def test_actual_actor_actions_labels_probabilities_and_private_rng_match_eager(self):
        q=self.model();average=self.model(PolicyNet);a=FrozenPolicies(q,average,version=9,average_ready=True);b=FrozenPolicies(q,average,version=9,average_ready=True);b._cpu_forward_cache=None
        states=[State(Rules(3,6))];states.append(resolve(states[0],0,1).state)
        for acting,br in [('q',True),('q',False),('average',False)]:
            choice=EpisodeChoice(8000000,br,acting,.08,.5,.01 if br else .8,.8);x=EpisodeActor(a,choice,np.random.default_rng(31));y=EpisodeActor(b,choice,np.random.default_rng(31))
            for _ in range(8):
                for state in states:
                    for role in (BLACK,WHITE):
                        obs=encode(state,role);left=x.commit(obs,label_eligible=br);right=y.commit(obs,label_eligible=br);self.assertEqual((left.action,left.probability,left.policy_version),(right.action,right.probability,right.policy_version))
                        if left.greedy is not None:np.testing.assert_array_equal(left.greedy,right.greedy)
            self.assertEqual(x.rng.bit_generator.state,y.rng.bit_generator.state)
        self.assertGreater(a._cpu_forward_cache.stats()['hits'],0)


if __name__=='__main__':unittest.main()

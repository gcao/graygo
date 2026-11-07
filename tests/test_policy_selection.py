import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from graygo.confidence import canonical_sha
from graygo.engine import BLACK,Rules,State
from graygo.evaluation import FrozenNeuralPolicy
from graygo.models import ModelConfig,PolicyNet
from tools.policy_selection import select_weights,selection_spec,SelectedAveragePolicy,apply_selection


class SelectionTests(unittest.TestCase):
    def test_greedy_placements_preserve_pass_and_legal_support(self):
        result=select_weights([.3,0.,.2,.3,.2],selection_spec())
        np.testing.assert_allclose(result,[.4,0.,0.,.4,.2],rtol=0,atol=1e-15)
        self.assertAlmostEqual(sum(result),1)
    def test_empty_placement_domain_forces_pass(self):
        self.assertEqual(select_weights([0.,0.,1.],selection_spec()),[0.,0.,1.])
    def test_temperature_one_preserves_distribution(self):
        np.testing.assert_allclose(select_weights([.1,.4,.2,.3],selection_spec(1.)),[.1,.4,.2,.3],atol=1e-15)
    def test_soft_placement_sharpening_keeps_pass_marginal(self):
        p=select_weights([.6,.2,.2],selection_spec(.5))
        self.assertAlmostEqual(p[-1],.2);self.assertGreater(p[0],.6)
    def test_joint_greedy_would_distort_pass_but_placement_mode_does_not(self):
        p=[.9726/81]*81+[.0274]
        self.assertAlmostEqual(select_weights(p,selection_spec())[-1],.0274)
        self.assertEqual(select_weights(p,selection_spec(0.,False))[-1],1.)
    def test_tolerance_stabilizes_symmetric_placements(self):
        p=select_weights([.4,.4*(1-1e-7),.2],selection_spec())
        self.assertAlmostEqual(p[0],p[1])
    def test_bad_inputs_and_mutation_are_rejected(self):
        for p in ([0.,0.],[math.nan,1.],[-1.,2.]):
            with self.assertRaises(ValueError):select_weights(p,selection_spec())
        for temperature in (-1.,math.inf,True):
            with self.assertRaises(ValueError):selection_spec(temperature)
    def test_probability_transform_is_equivariant_under_placement_permutations(self):
        rng=np.random.default_rng(9)
        for temperature in (0.,.25,.5,1.):
            p=rng.random(82);p[3]=0.;permutation=rng.permutation(81)
            actual=select_weights(list(p[permutation])+[p[-1]],selection_spec(temperature))
            expected=np.asarray(select_weights(p,selection_spec(temperature)))
            np.testing.assert_allclose(actual,list(expected[permutation])+[expected[-1]],atol=1e-15)
    def test_frozen_policy_and_sidecar_round_trip(self):
        torch.set_num_threads(1)
        base=FrozenNeuralPolicy(PolicyNet(ModelConfig(8,1,2)),label='fixture',origin={'rules':{'size':3,'max_rounds':5,'collision':'gray'}})
        selection=selection_spec();policy=SelectedAveragePolicy(base,selection);state=State(Rules(3,5))
        before=base._weight_sha();p=policy.initial_distribution(state,BLACK,fresh_episode=True)
        self.assertAlmostEqual(p[-1],base.initial_distribution(state,BLACK,fresh_episode=True)[-1])
        policy.assert_frozen();self.assertEqual(before,base._weight_sha())
        with tempfile.TemporaryDirectory() as directory:
            import tools.policy_selection as implementation
            import shutil
            path=Path(directory)/'policy.pt';shutil.copyfile(implementation.__file__,path.with_name('policy_selection.py'))
            restored=apply_selection(base,selection,path)
            self.assertEqual(restored.sha256,policy.sha256)
            np.testing.assert_allclose(restored.initial_distribution(state,BLACK),p,atol=0,rtol=0)
        policy.selection['temperature']=.5
        with self.assertRaises(ValueError):policy.assert_frozen()


if __name__=='__main__':unittest.main()

"""Graph execution is opt-in; weights, probabilities and clocks stay exact."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from graygo.board_runtime import BoardNFSPRuntime,BoardRuntimeConfig,configure_runtime
from graygo.encoding import encode
from graygo.engine import BLACK,WHITE,State,resolve_reference
from graygo.evaluation import FrozenNeuralPolicy,BaselinePolicy,evaluate_pair
from graygo.frozen_forward import FrozenForwardPool,execution_identity
from graygo.models import QNet,PolicyNet
from graygo.nfsp_collection import FrozenPolicies,EpisodeActor,commit_many
from graygo.rule_state import RuleObservation
from graygo.schedules import EpisodeChoice
from graygo.pilot_gates import PilotProtocol,ScreeningProtocol,DEVELOPMENT_SEEDS


class FrozenExecutionContractTests(unittest.TestCase):
    def test_cpu_cannot_select_cuda_and_default_snapshot_schema_is_unchanged(self):
        with self.assertRaises(ValueError):BoardRuntimeConfig(cuda_graphs=True)
        with self.assertRaises(ValueError):FrozenForwardPool('cpu')
        self.assertIsNone(execution_identity(False))
        for value in (None,1,'yes'):
            with self.assertRaises(ValueError):execution_identity(value)
        q=QNet();p=PolicyNet();snapshot=FrozenPolicies(q,p,version=1,average_ready=False)
        self.assertEqual(set(snapshot.state_dict()),{'q','average','version','average_ready'})

    def test_graph_cohort_settings_roundtrip_including_scalar_batch_and_device(self):
        ids={seed:f'{seed:064x}' for seed in DEVELOPMENT_SEEDS}
        for batch in (1,32):
            settings=dict(inference_batch_size=batch,pair_batch_size=16,inference_device='cuda',cuda_graphs=True)
            pilot=PilotProtocol('graph','a'*64,'b'*64,ids,**settings)
            self.assertEqual(PilotProtocol.from_dict(pilot.to_dict()),pilot)
            screen=ScreeningProtocol('graph','a'*64,'b'*64,ids,**settings)
            self.assertEqual(ScreeningProtocol.from_dict(screen.to_dict()).to_dict(),screen.to_dict())
            bad=deepcopy(pilot.to_dict());bad['evaluation_execution']['cuda_graphs']=False
            with self.assertRaises(ValueError):PilotProtocol.from_dict(bad)
        with self.assertRaises(ValueError):PilotProtocol('bad','a'*64,'b'*64,ids,cuda_graphs=True)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class FrozenCUDAForwardTests(unittest.TestCase):
    def setUp(self):configure_runtime('cuda',True);torch.set_num_threads(1)

    def inputs(self,batch):
        states=[State()];rng=np.random.default_rng(63)
        for _ in range(3):
            s=states[-1];states.append(resolve_reference(s,int(rng.choice(s.legal_actions(BLACK))),int(rng.choice(s.legal_actions(WHITE)))).state)
        observations=[encode(states[i%4],i%2+1) for i in range(batch)]
        return torch.tensor(np.stack([o.spatial for o in observations]),device='cuda',dtype=torch.float32),torch.tensor(np.stack([o.global_features for o in observations]),device='cuda')

    def model(self,kind,seed):
        torch.manual_seed(seed);return kind().to('cuda').eval().requires_grad_(False)

    def test_shared_templates_match_multiple_full_weight_versions_exactly_and_preserve_rng(self):
        pool=FrozenForwardPool('cuda');models=[self.model(kind,seed) for kind in (QNet,PolicyNet) for seed in (21,22,23)]
        cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
        for batch in (1,2,32):
            x,g=self.inputs(batch)
            for model in models:
                for offset in (0.,.125):
                    expected=model(x+offset,g-offset);actual=pool.forward(model,x+offset,g-offset)
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        self.assertEqual(pool.stats()['captures'],6);self.assertEqual(pool.stats()['weight_packs'],6)
        self.assertEqual(pool.stats()['resident_templates'],6);self.assertEqual(pool.stats()['eager_fallbacks'],0)
        torch.testing.assert_close(torch.get_rng_state(),cpu_rng,atol=0,rtol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(),cuda_rng,atol=0,rtol=0)

    def test_output_is_not_aliased_and_explicit_restore_rebuilds_weight_pack(self):
        pool=FrozenForwardPool('cuda');a=self.model(QNet,1);b=self.model(QNet,2);x,g=self.inputs(1)
        original=pool.forward(a,x,g);saved=original.clone();pool.forward(b,x+1,g)
        torch.testing.assert_close(original,saved,atol=0,rtol=0)
        with torch.no_grad():next(a.parameters()).add_(.01)
        with self.assertRaisesRegex(ValueError,'parameters changed'):pool.forward(a,x,g)
        pool.invalidate(a);torch.testing.assert_close(pool.forward(a,x,g),a(x,g),atol=0,rtol=0)
        self.assertEqual(pool.stats()['captures'],1)

    def test_runtime_context_and_live_training_models_cannot_enter_capture(self):
        pool=FrozenForwardPool('cuda');model=self.model(QNet,8);x,g=self.inputs(1);pool.forward(model,x,g)
        old=torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cudnn.allow_tf32=not old
            with self.assertRaisesRegex(ValueError,'runtime/stream changed'):pool.forward(model,x,g)
        finally:torch.backends.cudnn.allow_tf32=old
        model.train()
        with self.assertRaisesRegex(ValueError,'immutable'):pool.forward(model,x,g)

    def test_other_shapes_use_recorded_eager_path_without_growing_capture_inventory(self):
        pool=FrozenForwardPool('cuda');model=self.model(QNet,9);x,g=self.inputs(33)
        torch.testing.assert_close(pool.forward(model,x,g),model(x,g),atol=0,rtol=0)
        self.assertEqual(pool.stats()['eager_fallbacks'],1);self.assertEqual(pool.stats()['captures'],0)

    def test_sixteen_private_snapshot_versions_keep_probabilities_labels_and_rng_exact(self):
        pool=FrozenForwardPool('cuda');q=self.model(QNet,51);average=self.model(PolicyNet,52)
        state=resolve_reference(State(),0,1).state;eager=[];graphed=[]
        for version in range(16):
            with torch.no_grad():next(q.parameters()).add_(.0001);next(average.parameters()).sub_(.0001)
            a=FrozenPolicies(q,average,version=version,average_ready=True)
            b=FrozenPolicies(q,average,version=version,average_ready=True,forward_pool=pool)
            for role in (BLACK,WHITE):
                choice=EpisodeChoice(7000000,role==BLACK,'q' if role==BLACK else 'average',.08,1.,0. if role==BLACK else .9,.9)
                x=EpisodeActor(a,choice,np.random.default_rng(version*2+role));y=EpisodeActor(b,choice,np.random.default_rng(version*2+role))
                eager.append((x,RuleObservation(state,role),role==BLACK));graphed.append((y,RuleObservation(state,role),role==BLACK))
        expected=commit_many(eager);actual=commit_many(graphed)
        for left,right,(a,_,_),(b,_,_) in zip(expected,actual,eager,graphed):
            self.assertEqual((left.action,left.probability,left.policy_version),(right.action,right.probability,right.policy_version))
            if left.greedy is not None:np.testing.assert_array_equal(left.greedy,right.greedy)
            self.assertEqual(a.rng.bit_generator.state,b.rng.bit_generator.state)
        self.assertEqual(pool.stats()['captures'],2);self.assertEqual(pool.stats()['weight_packs'],32)

    def test_probe_marginals_and_complete_baseline_games_match_eager_selected_policy(self):
        for representation,kind in [('average',PolicyNet),('greedy_q',QNet)]:
            model=self.model(kind,104);a=FrozenNeuralPolicy(model,representation=representation,inference_batch_size=32)
            b=FrozenNeuralPolicy(model,representation=representation,inference_batch_size=32,cuda_graphs=True)
            self.assertNotEqual(a.sha256,b.sha256)
            root=State()
            for state in (root,resolve_reference(root,0,1).state):
                for role in (BLACK,WHITE):np.testing.assert_array_equal(a.initial_distribution(state,role),b.initial_distribution(state,role))
            opponent=BaselinePolicy('random')
            from json import loads
            path=Path(__file__).resolve().parents[1]/'docs/verification/probe-multicell-v2-construction/benchmark.json'
            dense=State.from_dict(loads(path.read_text())['cases'][0]['before']['root'])
            self.assertEqual(evaluate_pair(a,opponent,dense,unit_seed=99),evaluate_pair(b,opponent,dense,unit_seed=99))
            b.assert_frozen();b.cuda_graphs=False
            with self.assertRaises(ValueError):b.assert_frozen()

    def test_full_runtime_graph_mode_preserves_optimizer_buffers_and_checkpoint_continuation(self):
        base=BoardRuntimeConfig(device='cuda',envs=2,capacity=32,batch=8,learning_start=8,target_every=16,seed=28)
        eager=BoardNFSPRuntime(base);eager.run_profile(32,time_cap_seconds=120)
        graphed=BoardNFSPRuntime(replace(base,cuda_graphs=True));graphed.run_profile(16,time_cap_seconds=120)
        with tempfile.TemporaryDirectory() as directory:
            saved=graphed.save(directory,reserve_bytes=0);resumed=BoardNFSPRuntime(replace(base,cuda_graphs=True));resumed.restore(saved)
            resumed.run_profile(32,time_cap_seconds=120)
        for name in ('q','target','average'):
            for key,value in getattr(eager.backend,name).state_dict().items():torch.testing.assert_close(value,getattr(resumed.backend,name).state_dict()[key],atol=0,rtol=0)
        self.assertEqual(eager.backend.counters,resumed.backend.counters)
        for left,right in ((eager.replay,resumed.replay),(eager.reservoir,resumed.reservoir)):
            for name,array in left.arrays.items():np.testing.assert_array_equal(array,right.arrays[name])
        self.assertEqual(eager.collector.state_dict()['active'],resumed.collector.state_dict()['active'])
        self.assertGreater(resumed.forward_pool.stats()['replays'],0)


if __name__=='__main__':unittest.main()

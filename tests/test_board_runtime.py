from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from graygo.board_augmentation import augment,permutations
from graygo.board_collection import BoardCollector
from graygo.board_runtime import BoardRuntimeConfig,BoardNFSPRuntime
from graygo.buffers import ReplayBuffer,ReservoirBuffer
from graygo.encoding import encode
from graygo.engine import BLACK,WHITE,Rules,State,resolve,transform,transform_action
from graygo.models import QNet,PolicyNet,ModelConfig
from graygo.nfsp_collection import FrozenPolicies,EpisodeActor,commit_many
from graygo.rule_state import RuleObservation,RuleStateSpec
from graygo.schedules import LearnerSchedule,EpisodeChoice


class BoardRuntimeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(73)
    def models(self):return QNet(ModelConfig(8,1,2)),PolicyNet(ModelConfig(8,1,2))
    def config(self,device='cpu'):return BoardRuntimeConfig(rules=Rules(3,5) if device=='cpu' else Rules(),envs=4,seed=3,batch=8,learning_start=8,
        capacity=64,target_every=16,width=8,blocks=1,norm_groups=2,reservoir_open=0,handoff_end=8,device=device)

    def test_batched_public_actions_match_scalar_private_sampling(self):
        q,p=self.models();snapshot=FrozenPolicies(q,p,version=0,average_ready=True)
        state=resolve(State(Rules(3,6)),0,1).state;requests=[];scalar=[]
        for index,kind in enumerate(('q','average','uniform','q')):
            choice=EpisodeChoice(0,kind=='q',kind,.1,0.,.05 if kind=='q' else .5,.5)
            obs=RuleObservation(state,BLACK if index%2==0 else WHITE)
            actor=EpisodeActor(snapshot,choice,np.random.default_rng(index))
            other=EpisodeActor.restore(snapshot,actor.state_dict())
            requests.append((actor,obs,kind=='q'));scalar.append(other.commit(obs,label_eligible=kind=='q'))
        batched=commit_many(requests)
        for a,b in zip(scalar,batched):
            self.assertEqual(a.action,b.action);self.assertAlmostEqual(a.probability,b.probability,places=6)
            if a.greedy is not None:np.testing.assert_array_equal(a.greedy,b.greedy)

    def test_collector_resume_preserves_partial_environments_old_versions_and_rngs(self):
        rules=Rules(3,5);q,p=self.models();old=FrozenPolicies(q,p,version=0,average_ready=False)
        with torch.no_grad():next(q.parameters()).add_(.1)
        current=FrozenPolicies(q,p,version=1,average_ready=False)
        spec=RuleStateSpec(rules);a=BoardCollector(rules,4,5);replay=ReplayBuffer(128,spec);reservoir=ReservoirBuffer(128,spec)
        schedule=LearnerSchedule(40_000_000,0,8);rng=np.random.default_rng(4)
        for _ in range(3):a.collect(8,old,replay,reservoir,rng,schedule)
        saved=deepcopy(a.state_dict());b=BoardCollector(rules,4,5);b.load_state_dict(saved,q,p)
        rr=deepcopy(replay);rv=deepcopy(reservoir);other=np.random.default_rng();other.bit_generator.state=deepcopy(rng.bit_generator.state)
        for _ in range(8):a.collect(8,current,replay,reservoir,rng,schedule);b.collect(8,current,rr,rv,other,schedule)
        for key,value in replay.arrays.items():np.testing.assert_array_equal(value,rr.arrays[key])
        for key,value in reservoir.arrays.items():np.testing.assert_array_equal(value,rv.arrays[key])
        self.assertEqual(a.metrics,b.metrics);self.assertEqual(a.cursor,b.cursor);self.assertEqual(a.records,b.records)
        for left,right in zip(a.active,b.active):
            self.assertEqual(None if left is None else left.state,None if right is None else right.state)

    def check_runtime_resume(self,device,config=None):
        config=config or self.config(device);full=BoardNFSPRuntime(config);a=full.run_profile(64,time_cap_seconds=60)
        part=BoardNFSPRuntime(config);part.run_profile(24,time_cap_seconds=60)
        self.assertTrue(any(e is not None for e in part.collector.active))
        with tempfile.TemporaryDirectory() as directory:
            path=part.save(directory,reserve_bytes=0);restored=BoardNFSPRuntime(config);restored.restore(path)
            b=restored.run_profile(64,time_cap_seconds=60)
            for name in ('q','target','average'):
                for key,value in getattr(full.backend,name).state_dict().items():torch.testing.assert_close(value,getattr(restored.backend,name).state_dict()[key],rtol=0,atol=0)
            for key,value in full.replay.arrays.items():np.testing.assert_array_equal(value,restored.replay.arrays[key])
            for key,value in full.reservoir.arrays.items():np.testing.assert_array_equal(value,restored.reservoir.arrays[key])
            self.assertEqual(full.collector.metrics,restored.collector.metrics)
            for key,value in full.rng.items():self.assertEqual(value.bit_generator.state,restored.rng[key].bit_generator.state)
            self.assertEqual(a['counters'],b['counters']);self.assertFalse(a['formal_learning_run']);self.assertFalse(a['a2_complete'])
        with self.assertRaises(ValueError):full.run_profile(1_000_000,time_cap_seconds=60)

    def test_complete_cpu_runtime_resume_is_bit_exact_with_augmentation(self):self.check_runtime_resume('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_complete_cuda_runtime_resume_is_bit_exact(self):self.check_runtime_resume('cuda')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_full_network_cuda_resume_is_bit_exact(self):
        self.check_runtime_resume('cuda',replace(self.config('cuda'),width=64,blocks=8,norm_groups=8))

    def test_hardware_profile_cannot_turn_into_an_unbounded_training_run(self):
        from graygo.board_profile import profile_board
        config=BoardRuntimeConfig()
        with tempfile.TemporaryDirectory() as directory:
            for records,seconds in ((65536,60),(8192,3601),(8192,float('inf')),(1,60)):
                with self.assertRaises(ValueError):profile_board(config,directory,records=records,time_cap_seconds=seconds)
            with self.assertRaises(ValueError):profile_board(replace(config,width=8),directory,records=8192,time_cap_seconds=60)
            self.assertFalse((Path(directory)/'plan.json').exists())

    def test_augmented_transitions_and_frozen_labels_match_engine_symmetry(self):
        rules=Rules(3,8);state=resolve(resolve(State(rules),0,1).state,2,3).state
        action=state.legal_actions(BLACK)[0];other=state.legal_actions(WHITE)[-1];next_state=resolve(state,action,other).state
        a,b=encode(state,BLACK),encode(next_state,BLACK);labels=a.legal.astype(np.float32)/a.legal.sum()
        data={'spatial':a.spatial[None],'globals':a.global_features[None],'legal':a.legal[None],
            'next_spatial':b.spatial[None],'next_globals':b.global_features[None],'next_legal':b.legal[None],
            'action':np.array([action]),'targets':labels[None],'behavior_prob':np.array([.25]),'reward':np.array([0.]),'valid':np.array([True])}
        table,_=permutations(3)
        for rotation in range(4):
            for reflect in (False,True):
                params={'rotation':rotation,'reflect':reflect,'shift':(1,2)}
                expected=np.array([transform_action(p,rules,**params) for p in range(rules.points)])
                choice=int(np.flatnonzero(np.all(table==expected,axis=1))[0]);out=augment(data,np.random.default_rng(1),choices=[choice])
                x=encode(transform(state,**params),BLACK);y=encode(transform(next_state,**params),BLACK)
                np.testing.assert_array_equal(out['spatial'][0],x.spatial);np.testing.assert_array_equal(out['next_spatial'][0],y.spatial)
                np.testing.assert_array_equal(out['legal'][0],x.legal);np.testing.assert_array_equal(out['next_legal'][0],y.legal)
                self.assertEqual(out['action'][0],transform_action(action,rules,**params))
                self.assertEqual(out['behavior_prob'][0],.25);self.assertEqual(out['targets'][0,-1],labels[-1])
                for p in range(rules.points):self.assertEqual(out['targets'][0,expected[p]],labels[p])

    def test_sequence_augmentation_keeps_context_frame_and_padding(self):
        rules=Rules(3,4);o=encode(State(rules),BLACK)
        data={'spatial':np.stack([o.spatial,np.zeros_like(o.spatial)])[None],
              'next_spatial':np.stack([o.spatial,np.zeros_like(o.spatial)])[None],
              'legal':np.stack([o.legal,np.zeros_like(o.legal)])[None],'next_legal':np.stack([o.legal,np.zeros_like(o.legal)])[None],
              'action':np.array([[0,0]]),'valid':np.array([[True,False]]),'behavior_prob':np.array([[.1,1.]])}
        out=augment(data,np.random.default_rng(7));self.assertEqual(out['spatial'].shape,(1,2,30,3,3))
        self.assertFalse(out['spatial'][0,1].any());self.assertFalse(out['legal'][0,1].any());self.assertEqual(out['action'][0,1],0)
        np.testing.assert_array_equal(out['behavior_prob'],data['behavior_prob'])

if __name__=='__main__':unittest.main()

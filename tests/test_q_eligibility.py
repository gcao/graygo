from copy import deepcopy
from dataclasses import asdict,replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from graygo.board_runtime import configure_runtime
from graygo.buffers import ObservationSpec,ReplayBuffer,ReservoirBuffer
from graygo.encoding import Observation
from graygo.engine import BLACK,WHITE
from graygo.nfsp import NFSPBackend,NFSPConfig,NFSPCadence,OneStepBatch,one_step_targets,update_q
from graygo.nfsp_collection import FrozenPolicies,EpisodeActor,JointEpisode
from graygo.schedules import LearnerSchedule
from graygo.task_models import CompactQ,CompactPolicy
from graygo.task_training import NFSPTaskConfig,NFSPTaskTrainer
from graygo.tasks import DelayedGame


class ExactQ(torch.nn.Module):
    def __init__(self,game):
        super().__init__();values=torch.zeros(game.spec.spatial_shape[0],game.spec.actions)
        for (key,role),index in game.encoding.ids.items():
            _,controller,depth,branch=key
            if depth==0:
                if role==controller:values[index,0]=1.;values[index,1]=-1.
                else:values[index,0]=-1.
            else:values[index,:]=(1 if branch==0 else -1)*(1 if role==controller else -1)
        self.values=torch.nn.Parameter(values)
    def forward(self,spatial,globals):return spatial@self.values


class Indices:
    def __init__(self,indices):self.indices=np.asarray(indices)
    def integers(self,low,high,size):
        assert low==0 and len(self.indices)==size and (self.indices<high).all()
        return self.indices.copy()


def exact(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:exact(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


class QEligibilityTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(719)

    def test_stationary_passive_root_is_not_pushed_toward_intervention_mixture(self):
        for controller in (BLACK,WHITE):
            with self.subTest(controller=controller):
                game=DelayedGame(horizons=(5,),equivalent_actions=40,codebook_seed=719)
                q=ExactQ(game);target=deepcopy(q);snapshot=FrozenPolicies(q,deepcopy(q),version=0,average_ready=False)
                replay=ReplayBuffer(64,game.spec);reservoir=ReservoirBuffer(64,game.spec,opening_record=5000000)
                schedule=LearnerSchedule.a2();rng=np.random.default_rng(719);last=None
                for episode_id,forced in enumerate([None]*8+[0,1]):
                    actors=[]
                    for role in (BLACK,WHITE):
                        choice=replace(schedule.choose_episode(replay.insertions,rng,average_ready=False),epsilon=0.)
                        actors.append(EpisodeActor(snapshot,choice,np.random.default_rng(1000+10*episode_id+role)))
                    last=JointEpisode(game,game.root(5,controller),*actors,episode_id=episode_id,
                                      supplemental=forced is not None,forced_root=None if forced is None else (controller,forced))
                    last.step(replay,reservoir,rng,schedule,record=replay.insertions)
                passive=3-controller;physical=np.arange(1 if passive==WHITE else 0,20,2)
                raw=replay.sample(10,Indices(physical))
                with self.assertRaisesRegex(ValueError,'ineligible replay'):OneStepBatch.from_replay(raw,'cpu')
                eligible=np.flatnonzero(replay.arrays['value_mask'][:replay.size]);selected=physical[:8]
                data=replay.sample_q(8,Indices(np.searchsorted(eligible,selected)))
                batch=OneStepBatch.from_replay(data,'cpu');labels,_=one_step_targets(q,target,batch,0.)
                torch.testing.assert_close(labels,-torch.ones(8),rtol=0,atol=0)
                before=q.values.detach().clone()
                stats=update_q(q,target,torch.optim.SGD(q.parameters(),lr=.1),batch,epsilon=0.,max_grad_norm=1.)
                self.assertEqual(stats['loss'],0.);torch.testing.assert_close(q.values,before,rtol=0,atol=0)
                own=np.arange(0 if controller==BLACK else 1,20,2)
                data=replay.sample_q(10,Indices(np.searchsorted(eligible,own)))
                labels,_=one_step_targets(q,target,OneStepBatch.from_replay(data,'cpu'),0.)
                self.assertEqual({int(a):float(labels[torch.tensor(data['action']==a)].mean()) for a in (0,1)},{0:1.,1:-1.})
                last.step(replay,reservoir,rng,schedule,record=replay.insertions)
                self.assertTrue(replay.arrays['value_mask'][20:22].all());self.assertEqual(reservoir.seen,0)
                self.assertEqual((replay.insertions,replay.q_insertions),(22,20))

    def test_invalid_record_at_an_aligned_boundary_does_not_repeat_update_or_refresh(self):
        config=NFSPConfig(batch=4,learning_start=4,target_every=8,q_reuse=4)
        backend=NFSPBackend(CompactQ(2,0,2,8),CompactPolicy(2,0,2,8),config)
        spec=ObservationSpec((2,),0,2,'test/eligibility/1');replay=ReplayBuffer(16,spec);reservoir=ReservoirBuffer(16,spec,opening_record=100)
        obs=Observation(np.array([1,0]),np.empty(0),np.ones(2,dtype=bool));terminal=Observation(np.zeros(2),np.empty(0),np.zeros(2,dtype=bool))
        reports=[];rng=np.random.default_rng(17)
        for record,valid in enumerate((1,1,1,0,1,0,1,1,1,1)):
            replay.add(obs,0,1.,terminal,True,behavior_prob=.5,stream=record,step=0,value_mask=bool(valid),actor_mask=bool(valid),supplemental=not valid)
            reports.append(backend.learn_after_fresh(1,valid,replay,reservoir,rng,epsilon=.1,q_eligible=valid))
        self.assertEqual([len(r['q']) for r in reports],[0,0,0,0,4,0,0,0,0,4])
        self.assertEqual([i for i,r in enumerate(reports) if r['target_refreshed']],[9])
        self.assertEqual((backend.counters.fresh_records,backend.counters.fresh_q,backend.counters.fresh_ordinary),(10,8,8))
        self.assertEqual((backend.counters.q_updates,backend.counters.q_loss_targets,backend.counters.target_refreshes),(8,32,1))
        with self.assertRaises(ValueError):NFSPCadence(fresh_records=1,fresh_q=2).validate()

    def test_replay_overwrite_persistence_uniform_q_membership_and_rng(self):
        spec=ObservationSpec((2,),0,2,'test/eligibility/1');replay=ReplayBuffer(5,spec)
        obs=Observation(np.array([1,0]),np.empty(0),np.ones(2,dtype=bool));terminal=Observation(np.zeros(2),np.empty(0),np.zeros(2,dtype=bool))
        flags=[True,False,True,False,False,True,True,False,True,False,True,True]
        for record,valid in enumerate(flags):replay.add(obs,0,0.,terminal,True,behavior_prob=.5,stream=record,step=0,value_mask=valid)
        self.assertEqual(replay.q_insertions,sum(flags));self.assertEqual(replay.q_size,sum(flags[-5:]))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'replay.npz';replay.save(path);restored=ReplayBuffer.load(path)
            self.assertEqual((restored.q_size,restored.q_insertions),(replay.q_size,replay.q_insertions))
            rng=np.random.default_rng(37);other=np.random.default_rng(37)
            a=replay.sample_q(1000,rng);b=restored.sample_q(1000,other)
            for key,value in a.items():np.testing.assert_array_equal(value,b[key])
            self.assertTrue(a['value_mask'].all());self.assertEqual(set(a['ordinal']),{8,10,11})
            self.assertEqual(rng.bit_generator.state,other.bit_generator.state)

    def test_q_sequence_context_stops_before_opponent_intervention(self):
        spec=ObservationSpec((3,),0,1,'test/eligibility/sequence');replay=ReplayBuffer(8,spec)
        observations=[Observation(np.eye(3)[i],np.empty(0),np.array([True])) for i in range(3)]
        terminal=Observation(np.zeros(3),np.empty(0),np.array([False]))
        for i in range(3):replay.add(observations[i],0,0.,terminal if i==2 else observations[i+1],i==2,
            behavior_prob=1.,stream=1,step=i,value_mask=i!=1)
        start=replay.sample_sequences(1,3,Indices([0]),q_only=True)
        self.assertEqual(start['valid'].tolist(),[[True,False,False]])
        self.assertFalse(start['terminated'][0,0])
        last=replay.sample_sequences(1,3,Indices([1]),q_only=True)
        self.assertEqual(last['ordinal'][0,0],2);self.assertTrue(last['terminated'][0,0])

    def test_wrapped_legacy_history_stays_unknown_and_cannot_silently_resume(self):
        spec=ObservationSpec((2,),0,1,'test/legacy/eligibility');replay=ReplayBuffer(3,spec)
        obs=Observation(np.array([1,0]),np.empty(0),np.array([True]));terminal=Observation(np.zeros(2),np.empty(0),np.array([False]))
        for record in range(7):replay.add(obs,0,0.,terminal,True,behavior_prob=1.,stream=record,step=0,value_mask=record%2==0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'legacy.npz';replay.save(path)
            with np.load(path,allow_pickle=False) as data:
                meta=json.loads(str(data['metadata'].item()));arrays={k:data[k].copy() for k in data.files if k!='metadata'}
            for key in ('eligibility_version','q_insertions','q_size'):meta.pop(key)
            np.savez_compressed(path,metadata=np.array(json.dumps(meta)),**arrays)
            old=ReplayBuffer.load(path);self.assertIsNone(old.q_insertions);self.assertEqual(old.q_size,2)
            self.assertTrue(old.sample_q(20,np.random.default_rng(1))['value_mask'].all())
            before=old.insertions
            with self.assertRaisesRegex(ValueError,'unknown historical Q eligibility'):
                old.add(obs,0,0.,terminal,True,behavior_prob=1.,stream=8,step=0)
            self.assertEqual(old.insertions,before)
        config=NFSPConfig(batch=4,learning_start=4,target_every=8)
        backend=NFSPBackend(CompactQ(2,0,1,8),CompactPolicy(2,0,1,8),config)
        state=backend.state_dict();state.pop('clock_version')
        with self.assertRaisesRegex(ValueError,'original source'):backend.load_state_dict(state)

    def test_all_eligible_recordwise_and_paired_updates_match_exactly(self):
        cfg=NFSPConfig(batch=4,learning_start=4,target_every=8,q_reuse=4);spec=ObservationSpec((2,),0,2,'test/equivalence')
        a=NFSPBackend(CompactQ(2,0,2,8),CompactPolicy(2,0,2,8),cfg)
        b=NFSPBackend(CompactQ(2,0,2,8),CompactPolicy(2,0,2,8),cfg);b.load_state_dict(deepcopy(a.state_dict()))
        ra,rb=ReplayBuffer(32,spec),ReplayBuffer(32,spec);sa,sb=ReservoirBuffer(32,spec),ReservoirBuffer(32,spec)
        rng_a,rng_b=np.random.default_rng(7),np.random.default_rng(7)
        terminal=Observation(np.zeros(2),np.empty(0),np.zeros(2,dtype=bool))
        for record in range(16):
            obs=Observation(np.eye(2)[record%2],np.empty(0),np.ones(2,dtype=bool))
            for replay in (ra,rb):replay.add(obs,record%2,1. if record%2 else -1.,terminal,True,behavior_prob=.5,stream=record,step=0)
            a.learn_after_fresh(1,1,ra,sa,rng_a,epsilon=.1)
            if record%2:b.learn_after_fresh(2,2,rb,sb,rng_b,epsilon=.1)
        exact(a.state_dict(),b.state_dict());self.assertEqual(rng_a.bit_generator.state,rng_b.bit_generator.state)

    def check_odd_resume(self,device):
        config=NFSPTaskConfig(time_cap_seconds=300,task='delayed',learner='neural',seed=3,records=120,eval_every=24,
            batch=4,learning_start=4,capacity=128,target_every=8,reservoir_open=0,handoff_end=8,epsilon_decay=40,
            horizons=(3,),equivalent_actions=3,hidden=8,width=8,blocks=1,norm_groups=2,purpose='profile',device=device)
        if device=='cuda':configure_runtime('cuda',True)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);full=NFSPTaskTrainer(config,root/'full',reserve_bytes=0);full.run(verbose=False)
            part=NFSPTaskTrainer(config,root/'part',reserve_bytes=0)
            for stop in range(2,62,2):
                part.run(until_records=stop,verbose=False)
                if part.backend.counters.fresh_q%2:break
            self.assertEqual(part.records%2,0);self.assertEqual(part.backend.counters.fresh_q%2,1)
            resumed=NFSPTaskTrainer(config,root/'part',resume=True,reserve_bytes=0);resumed.run(verbose=False)
            exact(full.backend.state_dict(),resumed.backend.state_dict())
            for name,array in full.replay.arrays.items():np.testing.assert_array_equal(array,resumed.replay.arrays[name])
            for name,array in full.reservoir.arrays.items():np.testing.assert_array_equal(array,resumed.reservoir.arrays[name])
            for name,array in full.metrics.items():np.testing.assert_array_equal(array,resumed.metrics[name])
            self.assertEqual((root/'full/episodes.bin').read_bytes(),(root/'part/episodes.bin').read_bytes())
            self.assertEqual(full.per_role_q,[58,58]);self.assertEqual(resumed.per_role_q,full.per_role_q)
            for key,rng in full.rng.items():self.assertEqual(rng.bit_generator.state,resumed.rng[key].bit_generator.state)

    def test_cpu_resume_with_odd_eligible_clock_preserves_complete_state(self):self.check_odd_resume('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_resume_with_odd_eligible_clock_preserves_complete_state(self):self.check_odd_resume('cuda')


if __name__=='__main__':unittest.main()

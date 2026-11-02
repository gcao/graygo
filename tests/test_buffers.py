import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from graygo.buffers import ObservationSpec,ReplayBuffer,ReservoirBuffer
from graygo.engine import BLACK,WHITE,Rules,State,resolve
from graygo.encoding import Observation,encode


def chain():
    s=State(Rules(3,4));states=[s]
    for b,w in ((0,1),(2,3),(4,5),(9,9)):
        s=resolve(s,b,w).state;states.append(s)
    return [encode(s,BLACK) for s in states]


def append(buffer,observations,*,stream=1,start=0,opponent=2):
    for step in range(start,len(observations)-1):
        obs,nxt=observations[step:step+2]
        action=int(np.flatnonzero(obs.legal)[0])
        buffer.add(obs,action,1. if not nxt.legal.any() else 0.,nxt,not bool(nxt.legal.any()),
                   behavior_prob=.2,stream=stream,step=step,opponent_version=opponent)


class BufferTests(unittest.TestCase):
    def test_lossless_pack_and_no_mutable_aliases(self):
        spec=ObservationSpec.board(3);obs=chain()[1]
        packed=spec.pack(obs)
        spatial,globals,legal=spec.unpack(*packed)
        np.testing.assert_array_equal(spatial,obs.spatial)
        np.testing.assert_array_equal(globals,obs.global_features)
        np.testing.assert_array_equal(legal,obs.legal)
        self.assertLess(spec.spatial_bytes,obs.spatial.nbytes)
        malformed=Observation(np.full((30,3,3),.5),obs.global_features,obs.legal)
        with self.assertRaises(ValueError):spec.pack(malformed)
        with self.assertRaises(ValueError):ReplayBuffer(500000,spec,max_bytes=100)

    def test_sequences_preserve_stream_boundaries_and_terminal(self):
        spec=ObservationSpec.board(3);r=ReplayBuffer(20,spec);obs=chain()
        # Interleaved independent player/episode streams.
        for step in range(4):
            for stream in (10,11):
                current,nxt=obs[step:step+2]
                r.add(current,int(np.flatnonzero(current.legal)[0]),float(step==3),nxt,step==3,
                      behavior_prob=.4,stream=stream,step=step,opponent_version=3)
        sample=r.sample_sequences(100,8,np.random.default_rng(4))
        self.assertEqual(sample['spatial'].shape,(100,8,30,3,3))
        for row,valid in enumerate(sample['valid']):
            length=int(valid.sum());self.assertGreater(length,0)
            self.assertTrue(sample['terminated'][row,length-1])
            self.assertTrue((sample['stream'][row,:length]==sample['stream'][row,0]).all())
            self.assertTrue((np.diff(sample['step'][row,:length])==1).all())
            self.assertTrue(sample['terminated'][row,length:].all())
            self.assertFalse(sample['legal'][row,length:].any())

    def test_wraparound_and_exact_resume_continue_links(self):
        spec=ObservationSpec.board(3);r=ReplayBuffer(5,spec);obs=chain()
        append(r,obs,stream=1)
        # New episode overwrites old slots; stale forward links must not be followed.
        r.add(obs[0],0,0.,obs[1],False,behavior_prob=.5,stream=2,step=0)
        r.add(obs[1],2,0.,obs[2],False,behavior_prob=.5,stream=2,step=1)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'replay.npz';r.save(path);loaded=ReplayBuffer.load(path)
            self.assertEqual((r.cursor,r.size,r.insertions),(loaded.cursor,loaded.size,loaded.insertions))
            for target in (r,loaded):
                target.add(obs[2],4,0.,obs[3],False,behavior_prob=.5,stream=2,step=2)
            a=r.sample_sequences(100,8,np.random.default_rng(1));b=loaded.sample_sequences(100,8,np.random.default_rng(1))
            for name in a:np.testing.assert_array_equal(a[name],b[name])
            for row,valid in enumerate(a['valid']):
                count=int(valid.sum())
                self.assertTrue((np.diff(a['ordinal'][row,:count])>0).all())
                self.assertTrue((a['stream'][row,:count]==a['stream'][row,0]).all())
        self.assertLessEqual(len(r._tails),r.capacity)

    def test_discontinuous_or_changed_opponent_does_not_link(self):
        spec=ObservationSpec.board(3);r=ReplayBuffer(10,spec);obs=chain()
        r.add(obs[0],0,0.,obs[1],False,behavior_prob=.5,stream=4,step=0,opponent_version=1)
        r.add(obs[1],2,0.,obs[2],False,behavior_prob=.5,stream=4,step=1,opponent_version=2)
        self.assertEqual(r.arrays['next_index'][0],-1)
        with self.assertRaises(ValueError):
            r.add(obs[0],0,0.,obs[1],False,behavior_prob=.5,stream=4,step=2,opponent_version=2)
        with self.assertRaises(ValueError):
            r.add(obs[1],0,0.,obs[2],False,behavior_prob=.5,stream=5,step=0)
        with self.assertRaises(ValueError):
            r.add(obs[0],0,0.,obs[1],True,behavior_prob=.5,stream=5,step=0)

    def test_reservoir_eligibility_and_frozen_tied_labels(self):
        spec=ObservationSpec.board(3);obs=chain()[0];rng=np.random.default_rng(5)
        r=ReservoirBuffer(10,spec,opening_record=100)
        greedy=np.zeros(10,bool);greedy[[0,2]]=True
        for kw in ({'record':99},{'record':100,'best_response_mode':False},{'record':101,'supplemental':True},{'record':102,'enabled':False}):
            params={'record':100,'best_response_mode':True,'policy_version':3,'rng':rng};params.update(kw)
            self.assertFalse(r.offer(obs,greedy,**params))
        self.assertEqual(r.seen,0)
        r.offer(obs,greedy,record=103,policy_version=3,rng=rng,best_response_mode=True)
        greedy[:]=False;greedy[9]=True
        out=r.sample(4,np.random.default_rng(1))
        np.testing.assert_array_equal(out['targets'],np.array([[.5,0,.5,0,0,0,0,0,0,0]]*4))
        self.assertEqual(r.seen,1)

    def test_reservoir_uniform_history_not_recent_window(self):
        spec=ObservationSpec((1,),0,1,'test/1')
        obs=Observation(np.ones(1,np.uint8),np.zeros(0,np.float32),np.ones(1,bool))
        counts=np.zeros(4,int)
        for seed in range(200):
            r=ReservoirBuffer(10,spec);rng=np.random.default_rng(seed)
            for record in range(100):
                r.offer(obs,np.ones(1,bool),record=record,policy_version=0,rng=rng,best_response_mode=True)
            self.assertEqual((r.seen,r.size),(100,10))
            counts+=np.bincount(r.arrays['record'][:r.size].astype(int)//25,minlength=4)
        # Fixed-seed distribution check with broad tolerance; a sliding window or
        # permanent early bias would fail by a wide margin.
        self.assertTrue(((counts>400)&(counts<600)).all(),counts)

    def test_reservoir_resume_insertion_and_sampling(self):
        spec=ObservationSpec.board(3);obs=chain()[0];rng=np.random.default_rng(73)
        r=ReservoirBuffer(8,spec,opening_record=10);greedy=np.ones(10,bool)
        for record in range(30):r.offer(obs,greedy,record=record,policy_version=record,rng=rng,best_response_mode=True)
        state=copy.deepcopy(rng.bit_generator.state)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'reservoir.npz';r.save(path);loaded=ReservoirBuffer.load(path)
            other=np.random.default_rng();other.bit_generator.state=state
            for target,generator in ((r,rng),(loaded,other)):
                for record in range(30,60):target.offer(obs,greedy,record=record,policy_version=record,rng=generator,best_response_mode=True)
            self.assertEqual(loaded.seen,r.seen)
            a=r.sample(20,rng);b=loaded.sample(20,other)
            for name in a:np.testing.assert_array_equal(a[name],b[name])


    def test_capacity_one_keeps_nonterminal_bootstrap_and_no_self_link(self):
        obs=chain();r=ReplayBuffer(1,ObservationSpec.board(3))
        for step in (0,1):r.add(obs[step],int(np.flatnonzero(obs[step].legal)[0]),0.,obs[step+1],False,behavior_prob=.5,stream=1,step=step)
        sample=r.sample_sequences(1,4,np.random.default_rng(1))
        self.assertEqual(int(sample["valid"].sum()),1)
        self.assertFalse(sample["terminated"][0,0])
        self.assertTrue(sample["next_legal"][0,0].any())
        self.assertEqual(int(r.arrays["next_index"][0]),-1)

    def test_verified_bundle_restores_both_buffers_and_sampler_state(self):
        from graygo.storage import CheckpointStore
        spec=ObservationSpec.board(3);obs=chain();replay=ReplayBuffer(8,spec);append(replay,obs)
        reservoir=ReservoirBuffer(4,spec);rng=np.random.default_rng(813)
        for record in range(10):reservoir.offer(obs[0],np.ones(10,bool),record=record,policy_version=record,rng=rng,best_response_mode=True)
        saved_rng=copy.deepcopy(rng.bit_generator.state)
        with tempfile.TemporaryDirectory() as root:
            store=CheckpointStore(root,reserve_bytes=0)
            def write(path):
                replay.save(path/'replay.npz');reservoir.save(path/'reservoir.npz')
                (path/'rng.json').write_text(json.dumps(saved_rng))
            bundle=store.save(4,write,metadata={"encoding":spec.version},estimated_peak_bytes=replay.snapshot_raw_bytes+reservoir.snapshot_raw_bytes)
            loaded_replay=ReplayBuffer.load(bundle/'replay.npz');loaded_reservoir=ReservoirBuffer.load(bundle/'reservoir.npz')
            restored_rng=np.random.default_rng();restored_rng.bit_generator.state=json.loads((bundle/'rng.json').read_text())
            for a,b in ((replay.sample_sequences(10,4,rng),loaded_replay.sample_sequences(10,4,restored_rng)),
                        (reservoir.sample(10,rng),loaded_reservoir.sample(10,restored_rng))):
                for key in a:np.testing.assert_array_equal(a[key],b[key])

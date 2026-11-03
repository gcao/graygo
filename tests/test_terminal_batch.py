from copy import deepcopy
from pathlib import Path
import ast
import hashlib
import tempfile
import unittest
import numpy as np
import torch
import graygo.nfsp_training as training
from graygo.buffers import ObservationSpec,ReplayBuffer
from graygo.encoding import Observation
from graygo.nfsp_training import NFSPMatrixConfig,NFSPMatrixTrainer


def batch(count,seed=0):
    rng=np.random.default_rng(seed);obs=[];nxt=[];actions=[]
    for i in range(count):
        legal=rng.random(5)>.3;legal[i%5]=True
        obs.append(Observation(rng.integers(0,2,11,dtype=np.uint8),rng.normal(size=2).astype(np.float32),legal))
        nxt.append(Observation(rng.integers(0,2,11,dtype=np.uint8),rng.normal(size=2).astype(np.float32),np.zeros(5,bool)))
        actions.append(int(rng.choice(np.flatnonzero(legal))))
    supplemental=rng.random(count)<.2;actor=rng.random(count)<.7;actor[supplemental]=False
    return obs,np.array(actions,dtype=np.int64),rng.uniform(-1,1,count),nxt,{
        'behavior_prob':rng.uniform(.01,1,count),'stream':rng.integers(0,10,count,dtype=np.uint64),
        'step':rng.integers(0,7,count,dtype=np.uint32),'policy_version':rng.integers(0,100,count,dtype=np.uint64),
        'opponent_version':rng.integers(0,100,count,dtype=np.uint64),'mode':rng.integers(0,2,count,dtype=np.uint8),
        'actor_mask':actor,'value_mask':rng.random(count)<.8,'supplemental':supplemental}


def scalar_add(buffer,data):
    obs,actions,rewards,nxt,kwargs=data
    return np.array([buffer.add(o,actions[i],rewards[i],nxt[i],True,**{k:v[i] for k,v in kwargs.items()})
                     for i,o in enumerate(obs)],dtype=np.int64)


def assert_buffers(test,a,b):
    test.assertEqual((a.size,a.cursor,a.insertions,a.q_size,a.q_insertions,a._tails),
                     (b.size,b.cursor,b.insertions,b.q_size,b.q_insertions,b._tails))
    for name,array in a.arrays.items():np.testing.assert_array_equal(array,b.arrays[name],err_msg=name)


def assert_tree(test,a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        test.assertEqual(a.keys(),b.keys())
        for key in a:assert_tree(test,a[key],b[key])
    elif isinstance(a,(tuple,list)):
        test.assertEqual(type(a),type(b));test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):assert_tree(test,x,y)
    else:test.assertEqual(a,b)


class TerminalBatchTests(unittest.TestCase):
    def test_batch_pack_matches_scalar_and_has_no_mutable_input_alias(self):
        spec=ObservationSpec((11,),2,5,'test');obs,*_=batch(13,3);packed=spec.pack_batch(obs)
        for i,o in enumerate(obs):
            for a,b in zip(spec.pack(o),packed):np.testing.assert_array_equal(a,b[i])
        before=tuple(a.copy() for a in packed);obs[0].spatial[:]=0;obs[0].global_features[:]=0
        for a,b in zip(before,packed):np.testing.assert_array_equal(a,b)

    def test_exact_arrays_counters_and_samples_across_all_ring_wrap_shapes(self):
        for capacity in (1,2,5,19):
            a=ReplayBuffer(capacity,ObservationSpec((11,),2,5,'test'));b=ReplayBuffer(capacity,a.spec)
            for seed,count in enumerate((0,1,2,capacity,capacity+1,2*capacity+3,1,capacity-1)):
                data=batch(count,seed);expected=scalar_add(a,data);actual=b.add_terminal_batch(*data[:4],**data[4]);np.testing.assert_array_equal(actual,expected);assert_buffers(self,a,b)
                if a.size:
                    x,y=a.sample_sequences(17,4,np.random.default_rng(7)),b.sample_sequences(17,4,np.random.default_rng(7))
                    assert_tree(self,x,y)
                    if a.q_size:assert_tree(self,a.sample_q(19,np.random.default_rng(8)),b.sample_q(19,np.random.default_rng(8)))

    def test_save_restore_and_append_continuation_are_identical(self):
        a=ReplayBuffer(17,ObservationSpec((11,),2,5,'test'));data=batch(37,4);a.add_terminal_batch(*data[:4],**data[4])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'replay.npz';a.save(path);b=ReplayBuffer.load(path)
            data=batch(29,9);a.add_terminal_batch(*data[:4],**data[4]);scalar_add(b,data);assert_buffers(self,a,b)

    def test_linked_streams_use_exact_scalar_fallback(self):
        spec=ObservationSpec((11,),2,5,'test');data=batch(2,3);o,nxt=data[0];terminal=data[3][0]
        a,b=ReplayBuffer(5,spec),ReplayBuffer(5,spec)
        for r in (a,b):r.add(o,int(np.flatnonzero(o.legal)[0]),0.,nxt,False,behavior_prob=.5,stream=7,step=0,opponent_version=3)
        action=int(np.flatnonzero(nxt.legal)[0]);a.add(nxt,action,1.,terminal,True,behavior_prob=.5,stream=7,step=1,opponent_version=3)
        b.add_terminal_batch([nxt],[action],[1.],[terminal],behavior_prob=.5,stream=7,step=1,opponent_version=3)
        assert_buffers(self,a,b);self.assertEqual(int(b.arrays['next_index'][0]),1)

    def test_specialized_spec_uses_its_scalar_packer(self):
        calls=[]
        class Custom(ObservationSpec):
            def pack(self,obs):calls.append(obs);return super().pack(obs)
        spec=Custom((11,),2,5,'test');a=ReplayBuffer(7,spec);data=batch(3)
        a.add_terminal_batch(*data[:4],**data[4]);self.assertEqual(len(calls),6)

    def test_invalid_fast_batch_never_partially_updates_storage(self):
        for mutate in (lambda d:d[1].__setitem__(-1,5),lambda d:d[2].__setitem__(-1,float('nan')),
                       lambda d:d[3][-1].legal.__setitem__(0,True),lambda d:d[4].update(mode=True),
                       lambda d:d[4]['behavior_prob'].__setitem__(-1,1e-100),
                       lambda d:d[4].update(actor_mask=True,supplemental=True)):
            a=ReplayBuffer(5,ObservationSpec((11,),2,5,'test'));before=deepcopy(a);data=list(batch(4));mutate(data)
            with self.assertRaises(ValueError):a.add_terminal_batch(*data[:4],**data[4])
            assert_buffers(self,a,before)
        a=ReplayBuffer(5,ObservationSpec((11,),2,5,'test'));d=list(batch(2));d[2]=np.array([0,np.iinfo(np.int64).min])
        with self.assertRaises(ValueError):a.add_terminal_batch(*d[:4],**d[4])

    def test_ordinal_exhaustion_and_legacy_unknown_history_fail_before_mutation(self):
        a=ReplayBuffer(3,ObservationSpec((11,),2,5,'test'));data=batch(1);a.add_terminal_batch(*data[:4],**data[4]);a.q_insertions=None
        with self.assertRaises(ValueError):a.add_terminal_batch(*data[:4],**data[4])
        a.q_insertions=1;a.insertions=2**64-2;a.cursor=a.insertions%3
        a.add_terminal_batch(*data[:4],**data[4]);self.assertEqual(a.insertions,2**64-1)
        with self.assertRaises(OverflowError):a.add_terminal_batch(*data[:4],**data[4])


class MatrixBatchTrajectoryTests(unittest.TestCase):
    def test_literal_reference_body_is_preserved(self):
        source=Path(training.__file__).read_text();tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NFSPMatrixTrainer');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='collect_reference')
        text=ast.get_source_segment(source,fn).replace('def collect_reference(','def collect(',1)
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),'050d6c32cc5890ec42cb9a46e141dd375c432fe3104f0047441de3184da70d3b')
        self.assertEqual([ast.unparse(d) for d in fn.decorator_list],['torch.no_grad()'])

    def test_both_learners_and_games_match_models_optimizers_buffers_rng_and_episode_bytes(self):
        for task in ('asymmetric','rps'):
            for learner in ('neural','tabular'):
                with self.subTest(task=task,learner=learner),tempfile.TemporaryDirectory() as tmp:
                    cfg=NFSPMatrixConfig(task=task,learner=learner,seed=2,records=320,eval_every=64,batch=16,learning_start=32,capacity=128,target_every=64,hidden=8,reservoir_open=32,handoff_end=96,epsilon_decay=160)
                    a=NFSPMatrixTrainer(cfg,Path(tmp)/'ref',reserve_bytes=0,collection='scalar-reference');a.run(verbose=False)
                    b=NFSPMatrixTrainer(cfg,Path(tmp)/'batch',reserve_bytes=0);b.run(verbose=False)
                    for model in ('q','average','target'):assert_tree(self,getattr(a.backend,model).state_dict(),getattr(b.backend,model).state_dict())
                    assert_tree(self,a.backend.q_optimizer.state_dict(),b.backend.q_optimizer.state_dict())
                    if a.backend.average_optimizer:assert_tree(self,a.backend.average_optimizer.state_dict(),b.backend.average_optimizer.state_dict())
                    assert_buffers(self,a.replay,b.replay)
                    for name in a.reservoir.arrays:np.testing.assert_array_equal(a.reservoir.arrays[name],b.reservoir.arrays[name])
                    for name in a.rng:self.assertEqual(a.rng[name].bit_generator.state,b.rng[name].bit_generator.state)
                    self.assertEqual((Path(tmp)/'ref/episodes.bin').read_bytes(),(Path(tmp)/'batch/episodes.bin').read_bytes())
                    for left,right in zip(a.evaluations,b.evaluations):
                        for key in left:
                            if key!='elapsed_seconds':self.assertEqual(left[key],right[key])

    def test_execution_mode_cannot_change_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=NFSPMatrixConfig(records=16,eval_every=16,batch=4,learning_start=4,capacity=8,hidden=8)
            a=NFSPMatrixTrainer(cfg,tmp,reserve_bytes=0);a.run(until_records=8,verbose=False)
            with self.assertRaises(ValueError):NFSPMatrixTrainer(cfg,tmp,resume=True,reserve_bytes=0,collection='scalar-reference')

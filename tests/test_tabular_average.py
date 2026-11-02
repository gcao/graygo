from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from graygo.buffers import ObservationSpec,ReservoirBuffer
from graygo.encoding import Observation
from graygo.tabular_average import TabularAverage
from graygo.nfsp_collection import FrozenPolicies,EpisodeActor
from graygo.task_models import TabularQ
from graygo.schedules import EpisodeChoice


class TabularTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def expected(self,reservoir,states,actions):
        x,_,_=reservoir.spec.unpack(reservoir.arrays['spatial'][:reservoir.size],reservoir.arrays['globals'][:reservoir.size],reservoir.arrays['legal'][:reservoir.size])
        targets=np.unpackbits(reservoir.arrays['greedy'][:reservoir.size],axis=-1,count=actions,bitorder='little').astype(float)
        targets/=targets.sum(-1,keepdims=True)
        rows=[]
        for i in range(states):
            chosen=x[:,i]>0
            rows.append(targets[chosen].mean(0) if chosen.any() else np.ones(actions)/actions)
        return np.array(rows)

    def test_replacements_and_ties_equal_independent_full_reservoir_mean(self):
        spec=ObservationSpec((3,),0,4,'tabular-test');r=ReservoirBuffer(23,spec);model=TabularAverage(3,4)
        rng=np.random.default_rng(7);changed=0
        for n in range(1000):
            obs=Observation(np.eye(3)[n%3],np.empty(0),np.ones(4,dtype=bool))
            label=np.zeros(4,dtype=bool);label[n%4]=True
            if n%2:label[(n+1)%4]=True
            if n%3==0:label[(n+2)%4]=True
            index=r.offer_index(obs,label,record=n,policy_version=n,rng=rng,best_response_mode=True)
            if index is not None:changed+=1;model.observe_reservoir_change(r,index)
            if n%37==0:
                model.publish_reservoir(r)
                np.testing.assert_allclose(model.probabilities.numpy(),self.expected(r,3,4),atol=1e-12,rtol=0)
        rebuilt=TabularAverage(3,4);rebuilt.rebuild_tracker(r);rebuilt.publish_reservoir(r)
        model.publish_reservoir(r)
        np.testing.assert_array_equal(model.probabilities.numpy(),rebuilt.probabilities.numpy())
        self.assertGreater(changed,r.size);self.assertEqual(changed,r.revisions)
        model.publish_reservoir(r)
        np.testing.assert_allclose(model.probabilities.numpy(),self.expected(r,3,4),atol=1e-12,rtol=0)

    def test_restore_rebuild_does_not_publish_pending_labels(self):
        spec=ObservationSpec((2,),0,2,'tabular-test');r=ReservoirBuffer(7,spec);model=TabularAverage(2,2);rng=np.random.default_rng(4)
        for n in range(30):
            obs=Observation(np.eye(2)[n%2],np.empty(0),np.ones(2,dtype=bool))
            index=r.offer_index(obs,np.array([n%3==0,n%3!=0]),record=n,policy_version=0,rng=rng,best_response_mode=True)
            if index is not None:model.observe_reservoir_change(r,index)
            if n==20:model.publish_reservoir(r)
        published=model.probabilities.clone()
        restored=TabularAverage(2,2);restored.load_state_dict(model.state_dict());restored.rebuild_tracker(r)
        torch.testing.assert_close(restored.probabilities,published,rtol=0,atol=0)
        restored.publish_reservoir(r)
        np.testing.assert_allclose(restored.probabilities.numpy(),self.expected(r,2,2),atol=1e-12,rtol=0)
        with tempfile.TemporaryDirectory() as d:
            r.save(Path(d)/'r.npz');loaded=ReservoirBuffer.load(Path(d)/'r.npz')
            self.assertEqual(loaded.revisions,r.revisions)

    def test_zero_support_average_is_legal_and_snapshot_excludes_tracker(self):
        model=TabularAverage(2,3);model.probabilities[:]=torch.tensor([0.,1.,0.],dtype=torch.float64)
        snapshot=FrozenPolicies(TabularQ(2,3),model,version=0,average_ready=True)
        self.assertIsNone(snapshot.average._tracker)
        actor=EpisodeActor(snapshot,EpisodeChoice(8_000_000,False,'average',.01,1.,.8,.2),np.random.default_rng(1))
        obs=Observation(np.array([1,0]),np.empty(0),np.ones(3,dtype=bool))
        result=actor.commit(obs);self.assertEqual(result.action,1);self.assertEqual(result.probability,1.)

    def test_missed_changes_fail_closed(self):
        spec=ObservationSpec((2,),0,2,'tabular-test');r=ReservoirBuffer(1,spec);model=TabularAverage(2,2);rng=np.random.default_rng(2)
        obs=Observation(np.array([1,0]),np.empty(0),np.ones(2,dtype=bool))
        missed=0
        for n in range(100):
            index=r.offer_index(obs,np.array([True,False]),record=n,policy_version=0,rng=rng,best_response_mode=True)
            if index is not None:
                if n==0:model.observe_reservoir_change(r,index)
                else:
                    missed+=1
                    if missed==2:break
        with self.assertRaises(ValueError):model.publish_reservoir(r)
        with self.assertRaises(ValueError):model.observe_reservoir_change(r,index)

if __name__=='__main__':unittest.main()

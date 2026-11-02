"""Exact per-state reservoir-label mean for tabular NFSP controls.

A change callback keeps host sufficient statistics in sync with Algorithm R,
including replaced labels. Publication follows the same ordinary-record cadence
as the neural supervised update. Frozen actors copy only the published table.
"""
import numpy as np
import torch
from torch import nn


class TabularAverage(nn.Module):
    exact_reservoir_average=True

    def __init__(self,states,actions):
        super().__init__()
        if type(states) is not int or states<1 or type(actions) is not int or actions<1:raise ValueError('positive table dimensions required')
        self.register_buffer('probabilities',torch.full((states,actions),1/actions,dtype=torch.float64))
        self._tracker=None

    def forward(self,spatial,globals):
        if spatial.ndim!=2 or spatial.shape[1]!=len(self.probabilities) or not bool(((spatial==0)|(spatial==1)).all()) or not bool((spatial.sum(-1)==1).all()):
            raise ValueError('tabular average requires a one-hot public state ID')
        return self.probabilities[spatial.argmax(-1)].log()

    def inference_copy(self):
        result=TabularAverage(*self.probabilities.shape).to(self.probabilities.device)
        result.load_state_dict(self.state_dict());return result

    def _create_tracker(self,reservoir):
        states,actions=self.probabilities.shape
        if reservoir.spec.spatial_shape!=(states,) or reservoir.spec.actions!=actions:
            raise ValueError('tabular reservoir dimensions disagree')
        self._tracker={'ids':np.full(reservoir.capacity,-1,dtype=np.int32),
                       'masks':np.zeros((reservoir.capacity,reservoir.spec.legal_bytes),dtype=np.uint8),
                       'histogram':np.zeros((actions+1,states,actions),dtype=np.int64),'counts':np.zeros(states,dtype=np.int64),
                       'revision':reservoir.revisions-1,'size':0}

    def _change(self,reservoir,index):
        t=self._tracker;a=reservoir.arrays;states,actions=self.probabilities.shape
        x=np.unpackbits(a['spatial'][index],count=states,bitorder='little')
        if int(x.sum())!=1:raise ValueError('reservoir observation is not a one-hot state')
        new_id=int(x.argmax());new_mask=a['greedy'][index]
        old_id=int(t['ids'][index])
        if old_id>=0:
            old=np.unpackbits(t['masks'][index],count=actions,bitorder='little').astype(float)
            t['histogram'][int(old.sum()),old_id]-=old.astype(np.int64);t['counts'][old_id]-=1
        else:t['size']+=1
        new=np.unpackbits(new_mask,count=actions,bitorder='little').astype(float)
        if not new.any():raise ValueError('empty tabular label')
        t['histogram'][int(new.sum()),new_id]+=new.astype(np.int64);t['counts'][new_id]+=1
        t['ids'][index]=new_id;t['masks'][index]=new_mask

    def observe_reservoir_change(self,reservoir,index):
        if self._tracker is None:self._create_tracker(reservoir)
        if not 0<=index<reservoir.size:raise ValueError('invalid changed reservoir slot')
        if self._tracker['revision']+1!=reservoir.revisions:raise ValueError('missing reservoir change callback')
        self._change(reservoir,index);self._tracker['revision']=reservoir.revisions

    def rebuild_tracker(self,reservoir):
        """Restore sufficient statistics without changing the published actor."""
        self._create_tracker(reservoir)
        for index in range(reservoir.size):self._change(reservoir,index)
        self._tracker['revision']=reservoir.revisions

    @torch.no_grad()
    def publish_reservoir(self,reservoir):
        if self._tracker is None:self.rebuild_tracker(reservoir)
        t=self._tracker
        if t['revision']!=reservoir.revisions or t['size']!=reservoir.size or t['counts'].sum()!=reservoir.size:raise ValueError('missing reservoir change callbacks')
        histogram=t['histogram'];counts=t['counts']
        if histogram.min()<0:raise ValueError('negative tabular label count')
        totals=np.zeros(self.probabilities.shape,dtype=np.float64)
        for ties in range(1,len(histogram)):totals+=histogram[ties]/ties
        if totals.min() < -1e-7 or not np.allclose(totals.sum(-1),counts,atol=1e-7,rtol=0):
            raise ValueError('tabular reservoir sufficient statistics disagree')
        p=np.full_like(totals,1/totals.shape[1]);present=counts>0
        p[present]=np.maximum(totals[present],0)/counts[present,None]
        p/=p.sum(-1,keepdims=True)
        self.probabilities.copy_(torch.as_tensor(p,device=self.probabilities.device))

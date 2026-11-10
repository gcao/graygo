"""Bounded memoization of immutable scalar CPU 3x3 network forwards.

Only raw network output is retained. Masks, episode modes, exploration,
probabilities and private random draws remain in their original callers.
The cache is derived data and is deliberately absent from checkpoints.
"""
from collections import OrderedDict
from weakref import WeakKeyDictionary
import torch
from .models import QNet,PolicyNet

CACHE_VERSION='graygo/cpu-3x3-frozen-raw-forward/1'


def supports_model(model):
    if getattr(getattr(model,"encoder",None),"spatial_channels",30)!=30:return False
    return type(model) in (QNet,PolicyNet)


class CPU3x3ForwardCache:
    def __init__(self,*,max_entries_per_model=128):
        if type(max_entries_per_model) is not int or max_entries_per_model<1:raise ValueError('positive bounded CPU forward cache required')
        self.max_entries=max_entries_per_model;self._models=WeakKeyDictionary()
        self.hits=self.misses=self.fallbacks=self.invalidations=self.evictions=0

    def invalidate(self,model=None):
        if model is None:self._models.clear()
        else:self._models.pop(model,None)
        self.invalidations+=1

    @staticmethod
    def _signature(model,inputs):
        if not supports_model(model) or torch.is_grad_enabled() or torch.is_autocast_enabled('cpu') or set(inputs)!={'spatial','globals'}:return None
        x,g=inputs['spatial'],inputs['globals']
        if not all(isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==torch.float32 and t.layout==torch.strided and t.is_contiguous() for t in (x,g)):return None
        if x.shape!=(1,30,3,3) or g.shape!=(1,14):return None
        modules=tuple(model.modules())
        if any(m.training or m._forward_hooks or m._forward_pre_hooks or m._backward_hooks or m._backward_pre_hooks or 'forward' in m.__dict__ for m in modules):return None
        tensors=tuple(model.parameters())+tuple(model.buffers())
        if not tensors or any(t.device.type!='cpu' or t.dtype!=torch.float32 or t.layout!=torch.strided or not t.is_contiguous() or t.requires_grad for t in tensors):return None
        try:signature=tuple((id(t),t.data_ptr(),t._version,tuple(t.shape),tuple(t.stride())) for t in tensors)
        except RuntimeError:return None
        runtime=(torch.get_num_threads(),torch.are_deterministic_algorithms_enabled(),torch.backends.mkldnn.enabled,torch.backends.mkldnn.deterministic)
        return signature,runtime

    def forward(self,model,**inputs):
        signature=self._signature(model,inputs)
        if signature is None:
            self.fallbacks+=1;return model(**inputs)
        previous=self._models.get(model)
        if previous is None or previous[0]!=signature:
            if previous is not None:self.invalidations+=1
            entries=OrderedDict();self._models[model]=(signature,entries)
        else:entries=previous[1]
        key=(inputs['spatial'].detach().numpy().tobytes(),inputs['globals'].detach().numpy().tobytes())
        if key in entries:
            self.hits+=1;value=entries.pop(key);entries[key]=value;return value.clone()
        self.misses+=1;result=model(**inputs)
        if isinstance(result,torch.Tensor) and result.shape==(1,10) and result.device.type=='cpu' and result.dtype==torch.float32 and not result.requires_grad and bool(torch.isfinite(result).all()):
            entries[key]=result.detach().clone()
            if len(entries)>self.max_entries:entries.popitem(last=False);self.evictions+=1
        return result

    def stats(self):
        return {'version':CACHE_VERSION,'hits':self.hits,'misses':self.misses,'fallbacks':self.fallbacks,'invalidations':self.invalidations,'evictions':self.evictions,
            'resident_models':len(self._models),'resident_entries':sum(len(value[1]) for value in self._models.values()),'max_entries_per_model':self.max_entries}

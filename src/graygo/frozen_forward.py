"""Opt-in replay of immutable full-board CUDA forwards with shared storage.

Sampling, legality checks, training and optimizer work stay outside capture.
Each replay copies the selected immutable weights and public inputs into fixed
addresses. Output cloning prevents a later model's replay from aliasing a result.
"""
from collections import OrderedDict
from copy import deepcopy
import os
import threading
import time
from weakref import WeakKeyDictionary
import torch

VERSION='graygo/shared-frozen-cuda-forward/1'
_CAPTURE_LOCK=threading.RLock()


def execution_identity(enabled):
    if type(enabled) is not bool:raise ValueError('explicit frozen CUDA graph selection required')
    if not enabled:return None
    return {'version':VERSION,'backend':'shared-static-weight-cuda-graph','eligible_models':['QNet','PolicyNet'],
        'model_shape':[64,8,8],'board_size':9,'batch_range':[1,32],'max_templates':64,
        'parameter_alignment_bytes':256,'warmup_forwards':3,'outside_capture':['legality','private RNG','optimizer'],
        'other_shapes':'eager forward; reported separately','checkpoint_cache':'reconstructed; never serialized'}


class FrozenForwardPool:
    """One owner thread/stream and deterministic context per runtime worker."""
    def __init__(self,device):
        device=torch.device(device)
        if device.type!='cuda':raise ValueError('frozen graph execution requires CUDA')
        self.device=torch.device('cuda',torch.cuda.current_device() if device.index is None else device.index)
        self.owner=threading.get_ident();self.context=None;self.templates=OrderedDict();self.weights=WeakKeyDictionary()
        self.metrics={'captures':0,'weight_packs':0,'replays':0,'eager_fallbacks':0,'capture_seconds':0.,'weight_pack_seconds':0.}

    def _context(self):
        if threading.get_ident()!=self.owner:raise ValueError('frozen CUDA pool cannot change owner thread')
        if torch.is_autocast_enabled():raise ValueError('frozen CUDA graph path requires explicit FP32 inference')
        if not torch.are_deterministic_algorithms_enabled():raise ValueError('frozen CUDA graph path requires deterministic runtime')
        value=(torch.cuda.current_stream(self.device).cuda_stream,torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic,
               torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32,os.environ.get('CUBLAS_WORKSPACE_CONFIG'),torch.get_num_threads())
        if self.context is None:self.context=value
        elif value!=self.context:raise ValueError('frozen CUDA graph runtime/stream changed after initialization')

    def _eligible(self,model,spatial,globals,method):
        from .models import QNet,PolicyNet,ModelConfig
        return (method=='forward' and type(model) in (QNet,PolicyNet) and model.config==ModelConfig()
            and spatial.device==globals.device==self.device and spatial.dtype==globals.dtype==torch.float32
            and spatial.ndim==4 and tuple(spatial.shape[1:])==(30,9,9) and 1<=spatial.shape[0]<=32
            and tuple(globals.shape)==(spatial.shape[0],14) and spatial.is_contiguous() and globals.is_contiguous()
            and not list(model.buffers()))

    def _pack(self,model):
        params=tuple(model.named_parameters())
        if model.training or any(p.requires_grad for _,p in params):raise ValueError('graph executor only accepts immutable evaluation models')
        if any(p.device!=self.device or p.dtype!=torch.float32 for _,p in params):raise ValueError('frozen model device/dtype differs')
        versions=tuple((p._version,p.data_ptr()) for _,p in params)
        saved=self.weights.get(model)
        if saved is not None:
            if saved[2]!=versions:raise ValueError('frozen parameters changed; invalidate explicitly before restoring weights')
            return saved
        torch.cuda.synchronize(self.device);started=time.perf_counter();layout=[];cursor=0
        for name,param in params:
            cursor=(cursor+63)//64*64;layout.append((name,cursor,tuple(param.shape),param.numel()));cursor+=param.numel()
        flat=torch.zeros((cursor+63)//64*64,device=self.device,dtype=torch.float32)
        for (_,offset,_,count),(_,param) in zip(layout,params):flat[offset:offset+count].copy_(param.reshape(-1))
        torch.cuda.synchronize(self.device);self.metrics['weight_pack_seconds']+=time.perf_counter()-started;self.metrics['weight_packs']+=1
        saved=(flat,tuple(layout),versions);self.weights[model]=saved;return saved

    def invalidate(self,model):
        """Checkpoint restoration is explicit; live inference never refreshes."""
        self.weights.pop(model,None)

    @torch.no_grad()
    def forward(self,model,spatial,globals,*,method='forward'):
        self._context()
        if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('graph executor only accepts immutable evaluation models')
        if not self._eligible(model,spatial,globals,method):
            self.metrics['eager_fallbacks']+=1;return getattr(model,method)(spatial,globals)
        with _CAPTURE_LOCK:
            flat,layout,_=self._pack(model);key=(type(model).__name__,int(spatial.shape[0]))
            template=self.templates.get(key)
            if template is None:
                # Official full model, two classes and batches 1..32 bound all
                # cache keys. Different contexts are rejected rather than leaked.
                if len(self.templates)>=64:raise RuntimeError('bounded frozen graph template inventory exhausted')
                torch.cuda.synchronize(self.device);started=time.perf_counter()
                cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state(self.device).clone()
                copy=deepcopy(model);storage=torch.empty_like(flat);storage.copy_(flat)
                lookup=dict(copy.named_parameters())
                for name,offset,shape,count in layout:lookup[name].data=storage[offset:offset+count].view(shape)
                x=spatial.clone();g=globals.clone();stream=torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):copy(x,g)
                torch.cuda.current_stream(self.device).wait_stream(stream);torch.cuda.synchronize(self.device)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):output=copy(x,g)
                torch.cuda.synchronize(self.device)
                if not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state(self.device)):
                    raise RuntimeError('frozen capture unexpectedly consumed a global RNG')
                template=(copy,storage,x,g,graph,output,layout);self.templates[key]=template
                self.metrics['captures']+=1;self.metrics['capture_seconds']+=time.perf_counter()-started
            _,storage,x,g,graph,output,expected_layout=template
            if layout!=expected_layout:raise ValueError('frozen graph parameter layout changed')
            storage.copy_(flat);x.copy_(spatial);g.copy_(globals);graph.replay();self.metrics['replays']+=1
            return output.clone()

    def stats(self):
        return dict(self.metrics,resident_templates=len(self.templates),resident_weight_packs=len(self.weights),execution=execution_identity(True))

    def clear(self):
        self._context();torch.cuda.synchronize(self.device);self.templates.clear();self.weights.clear()

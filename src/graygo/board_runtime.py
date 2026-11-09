"""Complete board collector/update/resume component and bounded hardware profiles.

This runtime is not an A2 approval or development/confirmation orchestrator. Its
public profile runner is capped and produces no deployment/strength certificate.
"""
from dataclasses import asdict,dataclass
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from .board_augmentation import augment,AUGMENTATION_VERSION
from .board_collection import BoardCollector,COLLECTOR_VERSION
from .buffers import ReplayBuffer,ReservoirBuffer
from .engine import Rules
from .models import ModelConfig,QNet,PolicyNet
from .nfsp import NFSPBackend,NFSPConfig
from .nfsp_collection import FrozenPolicies
from .rule_state import RuleStateSpec
from .schedules import LearnerSchedule
from .storage import CheckpointStore,DEFAULT_RESERVE,digest
from .manifest import make_manifest,source_digest,write_report
from .confidence import canonical_sha

RUNTIME_VERSION='graygo/board-nfsp-runtime/1'


def configure_runtime(device,deterministic):
    device=torch.device(device)
    if deterministic:
        if device.type=='cuda':
            allowed=(':4096:8',':16:8')
            if torch.cuda.is_initialized() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in allowed:raise ValueError('start a fresh CUDA process with deterministic CUBLAS workspace')
            if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in allowed:os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    return {'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
        'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cublas_workspace':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'device':str(device),'torch_threads':torch.get_num_threads()}


@dataclass(frozen=True)
class BoardRuntimeConfig:
    rules: Rules=Rules()
    envs: int=16
    seed: int=17
    batch: int=512
    learning_start: int=4096
    capacity: int=500000
    q_reuse: int=1
    target_every: int=12800
    learning_rate: float=3e-4
    max_grad_norm: float=.5
    width: int=64
    blocks: int=8
    norm_groups: int=8
    native_library: str|None=None
    augmentation: bool=True
    deterministic: bool=True
    device: str='cpu'
    reservoir_open: int=5_000_000
    handoff_end: int=7_000_000
    uniform_comparison: bool=False
    cuda_graphs: bool=False
    q_target: str='one-step'
    trace_length: int=32
    trace_lambda: float=.95
    target_batch: int=512
    initial_weights_archive: str|None=None
    initial_weights_sha256: str|None=None

    def __post_init__(self):
        if not isinstance(self.rules,Rules) or type(self.envs) is not int or self.envs<1 or type(self.seed) is not int or self.seed<0:raise ValueError('board rules, env count and seed required')
        if type(self.batch) is not int or self.batch%2 or type(self.capacity) is not int or self.capacity<self.learning_start:raise ValueError('paired update batch and adequate replay required')
        if self.initial_weights_archive is not None:
            if self.rules.ko!='full-history':raise ValueError('weights-only v2 initialization requires explicit v2 rules')
            path=Path(self.initial_weights_archive).expanduser().resolve();CheckpointStore.verify(path);sha=digest(path/'manifest.json')
            if self.initial_weights_sha256 is not None and self.initial_weights_sha256!=sha:raise ValueError('initial weights archive changed')
            object.__setattr__(self,'initial_weights_archive',str(path));object.__setattr__(self,'initial_weights_sha256',sha)
        elif self.initial_weights_sha256 is not None:raise ValueError('initial weights fingerprint requires an archive')
        self.backend_config();self.model_config();self.schedule()
        if self.reservoir_open%2 or self.handoff_end%2:raise ValueError('paired opening boundaries required')
        if any(type(v) is not bool for v in (self.augmentation,self.deterministic,self.uniform_comparison,self.cuda_graphs)):raise ValueError('explicit runtime flags required')
        if self.cuda_graphs and (torch.device(self.device).type!='cuda' or not self.deterministic):raise ValueError('frozen CUDA graph runtime requires deterministic CUDA')
        if self.native_library is not None:object.__setattr__(self,'native_library',str(Path(self.native_library).expanduser().resolve()))
    def backend_config(self):return NFSPConfig(self.batch,self.learning_start,self.q_reuse,self.target_every,self.learning_rate,self.max_grad_norm,
        self.q_target,self.trace_length,self.trace_lambda,self.target_batch)
    def model_config(self):return ModelConfig(self.width,self.blocks,self.norm_groups)
    def schedule(self):return LearnerSchedule(40_000_000,self.reservoir_open,self.handoff_end)
    def plan(self):
        result=json.loads(json.dumps(asdict(self)));result["rules"]=self.rules.to_dict()
        if self.initial_weights_archive is None:
            result.pop("initial_weights_archive");result.pop("initial_weights_sha256")
        return result


class BoardNFSPRuntime:
    def __init__(self,config,*,starts=None):
        self.config=config;torch.set_num_threads(1);self.settings=configure_runtime(config.device,config.deterministic)
        torch.manual_seed(config.seed);self.spec=RuleStateSpec(config.rules,config.native_library)
        self.backend=NFSPBackend(QNet(config.model_config()),PolicyNet(config.model_config()),config.backend_config(),device=config.device)
        from .frozen_forward import FrozenForwardPool
        self.forward_pool=FrozenForwardPool(config.device) if config.cuda_graphs else None
        self.replay=ReplayBuffer(config.capacity,self.spec);self.reservoir=ReservoirBuffer(config.capacity,self.spec,opening_record=config.reservoir_open)
        self.collector=BoardCollector(config.rules,config.envs,config.seed,starts=starts);self.schedule=config.schedule();self.snapshot=None
        from .evaluation import seed_for
        self.rng={name:np.random.default_rng(seed_for(config.seed,'board-runtime',name)) for name in ('labels','q_replay','supervised','augment_q','augment_supervised')}
        self.initial_weights=None
        if config.initial_weights_archive is not None:self._seed_weights(config.initial_weights_archive)
        self.timings={'collection':0.,'learning':0.,'checkpoint':0.};self.loss={'q_updates':0,'q_loss_sum':0.,'supervised_updates':0,'supervised_loss_sum':0.,'projected_targets':0}

    def _seed_weights(self,archive):
        path=Path(archive);data=torch.load(path/'models.pt',map_location='cpu',weights_only=True)
        if {k:data['config'][k] for k in ('width','blocks','norm_groups')}!=asdict(self.config.model_config()):raise ValueError('warm-start network architecture differs')
        old_rules=data['config']['rules']
        if (old_rules['size'],old_rules['max_rounds'],old_rules['collision'])!=(self.config.rules.size,self.config.rules.max_rounds,self.config.rules.collision):raise ValueError('warm-start board/capture configuration differs')
        self.backend.q.load_state_dict(data['q']);self.backend.target.load_state_dict(data['q']);self.backend.average.load_state_dict(data['average'])
        self.backend.average_seeded=True
        self.initial_weights={'archive':str(path),'manifest_sha256':self.config.initial_weights_sha256,'training_rules':old_rules,
            'new_rules':self.config.rules.to_dict(),'transfer':'weights-only','replay_imported':False,'optimizer_imported':False}

    @property
    def records(self):return self.backend.counters.fresh_records
    def _snapshot(self):
        version=self.backend.counters.q_updates+self.backend.counters.supervised_updates
        if self.snapshot is None:
            # Restored active episodes already own the canonical frozen object
            # for this version. New episodes must share it for batched commits.
            self.snapshot=next((actor.snapshot for episode in self.collector.active if episode is not None
                                for actor in episode.actors.values() if actor.snapshot.version==version),None)
        if self.snapshot is None or self.snapshot.version!=version:self.snapshot=FrozenPolicies(self.backend.q,self.backend.average,version=version,average_ready=self.backend.average_ready,forward_pool=self.forward_pool)
        return self.snapshot

    def advance(self,count,*,opening_enabled=True):
        if type(count) is not int or count<2 or count%2 or count>2*self.config.envs or count>self.backend.counters.until_q_boundary(self.backend.config):raise ValueError('collector chunk must stop at a paired optimizer boundary')
        if self.records<self.schedule.reservoir_open<self.records+count or self.records<self.schedule.handoff_end<self.records+count:raise ValueError('chunk crosses a fixed hand-off boundary')
        start=time.perf_counter();self.collector.collect(count,self._snapshot(),self.replay,self.reservoir,self.rng['labels'],self.schedule,opening_enabled=opening_enabled,uniform_comparison=self.config.uniform_comparison)
        self.timings['collection']+=time.perf_counter()-start;start=time.perf_counter()
        q_transform=(lambda data:augment(data,self.rng['augment_q'])) if self.config.augmentation else None
        sl_transform=(lambda data:augment(data,self.rng['augment_supervised'])) if self.config.augmentation else None
        stats=self.backend.learn_after_fresh(count,count,self.replay,self.reservoir,self.rng['q_replay'],epsilon=self.schedule.epsilon(self.records+count),
            supervised_rng=self.rng['supervised'],q_transform=q_transform,supervised_transform=sl_transform)
        self.timings['learning']+=time.perf_counter()-start
        for q in stats['q']:self.loss['q_updates']+=1;self.loss['q_loss_sum']+=q['loss'];self.loss['projected_targets']+=q['projected_targets']
        if stats['supervised'] is not None:self.loss['supervised_updates']+=1;self.loss['supervised_loss_sum']+=stats['supervised']['loss']
        if self.collector.records!=self.records or self.replay.insertions!=self.records or self.replay.q_insertions!=self.backend.counters.fresh_q:raise ValueError('board collector/fresh clocks disagree')
        return stats

    def run_profile(self,until_records,*,time_cap_seconds,stop_requested=lambda:False):
        if type(until_records) is not int or until_records%2 or not self.records<=until_records<=65536 or not math.isfinite(time_cap_seconds) or time_cap_seconds<=0:raise ValueError('bounded hardware-profile endpoint/time required; no development allocation')
        started=time.monotonic();status='complete'
        while self.records<until_records:
            if stop_requested():status='interrupted';break
            if time.monotonic()-started>=time_cap_seconds:status='budget_limited';break
            count=min(2*self.config.envs,self.backend.counters.until_q_boundary(self.backend.config),until_records-self.records)
            for boundary in (self.schedule.reservoir_open,self.schedule.handoff_end):
                if self.records<boundary:count=min(count,boundary-self.records)
            self.advance(count)
        return {'status':status,'records':self.records,'per_role_records':self.collector.metrics['per_role_records'],
            'frozen_forward_stats':None if self.forward_pool is None else self.forward_pool.stats(),
            'counters':asdict(self.backend.counters),'metrics':deepcopy(self.collector.metrics),'timings':dict(self.timings),'loss':dict(self.loss),
            **({'trace_diagnostics':self.backend.trace.report()} if self.backend.trace is not None else {}),
            'elapsed_seconds':time.monotonic()-started,'formal_learning_run':False,'a2_complete':False,'promotion_permitted':False,
            'scope':'bounded real pipeline hardware exercise; untrained game lengths/modes are not full-campaign estimates'}

    def state_dict(self):
        return {'version':RUNTIME_VERSION,'config':self.config.plan(),'settings':self.settings,'backend':self.backend.state_dict(),'collector':self.collector.state_dict(),
            'rng':{k:v.bit_generator.state for k,v in self.rng.items()},'timings':self.timings,'loss':self.loss,
            **({'initial_weights':self.initial_weights} if self.initial_weights is not None else {}),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if self.backend.device.type=='cuda' else []}

    def save(self,root,*,reserve_bytes=DEFAULT_RESERVE,controller_state=None):
        start=time.perf_counter();store=CheckpointStore(root,reserve_bytes=reserve_bytes)
        control=None if controller_state is None else (json.dumps(controller_state,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        raw=io.BytesIO();torch.save(self.state_dict(),raw);payload=raw.getvalue()
        def writer(path):
            self.replay.save(path/'replay.npz');self.reservoir.save(path/'reservoir.npz');(path/'runtime.pt').write_bytes(payload)
            if control is not None:(path/'controller.json').write_bytes(control)
        peak=math.ceil(1.05*(len(payload)+(0 if control is None else len(control))+self.replay.snapshot_raw_bytes+self.reservoir.snapshot_raw_bytes))+65536
        metadata={'version':RUNTIME_VERSION,'source_sha256':source_digest(),'native_schema':self.spec.to_dict()}
        if control is not None:metadata['controller_sha256']=canonical_sha(controller_state)
        path=store.save(self.records,writer,metadata=metadata,estimated_peak_bytes=peak)
        self.timings['checkpoint']+=time.perf_counter()-start;return path

    def restore(self,path,*,compatible_source_sha256=None):
        meta=CheckpointStore.verify(path)
        # Explicit handoffs can admit a separately verified compatible source.
        # Ordinary checkpoint resume retains its strict exact-source check.
        sources={source_digest()}
        if compatible_source_sha256 is not None:
            if not isinstance(compatible_source_sha256,str) or len(compatible_source_sha256)!=64 or any(c not in '0123456789abcdef' for c in compatible_source_sha256):raise ValueError('explicit compatible source fingerprint required')
            sources.add(compatible_source_sha256)
        if meta['metadata']['source_sha256'] not in sources or meta['metadata']['native_schema']!=self.spec.to_dict():raise ValueError('runtime source/decoder identity changed')
        # Load storage on CPU, then let each model/optimizer restore its own
        # device layout. Mapping the entire bundle to CUDA moves Adam's CPU
        # step counters and changes the otherwise identical runtime state.
        state=torch.load(Path(path)/'runtime.pt',weights_only=True,map_location='cpu')
        if state['version']!=RUNTIME_VERSION or state['config']!=self.config.plan() or state['settings']!=self.settings:raise ValueError('runtime resume configuration/settings differ')
        self.backend.load_state_dict(state['backend']);self.replay=ReplayBuffer.load(Path(path)/'replay.npz');self.reservoir=ReservoirBuffer.load(Path(path)/'reservoir.npz')
        if self.replay.spec!=self.spec or self.reservoir.spec!=self.spec or self.replay.capacity!=self.config.capacity or self.reservoir.capacity!=self.config.capacity or self.reservoir.opening_record!=self.config.reservoir_open:raise ValueError('runtime buffer identity differs')
        if self.forward_pool is not None:self.forward_pool.clear()
        self.collector.load_state_dict(state['collector'],self.backend.q,self.backend.average,forward_pool=self.forward_pool)
        if self.collector.records!=self.records or self.replay.insertions!=self.records or self.replay.q_insertions!=self.backend.counters.fresh_q or meta['step']!=self.records:raise ValueError('runtime record accounting differs')
        for k,v in state['rng'].items():self.rng[k].bit_generator.state=v
        self.timings=state['timings'];self.loss=state['loss'];self.snapshot=None;self.initial_weights=state.get('initial_weights')
        torch.set_rng_state(state['torch_rng'].cpu())
        if self.backend.device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])

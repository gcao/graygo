"""Bounded, source-bound observation of actual CPU neural task update boundaries.

Wrappers delegate to the exact original production functions and retain their
outputs. Recording is single-process, non-resumable and never selects training.
"""
from copy import deepcopy
from dataclasses import asdict
from importlib import metadata
import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import torch
from graygo.manifest import source_digest, write_report
from graygo.storage import DEFAULT_RESERVE
from tools.nfsp_resources import require

VERSION = 'graygo/retrace-boundary-recording/1'


def cpu_copy(value):
    if isinstance(value, torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):return torch.from_numpy(value.copy())
    if isinstance(value, dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value, tuple):return tuple(cpu_copy(v) for v in value)
    if isinstance(value, list):return [cpu_copy(v) for v in value]
    return deepcopy(value)


def arrays(value):
    return {k:v.cpu().numpy().copy() for k,v in value.items()}


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()


def raw_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(raw_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(map(raw_bytes,value))
    return 64


class BoundaryRecorder:
    def __init__(self,backend,root,task_config,*,q_boundaries,first_supervised=True,
                 reserve_bytes=DEFAULT_RESERVE,max_bytes=8*1024**3):
        require(backend.device.type=='cpu' and backend.config.q_target=='retrace'
                and not backend.exact_average,'CPU neural Retrace backend required')
        require(task_config.device=='cpu' and task_config.learner=='neural'
                and task_config.q_target=='retrace' and task_config.storage=='features' and task_config.native_library is None,
                'matching CPU/features neural task specification required')
        require(tuple(q_boundaries)==tuple(sorted(set(q_boundaries))) and q_boundaries
                and all(type(v) is int and v>=backend.config.learning_start and v%backend.config.batch==0 for v in q_boundaries),
                'fixed aligned Q-boundary inventory required')
        require(type(first_supervised) is bool and type(max_bytes) is int and max_bytes>0,'explicit bounded recording settings required')
        require(torch.get_default_dtype()==torch.float32 and not torch.is_autocast_enabled('cpu'),'explicit float32 non-autocast recording required')
        require(all(value==0 for value in asdict(backend.counters).values()),'recording starts at zero counters; no implicit resume')
        self.backend=backend;self.root=Path(root);self.config=task_config
        require(not self.root.exists(),'preserve original recording; use a fresh output')
        self.root.mkdir(parents=True);self.q_boundaries=tuple(q_boundaries);self.first_supervised=first_supervised
        self.reserve=reserve_bytes;self.max_bytes=max_bytes;self.used=0;self.rows=[];self.seen=set();self.sl_seen=False;self.active=None
        self.header={'version':VERSION,'source_sha256':source_digest(),'observer_sha256':file_sha(__file__),
            'dependencies':{name:metadata.version(name) for name in ('numpy','torch','scipy')},
            'task_config':task_config.plan(),'backend_config':asdict(backend.config),'q_boundaries':list(q_boundaries),
            'first_supervised':first_supervised,'device':'cpu','torch_threads':torch.get_num_threads(),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'mkldnn_enabled':torch.backends.mkldnn.enabled,'matmul_precision':torch.get_float32_matmul_precision(),
            'max_bytes':max_bytes,'new_full_training_selected':False,'source_admission_qualified':False}
        self._publish('recording')

    def _publish(self,status):
        complete=set(self.q_boundaries)<=self.seen and (not self.first_supervised or self.sl_seen)
        write_report(self.root/'recording.json',dict(self.header,status=status,boundaries=self.rows,
            complete_predeclared_inventory=complete,recorded_bytes=self.used),reserve_bytes=self.reserve)

    def __enter__(self):
        import graygo.nfsp as core
        import graygo.sequence_targets as target
        require('learn_after_fresh' not in self.backend.__dict__,'unwrapped backend method required')
        self.original_learn=self.backend.learn_after_fresh
        self.original_q=target.update_retrace_q;self.original_sl=core.update_average
        self.patches=[patch.object(target,'update_retrace_q',self._q),patch.object(core,'update_average',self._sl)]
        for item in self.patches:item.start()
        self.backend.learn_after_fresh=self._learn
        return self

    def __exit__(self,kind,value,traceback):
        del self.backend.learn_after_fresh
        for item in reversed(self.patches):item.stop()
        self._publish('closed' if kind is None else 'failed')
        return False

    def _learn(self,count,ordinary,replay,reservoir,rng,**kwargs):
        c=self.backend.counters;cfg=self.backend.config
        eligible=kwargs.get('q_eligible')
        if eligible is None:eligible=count
        next_q=c.fresh_q+eligible
        q_due=eligible>0 and next_q%cfg.batch==0 and next_q>=cfg.learning_start
        sl_due=(c.fresh_ordinary+ordinary)//cfg.batch>c.fresh_ordinary//cfg.batch and reservoir.size>=cfg.batch
        selected=(q_due and next_q in self.q_boundaries) or (sl_due and self.first_supervised and not self.sl_seen)
        require(kwargs.get('q_transform') is None and kwargs.get('supervised_transform') is None,
                'this observer authenticates unaugmented A2 task samples; board augmentation needs a separate proof')
        srng=kwargs.get('supervised_rng')
        if srng is None:srng=rng
        if selected:
            self.active={'version':VERSION,'pre':cpu_copy(self.backend.state_dict()),'q_steps':[],
                'supervised':None,'count':count,'ordinary':ordinary,'q_eligible':eligible,
                'epsilon':float(kwargs['epsilon']),'same_sampler':rng is srng,
                'q_rng_before':deepcopy(rng.bit_generator.state),'sl_rng_before':deepcopy(srng.bit_generator.state),
                'replay':{'size':replay.size,'capacity':replay.capacity,'insertions':replay.insertions,
                    'q_size':replay.q_size,'q_insertions':replay.q_insertions,
                    'arrays':cpu_copy({k:v[:replay.size] for k,v in replay.arrays.items()})},
                'reservoir':{'size':reservoir.size,'seen':reservoir.seen,
                    'arrays':cpu_copy({k:v[:reservoir.size] for k,v in reservoir.arrays.items()})}}
        result=self.original_learn(count,ordinary,replay,reservoir,rng,**kwargs)
        if selected:
            item=self.active;item['post']=cpu_copy(self.backend.state_dict());item['report']=deepcopy(result)
            item['q_rng_after']=deepcopy(rng.bit_generator.state);item['sl_rng_after']=deepcopy(srng.bit_generator.state)
            if result['q']:self.seen.add(next_q)
            if result['supervised'] is not None:self.sl_seen=True
            estimated=int(1.05*raw_bytes(item))+1024*1024
            require(self.used+estimated<=self.max_bytes and shutil.disk_usage(self.root).free-estimated>=self.reserve,
                    'recording would exceed its fixed storage ceiling/reserve')
            name=f'boundary-{len(self.rows):03d}-{self.backend.counters.fresh_records:012d}.pt'
            temporary=self.root/('.'+name);torch.save(item,temporary)
            size=temporary.stat().st_size
            require(self.used+size<=self.max_bytes,'serialized boundary exceeds declared recording ceiling')
            path=self.root/name;temporary.rename(path);self.used+=size
            self.rows.append({'file':name,'sha256':file_sha(path),'bytes':size,
                'fresh_records':self.backend.counters.fresh_records,'fresh_q':next_q,
                'q_updates':len(item['q_steps']),'supervised_update':item['supervised'] is not None,
                'target_refreshed':result['target_refreshed']})
            self.active=None;self._publish('recording')
        return result

    def _q(self,online,target,optimizer,data,**kwargs):
        if self.active is None:return self.original_q(online,target,optimizer,data,**kwargs)
        require(online is self.backend.q and target is self.backend.target and optimizer is self.backend.q_optimizer,
                'recorded update belongs to another backend')
        require(kwargs.get('target_observer') is None,'one exact target observation path required')
        frame={'pre':cpu_copy(self.backend.state_dict()),'data':cpu_copy(data)}
        def observe(labels,stats,details):
            require('targets' not in frame,'one target computation per optimizer step required')
            frame.update(targets=cpu_copy(labels),stats=deepcopy(stats),details=cpu_copy(details))
        result=self.original_q(online,target,optimizer,data,target_observer=observe,**kwargs)
        frame['post']=cpu_copy(self.backend.state_dict());frame['result']=deepcopy(result)
        self.active['q_steps'].append(frame)
        return result

    def _sl(self,policy,optimizer,data,**kwargs):
        if self.active is None:return self.original_sl(policy,optimizer,data,**kwargs)
        require(policy is self.backend.average and optimizer is self.backend.average_optimizer,
                'recorded supervised update belongs to another backend')
        frame={'pre':cpu_copy(self.backend.state_dict()),'data':cpu_copy(data)}
        result=self.original_sl(policy,optimizer,data,**kwargs)
        frame['post']=cpu_copy(self.backend.state_dict());frame['result']=deepcopy(result)
        self.active['supervised']=frame
        return result

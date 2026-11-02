"""Paired public-input/action augmentation for off-policy NFSP board records."""
from functools import lru_cache
import numpy as np
from .engine import Rules,transform_action

AUGMENTATION_VERSION='graygo/board-paired-augmentation/1'


@lru_cache(maxsize=8)
def permutations(size):
    rules=Rules(size,1);points=rules.points;result=[]
    for rotation in range(4):
        for reflect in (False,True):
            for row in range(size):
                for col in range(size):
                    result.append([transform_action(p,rules,shift=(row,col),rotation=rotation,reflect=reflect) for p in range(points)])
    p=np.array(result,dtype=np.int64);p.flags.writeable=False
    inverse=np.argsort(p,axis=1);inverse.flags.writeable=False
    return p,inverse


def augment(data,rng,*,choices=None):
    """One bijection per sampled start, shared across its full sequence context.

Behavior probability/reward/globals stay scalar-invariant; illegal masks, pass,
executed actions and frozen target distributions transform together.
"""
    x=data['spatial']
    if x.ndim not in (4,5) or x.shape[-1]!=x.shape[-2] or x.shape[-3]!=30:raise ValueError('board feature batch/sequence required')
    n=x.shape[-1];points=n*n;table,inverse=permutations(n);batch=len(x)
    if choices is None:choices=rng.integers(0,len(table),batch)
    choices=np.asarray(choices)
    if choices.shape!=(batch,) or not np.issubdtype(choices.dtype,np.integer) or (choices<0).any() or (choices>=len(table)).any():raise ValueError('one valid transform per start required')
    result={key:value.copy() for key,value in data.items()}
    for row,choice in enumerate(choices):
        permutation,back=table[choice],inverse[choice]
        for key in ('spatial','next_spatial'):
            if key in data:
                source=data[key][row];result[key][row]=source.reshape(*source.shape[:-2],points)[...,back].reshape(source.shape)
        for key in ('legal','next_legal','targets'):
            if key in data:
                if data[key].shape[-1]!=points+1:raise ValueError('action-aligned masks/targets required')
                result[key][row,...,:points]=np.take(data[key][row,...,:points],back,axis=-1)
                result[key][row,...,points]=data[key][row,...,points]
        if 'action' in data:
            a=np.asarray(data['action'][row]);safe=np.minimum(a,points-1)
            result['action'][row]=np.where(a==points,points,permutation[safe])
            if 'valid' in data:result['action'][row]=np.where(data['valid'][row],result['action'][row],0)
    return result

"""Shared independent action sampling for scalar and batched NFSP actors."""
import numpy as np


def draw_actions(probabilities,rng):
    """Independent inverse-CDF draws, with exact marginal probabilities logged."""
    p=np.asarray(probabilities,dtype=np.float64).copy()
    if p.ndim!=2 or not len(p) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(-1),1,atol=1e-6,rtol=0):
        raise ValueError('normalized nonnegative action rows required')
    p/=p.sum(-1,keepdims=True)
    cdf=p.cumsum(-1);cdf[:,-1]=1.
    actions=(rng.random(len(p))[:,None]>=cdf).sum(-1)
    return actions,p[np.arange(len(p)),actions]



"""Fixed-mixture betting tests for bounded independent color-pair means.

For every fixed nonnegative stake b <= .95/(hi-lo), the product of
1+b*(X-threshold) is a nonnegative supermartingale under E[X|past]<=threshold.
A uniform mixture preserves that property. Ville's inequality controls all
looks at once. Conjunctive release conditions form an intersection-union test:
any false promotion must cross the threshold for at least one true null.
https://arxiv.org/abs/2010.09686
"""
from dataclasses import dataclass,asdict
import math,json
import numpy as np
from .confidence import canonical_sha

VERSION='graygo/paired-betting/1'


def stakes(minimum,maximum):
    if not math.isfinite(minimum+maximum) or minimum>=maximum:raise ValueError('ordered bounded support required')
    return np.concatenate((np.geomspace(.0001,.025,8),np.linspace(.05,.95,40)))/(maximum-minimum)


def log_e(values,threshold,*,minimum=-1.,maximum=1.):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or (x<minimum).any() or (x>maximum).any():raise ValueError('finite bounded observations required')
    if not minimum<=threshold<=maximum:raise ValueError('null threshold outside support')
    capital=np.log1p((x[:,None]-threshold)*stakes(minimum,maximum)).sum(0)
    top=float(capital.max())
    return top+math.log(float(np.exp(capital-top).mean()))


def estimate(values,*,alpha,minimum=-1.,maximum=1.):
    if not 0<alpha<1:raise ValueError('positive test alpha required')
    x=np.asarray(values,dtype=float);left,right=minimum,float(x.mean());barrier=math.log(1/alpha)
    # Fail-closed support checking occurs even when no nontrivial bound exists.
    initial=log_e(x,left,minimum=minimum,maximum=maximum)
    if initial<barrier:lower=minimum
    else:
        for _ in range(45):
            midpoint=(left+right)/2
            if log_e(x,midpoint,minimum=minimum,maximum=maximum)>=barrier:left=midpoint
            else:right=midpoint
        lower=left
    return {'mean':float(x.mean()),'lower':lower,'independent_units':len(x),'games':2*len(x),
            'sample_variance':float(x.var(ddof=1)) if len(x)>1 else 0.,'alpha':alpha,'method':VERSION,'support':[minimum,maximum]}


@dataclass(frozen=True)
class BettingFamily:
    attempt:int
    looks:tuple
    endpoints:tuple
    total_alpha:float=.05
    @property
    def alpha(self):return self.total_alpha/(self.attempt*(self.attempt+1))
    def to_dict(self):return json.loads(json.dumps({'method':VERSION,**asdict(self),'attempt_alpha':self.alpha,'endpoint_allocation':'intersection-union; all required tests must pass','look_allocation':'time-uniform mixture; no look split'}))
    @property
    def sha256(self):return canonical_sha(self.to_dict())

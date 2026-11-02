"""Finite-sample one-sided bounds for independent paired evaluation units.

Empirical Bernstein: Maurer & Pontil (2009), Theorem 4, scaled to [a,b].
https://arxiv.org/abs/0907.3740 . Families allocate alpha over endpoints AND
planned looks; dependence between those tests does not invalidate the union bound.
"""
from dataclasses import asdict,dataclass
import hashlib
import json
import math
import numpy as np

METHOD='graygo/empirical-bernstein/1'


def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def lower_bound(values,*,minimum=-1.,maximum=1.,delta=.05):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():raise ValueError('at least two finite independent units required')
    if not all(math.isfinite(v) for v in (minimum,maximum,delta)) or minimum>=maximum or not 0<delta<1:
        raise ValueError('finite ordered bounds and delta in (0,1) required')
    if (x<minimum).any() or (x>maximum).any():raise ValueError('observations exceed preregistered support')
    mean=float(x.mean());variance=float(x.var(ddof=1));log=math.log(2/delta)
    radius=math.sqrt(2*variance*log/len(x))+7*(maximum-minimum)*log/(3*(len(x)-1))
    return {'mean':mean,'sample_variance':variance,'lower':max(minimum,mean-radius),
            'radius':radius,'independent_units':len(x),'delta':delta,'support':[minimum,maximum],'method':METHOD}


@dataclass(frozen=True)
class ConfidenceFamily:
    name: str
    endpoints: tuple[str,...]
    looks: tuple[int,...]=(500,2000)
    alpha: float=.05
    method: str=METHOD

    def __post_init__(self):
        object.__setattr__(self,'endpoints',tuple(self.endpoints));object.__setattr__(self,'looks',tuple(self.looks))
        if not self.name or not self.endpoints or len(set(self.endpoints))!=len(self.endpoints) or any(not isinstance(e,str) or not e for e in self.endpoints):raise ValueError('named unique confidence endpoints required')
        if not self.looks or tuple(sorted(set(self.looks)))!=self.looks or any(type(n) is not int or n<4 or n%2 for n in self.looks):raise ValueError('increasing even game looks required')
        if not math.isfinite(self.alpha) or not 0<self.alpha<1 or self.method!=METHOD:raise ValueError('unsupported confidence plan')

    def to_dict(self):return json.loads(json.dumps(asdict(self)))
    @property
    def sha256(self):return canonical_sha(self.to_dict())
    @property
    def delta(self):return self.alpha/(len(self.endpoints)*len(self.looks))

    def estimate(self,endpoint,games,paired_values,*,minimum=-1.,maximum=1.):
        if endpoint not in self.endpoints or games not in self.looks:raise ValueError('unregistered endpoint or sample look')
        if len(paired_values)!=games//2:raise ValueError('one independent observation per color pair required')
        result=lower_bound(paired_values,minimum=minimum,maximum=maximum,delta=self.delta)
        return dict(result,endpoint=endpoint,games=games,family_sha256=self.sha256,simultaneous_alpha=self.alpha,comparisons=len(self.endpoints)*len(self.looks))


def validate_estimate(estimate,family,*,minimum=-1.,maximum=1.):
    """Check receipt arithmetic/bindings; its evidence hash still needs auditing."""
    e=estimate
    if e.get('endpoint') not in family.endpoints or e.get('games') not in family.looks or e.get('family_sha256')!=family.sha256 or e.get('method')!=family.method or e.get('delta')!=family.delta or e.get('support')!=[minimum,maximum] or e.get('simultaneous_alpha')!=family.alpha or e.get('comparisons')!=len(family.endpoints)*len(family.looks):raise ValueError('confidence receipt uses a different family/support')
    n=e.get('independent_units')
    if type(n) is not int or n!=e['games']//2 or n<2:raise ValueError('wrong independent sampling-unit count')
    if not all(isinstance(e.get(k),(int,float)) and math.isfinite(e[k]) for k in ('mean','sample_variance','lower','radius')):raise ValueError('nonfinite confidence receipt')
    if not minimum<=e['mean']<=maximum or not 0<=e['sample_variance']<=(maximum-minimum)**2*n/(4*(n-1))+1e-12:raise ValueError('receipt exceeds bounded variance/mean')
    log=math.log(2/family.delta);radius=math.sqrt(2*e['sample_variance']*log/n)+7*(maximum-minimum)*log/(3*(n-1))
    if not math.isclose(radius,e['radius'],rel_tol=0,abs_tol=1e-12) or not math.isclose(max(minimum,e['mean']-radius),e['lower'],rel_tol=0,abs_tol=1e-12):raise ValueError('confidence receipt arithmetic differs')

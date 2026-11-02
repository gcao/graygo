"""Frozen learned-policy collision benchmark and scientific-registry bindings."""
from dataclasses import dataclass
import math
from .confidence import canonical_sha
from .pilot_gates import CONFIRMATION_SEEDS
from .science_protocol import ScienceProtocol,fingerprint

PROBE_POLICY_VERSION='graygo/frozen-probe-policy/1'


@dataclass(frozen=True)
class ProbePolicyProtocol:
    name:str
    benchmark_sha256:str
    freeze_sha256:str
    policy_ids:tuple
    time_ceiling_seconds:float
    purpose:str='confirmation'
    curriculum:str='exposure'
    representation:str='average'
    catalog_sha256:str|None=None
    science:ScienceProtocol|None=None

    def __post_init__(self):
        try:registry=tuple(sorted(self.policy_ids.items())) if isinstance(self.policy_ids,dict) else tuple(tuple(row) for row in self.policy_ids)
        except TypeError as exc:raise ValueError('complete three-seed policy registry required') from exc
        object.__setattr__(self,'policy_ids',registry)
        if not isinstance(self.name,str) or not self.name or not fingerprint(self.benchmark_sha256) or not fingerprint(self.freeze_sha256):
            raise ValueError('named frozen complete benchmark/freeze receipt required')
        if any(len(row)!=2 for row in registry) or tuple(row[0] for row in registry)!=CONFIRMATION_SEEDS or any(not fingerprint(row[1]) for row in registry):
            raise ValueError('all three fixed probe policy fingerprints required')
        if self.purpose not in ('confirmation','diagnostic') or self.curriculum not in ('empty','exposure') or not isinstance(self.representation,str) or not self.representation:
            raise ValueError('fixed probe purpose/curriculum/representation required')
        if type(self.time_ceiling_seconds) not in (int,float) or not math.isfinite(self.time_ceiling_seconds) or not 0<self.time_ceiling_seconds<=16*3600:
            raise ValueError('finite probe-inference ceiling at most sixteen hours required')
        if self.purpose=='confirmation':
            if self.representation!='average' or not isinstance(self.science,ScienceProtocol) or self.science.purpose!='confirmation' or self.science.representation!='average':
                raise ValueError('confirmation probes require the actual average-policy scientific registry')
            if self.catalog_sha256!=self.science.catalog_sha256 or len({sha for _,sha in registry})!=3:
                raise ValueError('confirmation catalog or independent seed policies differ')
            expected={seed:sha for curriculum,rule,seed,sha in self.science.policy_ids if curriculum==self.curriculum and rule=='gray'}
            if dict(registry)!=expected:raise ValueError('probe policies differ from the frozen scientific base-rule arm')
        elif self.time_ceiling_seconds>3600:raise ValueError('diagnostic probe inference is bounded to one cumulative hour')
        if self.catalog_sha256 is not None and not fingerprint(self.catalog_sha256):raise ValueError('invalid exposure catalog fingerprint')

    def to_dict(self):
        return {'version':PROBE_POLICY_VERSION,'name':self.name,'benchmark_sha256':self.benchmark_sha256,
            'freeze_sha256':self.freeze_sha256,'policy_ids':[list(row) for row in self.policy_ids],
            'time_ceiling_seconds':self.time_ceiling_seconds,'purpose':self.purpose,'curriculum':self.curriculum,
            'representation':self.representation,'catalog_sha256':self.catalog_sha256,
            'science_protocol':None if self.science is None else self.science.to_dict(),
            'seeds':list(CONFIRMATION_SEEDS),'families':32,'required_passes_per_seed':24,
            'individual_gap':'1/20','required_collision_shift':'1/20',
            'unit':'one complete paired family, both roots and roles, all three original frozen policies',
            'inference':'descriptive fixed finite benchmark; no binomial population confidence bound'}

    @property
    def sha256(self):return canonical_sha(self.to_dict())

    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or value.get('version')!=PROBE_POLICY_VERSION:raise ValueError('unknown frozen probe-policy protocol')
        try:
            science=None if value['science_protocol'] is None else ScienceProtocol.from_dict(value['science_protocol'])
            result=cls(value['name'],value['benchmark_sha256'],value['freeze_sha256'],value['policy_ids'],value['time_ceiling_seconds'],
                value['purpose'],value['curriculum'],value['representation'],value['catalog_sha256'],science)
        except (KeyError,TypeError) as exc:raise ValueError('malformed frozen probe-policy protocol') from exc
        if result.to_dict()!=value:raise ValueError('frozen probe-policy criteria/registry changed')
        return result

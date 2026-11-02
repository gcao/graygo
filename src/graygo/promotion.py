"""Protected-pool statistical promotion gate and frozen evidence alignment.

Only conditional tested-matchup claims are made. Numerical bounds alone do not
supply the separate A2/9x9 readiness approval or training-seed certainty.
"""
from dataclasses import asdict,dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import numpy as np
from .confidence import ConfidenceFamily,canonical_sha
from .engine import Rules,State
from .evaluation import EVALUATION_VERSION,seed_for,validate_pair,FixedStarts,EmptyStarts

PROMOTION_VERSION='graygo/protected-promotion/1'


def fingerprint(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def agent_signature(identity):
    """Behavioral identity used to prevent relabeling a pilot as a candidate."""
    if 'weights_sha256' in identity:
        return canonical_sha({key:identity.get(key) for key in ('kind','model_class','weights_sha256','encoding_version')})
    if identity.get('kind') in ('random','tactical'):
        return canonical_sha({key:identity.get(key) for key in ('kind','version')})
    return canonical_sha(identity)


@dataclass(frozen=True)
class ProtectedPool:
    name: str
    version: int
    opponents: tuple # (stable label, immutable policy SHA-256)
    weights: tuple
    rules: Rules=Rules()
    opening_sha256: str=EmptyStarts().sha256

    def __post_init__(self):
        object.__setattr__(self,'opponents',tuple(tuple(p) for p in self.opponents));object.__setattr__(self,'weights',tuple(self.weights))
        if not self.name or type(self.version) is not int or self.version<1 or not 1<=len(self.opponents)<=10:raise ValueError('named versioned pool with at most ten protected opponents required')
        if any(len(p)!=2 or not isinstance(p[0],str) or not p[0] or not fingerprint(p[1]) for p in self.opponents):raise ValueError('frozen opponent labels/fingerprints required')
        if len({p[0] for p in self.opponents})!=len(self.opponents) or len({p[1] for p in self.opponents})!=len(self.opponents):raise ValueError('duplicate protected opponents')
        w=np.asarray(self.weights,dtype=float)
        if w.shape!=(len(self.opponents),) or not np.isfinite(w).all() or (w<0).any() or abs(w.sum()-1)>1e-12:raise ValueError('fixed normalized evaluation weights required')
        object.__setattr__(self,'weights',tuple(map(float,w/w.sum())))
        if not isinstance(self.rules,Rules) or not fingerprint(self.opening_sha256):raise ValueError('frozen rules/opening distribution required')
    def to_dict(self):return json.loads(json.dumps(asdict(self)))
    @property
    def sha256(self):return canonical_sha(self.to_dict())


@dataclass(frozen=True)
class PromotionProtocol:
    pool: ProtectedPool
    looks: tuple=(500,2000)
    alpha: float=.05
    minimum_pilot_pairs: int=250
    minimum_simulations: int=1000
    useful_improvement: float=.03
    regression_tolerance: float=-.15
    absolute_floor: float=-.10
    target_power: float=.80
    maximum_false_promotion: float=.05

    def __post_init__(self):
        object.__setattr__(self,'looks',tuple(self.looks))
        if self.looks not in ((500,2000),(500,2000,8000)) or self.alpha!=.05:raise ValueError('pre-candidate 500/2000 or declared 8000 look and simultaneous 95% coverage required')
        if (self.useful_improvement,self.regression_tolerance,self.absolute_floor,self.target_power,self.maximum_false_promotion)!=(.03,-.15,-.10,.80,.05):raise ValueError('promotion tolerances/power may not be changed silently')
        if type(self.minimum_pilot_pairs) is not int or self.minimum_pilot_pairs<2 or type(self.minimum_simulations) is not int or self.minimum_simulations<100:raise ValueError('declared calibration sample minima required')
    @property
    def family(self):
        keys=['aggregate-improvement']+[prefix+'/'+label for label,_ in self.pool.opponents for prefix in ('delta','utility')]
        return ConfidenceFamily('promotion/'+self.pool.sha256,tuple(keys),self.looks,self.alpha)
    def to_dict(self):
        return {'version':PROMOTION_VERSION,'pool':self.pool.to_dict(),'confidence':self.family.to_dict(),
            'stopping_rule':'first complete planned look clearing all criteria; later collected looks cannot be hidden',
            'minimum_pilot_pairs':self.minimum_pilot_pairs,'minimum_simulations':self.minimum_simulations,
            'useful_improvement':self.useful_improvement,'regression_tolerance':self.regression_tolerance,'absolute_floor':self.absolute_floor,
            'target_power':self.target_power,'maximum_false_promotion':self.maximum_false_promotion}
    @property
    def sha256(self):return canonical_sha(self.to_dict())
    @classmethod
    def from_dict(cls,value):
        if value.get('version')!=PROMOTION_VERSION:raise ValueError('unknown promotion protocol')
        p=value['pool'];pool=ProtectedPool(p['name'],p['version'],p['opponents'],p['weights'],Rules(**p['rules']),p['opening_sha256'])
        result=cls(pool,tuple(value['confidence']['looks']),value['confidence']['alpha'],value['minimum_pilot_pairs'],value['minimum_simulations'],
            value['useful_improvement'],value['regression_tolerance'],value['absolute_floor'],value['target_power'],value['maximum_false_promotion'])
        if result.to_dict()!=value:raise ValueError('promotion protocol constants or confidence coverage changed')
        return result

    def numerical_gate(self,candidate,incumbent,games):
        c,i=np.asarray(candidate,dtype=float),np.asarray(incumbent,dtype=float)
        shape=(games//2,len(self.pool.opponents))
        if games not in self.looks or c.shape!=shape or i.shape!=shape or not np.isfinite(c).all() or not np.isfinite(i).all() or (np.abs(c)>1).any() or (np.abs(i)>1).any():raise ValueError('aligned paired utilities at a registered sample look required')
        delta=c-i;family=self.family
        aggregate=family.estimate('aggregate-improvement',games,np.clip(delta@np.asarray(self.pool.weights),-2.,2.),minimum=-2.,maximum=2.)
        per=[]
        for column,(label,policy) in enumerate(self.pool.opponents):
            improvement=family.estimate('delta/'+label,games,delta[:,column],minimum=-2.,maximum=2.)
            absolute=family.estimate('utility/'+label,games,c[:,column])
            per.append({'opponent':label,'policy_sha256':policy,'improvement':improvement,'candidate_utility':absolute})
        criteria={'aggregate_improvement':aggregate['lower']>0,
                  'per_opponent_regression':all(p['improvement']['lower']>=self.regression_tolerance for p in per),
                  'absolute_floor':all(p['candidate_utility']['lower']>=self.absolute_floor for p in per)}
        return {'numerical_pass':all(criteria.values()),'criteria':criteria,'aggregate':aggregate,'opponents':per,
                'worst_matchup_lower':min(p['candidate_utility']['lower'] for p in per),'protocol_sha256':self.sha256,
                'scope':'fixed tested protected pool and opening distribution; not exact full-game exploitability'}


@dataclass(frozen=True)
class PairedEvidence:
    contract: dict
    rows: tuple
    sha256: str

    @classmethod
    def load(cls,root):
        root=Path(root);contract=json.loads((root/'plan.json').read_text())
        if contract['evaluation_version']!=EVALUATION_VERSION or contract['plan']['comparison']!='improvement' or contract['incumbent'] is None:raise ValueError('matched incumbent evidence required')
        for key in ('subject','opponent','incumbent'):
            if canonical_sha(contract[key])!=contract[key+'_sha256']:raise ValueError('frozen policy identity differs')
        if canonical_sha(contract['opening_distribution'])!=contract['opening_sha256']:raise ValueError('opening identity differs')
        rules=Rules(**contract['plan']['rules']);opening=contract['opening_distribution']
        if opening['kind']=='official-empty':sampler=EmptyStarts()
        elif opening['kind']=='fixed-public-opening-catalog':sampler=FixedStarts(tuple(State.from_dict(s) for s in opening['states']),label=opening['label'])
        else:raise ValueError('unsupported frozen opening distribution')
        db=sqlite3.connect(f'file:{root/"evidence.sqlite"}?mode=ro',uri=True)
        try:stored=list(db.execute('SELECT idx,uid,payload,sha256 FROM units ORDER BY idx'))
        finally:db.close()
        rows=[]
        for expected,(index,uid,payload,checksum) in enumerate(stored):
            if index!=expected or uid!=str(seed_for(contract['plan']['seed'],'unit',index)) or hashlib.sha256(payload.encode()).hexdigest()!=checksum:raise ValueError('paired evidence prefix/checksum differs')
            row=json.loads(payload);validate_pair(row,rules,True)
            if row['index']!=index or row['uid']!=uid:raise ValueError('paired unit metadata differs')
            root_state=sampler.sample(seed_for(int(uid),'opening'),rules)
            if row['root_sha256']!=hashlib.sha256(root_state.canonical_json().encode()).hexdigest():raise ValueError('paired opening differs')
            rows.append(row)
        return cls(contract,tuple(rows),canonical_sha({'contract':contract,'units':[canonical_sha(r) for r in rows]}))


def aligned_evidence(protocol,evidence,*,for_calibration=False):
    pool=protocol.pool;keys=[label for label,_ in pool.opponents]
    if set(evidence)!=set(keys):raise ValueError('protected opponent set is incomplete or changed')
    items=[evidence[key] for key in keys];first=items[0]
    c=[];i=[]
    for (label,sha),item in zip(pool.opponents,items):
        if canonical_sha({'contract':item.contract,'units':[canonical_sha(r) for r in item.rows]})!=item.sha256:raise ValueError('paired evidence changed after verification')
        contract=item.contract
        if contract['opponent_sha256']!=sha or contract['plan']['rules']!=json.loads(json.dumps(asdict(pool.rules))) or contract['opening_sha256']!=pool.opening_sha256:raise ValueError('protected rules/opponent/opening differs')
        if (contract['subject_sha256'],contract['incumbent_sha256'])!=(first.contract['subject_sha256'],first.contract['incumbent_sha256']):raise ValueError('different candidate/incumbent weights across opponents')
        if [(r['uid'],r['root_sha256']) for r in item.rows]!=[(r['uid'],r['root_sha256']) for r in first.rows]:raise ValueError('opening/color units are not aligned across comparisons')
        if not for_calibration and (contract['plan']['confidence_sha256']!=protocol.family.sha256 or tuple(contract['plan']['looks'])!=protocol.looks or contract['plan']['endpoint']!='delta/'+label or contract['plan'].get('decision_protocol_sha256')!=protocol.sha256 or not fingerprint(contract['plan'].get('calibration_sha256'))):raise ValueError('candidate data was not collected under this preregistered protected gate')
        c.append([r['utility'] for r in item.rows]);i.append([r['incumbent_utility'] for r in item.rows])
    candidate=np.asarray(c,dtype=float).T;incumbent=np.asarray(i,dtype=float).T
    return candidate,incumbent,first.contract['subject_sha256'],first.contract['incumbent_sha256']


def decision_from_evidence(protocol,evidence,games,*,calibration=None,readiness_verifier=None):
    c,i,subject,incumbent=aligned_evidence(protocol,evidence)
    if len(c)<games//2:raise ValueError('incomplete registered look')
    eligible=max(n for n in protocol.looks if n<=len(c)*2)
    if games!=eligible:raise ValueError('later completed planned look was hidden')
    numerical=protocol.numerical_gate(c[:games//2],i[:games//2],games)
    from .promotion_calibration import qualified_report
    qualified=bool(qualified_report(protocol,calibration) and subject not in calibration.get('pilot_subjects',[]) and incumbent not in calibration.get('pilot_subjects',[])
        and agent_signature(next(iter(evidence.values())).contract['subject']) not in calibration.get('pilot_agent_signatures',[])
        and agent_signature(next(iter(evidence.values())).contract['incumbent']) not in calibration.get('pilot_agent_signatures',[])
        and all(item.contract['plan'].get('calibration_sha256')==calibration['report_sha256'] for item in evidence.values()))
    readiness=None if readiness_verifier is None else readiness_verifier(subject,incumbent,evidence)
    ready=bool(isinstance(readiness,dict) and readiness.get('a2_passed') is True and readiness.get('board_smoke_passed') is True
               and fingerprint(readiness.get('a2_evidence_sha256')) and fingerprint(readiness.get('board_smoke_sha256'))
               and readiness.get('subject_sha256')==subject and readiness.get('incumbent_sha256')==incumbent)
    return dict(numerical,calibration_qualified=qualified,readiness=readiness,prerequisites_proven=ready,
        promotion_permitted=bool(numerical['numerical_pass'] and qualified and ready),subject_sha256=subject,incumbent_sha256=incumbent,
        evidence={key:value.sha256 for key,value in evidence.items()},training_seed_uncertainty='report separate training seeds; game counts do not remove it')

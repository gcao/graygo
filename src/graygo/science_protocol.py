"""Preregistered four-arm scientific contrast and conditional finite-cohort bounds.

One independent block holds both colors, both evaluation rules, all three frozen
training seeds, both training curricula and both opening contexts: 48 games.
Training-seed population uncertainty is not replaced by game-sampling precision.
"""
from dataclasses import dataclass
import math
import numpy as np
from .confidence import METHOD,canonical_sha,lower_bound
from .pilot_gates import CONFIRMATION_SEEDS

SCIENCE_VERSION='graygo/scientific-contrast/1'
CURRICULA=('empty','exposure')
CONTEXTS=('official-empty','heldout-gray')
RULES=('gray','remove')
METRICS=('u_G','u_E','interaction')
GAMES_PER_BLOCK=48


def policy_keys():return tuple((c,r,s) for c in CURRICULA for r in RULES for s in CONFIRMATION_SEEDS)
def fingerprint(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def endpoint(context,curriculum,metric,cohort):return '/'.join((context,curriculum,metric,cohort))


@dataclass(frozen=True)
class ScienceProtocol:
    name:str
    catalog_sha256:str
    policy_ids:tuple
    time_ceiling_seconds:float
    seed:int=0
    block_looks:tuple=(500,2000)
    alpha:float=.05
    purpose:str='confirmation'
    training_records:int=80000000
    representation:str='average'

    def __post_init__(self):
        registry=tuple(sorted((*key,sha) for key,sha in self.policy_ids.items())) if isinstance(self.policy_ids,dict) else tuple(sorted(tuple(row) for row in self.policy_ids))
        object.__setattr__(self,'policy_ids',registry);object.__setattr__(self,'block_looks',tuple(self.block_looks))
        if not isinstance(self.name,str) or not self.name or not fingerprint(self.catalog_sha256):raise ValueError('named frozen exposure catalog required')
        if tuple(row[:3] for row in registry)!=policy_keys() or any(len(row)!=4 or not fingerprint(row[3]) for row in registry):raise ValueError('all twelve fixed arm/seed policy fingerprints required')
        if type(self.seed) is not int or self.seed<0 or type(self.training_records) is not int or self.training_records<1:raise ValueError('fixed initialization cohort and record milestone required')
        if not math.isfinite(self.time_ceiling_seconds) or not 0<self.time_ceiling_seconds<=16*3600:raise ValueError('explicit scientific-evaluation ceiling at most sixteen hours required')
        if self.purpose not in ('confirmation','diagnostic') or self.alpha!=.05 or self.representation not in ('average','actor','episode-mixture'):raise ValueError('fixed scientific contrast/coverage settings required')
        if not self.block_looks or tuple(sorted(set(self.block_looks)))!=self.block_looks or any(type(n) is not int or n<2 for n in self.block_looks):raise ValueError('fixed increasing independent-block looks required')
        if self.purpose=='confirmation':
            if self.block_looks!=(500,2000) or self.training_records!=80000000:raise ValueError('confirmation protocol uses the declared 24k/96k allocation and 80M model milestone')
            if len({row[3] for row in registry})!=12:raise ValueError('confirmation arm/seed agents cannot be aliases')
        elif self.block_looks[-1]>2000:raise ValueError('diagnostic cannot exceed the scientific maximum')

    @property
    def endpoints(self):return tuple(endpoint(context,curriculum,metric,cohort) for context in CONTEXTS for curriculum in CURRICULA
                                    for metric in METRICS for cohort in ('mean3',*(f'seed{s}' for s in CONFIRMATION_SEEDS)))
    @property
    def delta(self):return self.alpha/(len(self.endpoints)*len(self.block_looks))
    @property
    def sha256(self):return canonical_sha(self.to_dict())
    @property
    def physical_looks(self):return tuple(n*GAMES_PER_BLOCK for n in self.block_looks)
    def to_dict(self):
        return {'version':SCIENCE_VERSION,'name':self.name,'catalog_sha256':self.catalog_sha256,'policy_ids':[list(row) for row in self.policy_ids],
            'time_ceiling_seconds':self.time_ceiling_seconds,'seed':self.seed,'block_looks':list(self.block_looks),'physical_game_looks':list(self.physical_looks),
            'alpha':self.alpha,'method':METHOD,'endpoints':list(self.endpoints),'delta':self.delta,'purpose':self.purpose,
            'training_records':self.training_records,'representation':self.representation,'seeds':list(CONFIRMATION_SEEDS),
            'curricula':list(CURRICULA),'evaluation_rules':list(RULES),'opening_contexts':list(CONTEXTS),'games_per_block':GAMES_PER_BLOCK,
            'unit':'one matched block: curricula x contexts x seeds x rules x paired colors',
            'thresholds':{'u_G_lower_strictly_above':.05,'interaction_lower_strictly_above':.05,'each_seed_point_estimates_strictly_above':0.},
            'primary':'official-empty/exposure','secondary':'official-empty/empty','heldout':'diagnostic only',
            'inference':'conditional on the fixed registered three-seed cohort; no training-seed population confidence claim'}
    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or value.get('version')!=SCIENCE_VERSION:raise ValueError('unknown scientific protocol')
        try:result=cls(value['name'],value['catalog_sha256'],value['policy_ids'],value['time_ceiling_seconds'],value['seed'],tuple(value['block_looks']),
                       value['alpha'],value['purpose'],value['training_records'],value['representation'])
        except (KeyError,TypeError) as exc:raise ValueError('malformed scientific protocol') from exc
        if result.to_dict()!=value:raise ValueError('scientific constants, units, cohorts or simultaneous coverage changed')
        return result

    def summarize(self,values,total_games):
        if type(total_games) is not int or total_games not in self.physical_looks:raise ValueError('predeclared scientific game look required')
        x=np.asarray(values,dtype=np.float64)
        if x.ndim!=5 or x.shape[1:]!=(2,2,3,2) or not np.isfinite(x).all() or (np.abs(x)>1).any() or not np.equal(x*2,np.round(x*2)).all():
            raise ValueError('complete color-pair utilities for both contexts/curricula/three seeds/rules required')
        if len(x)>self.block_looks[-1]:raise ValueError('scientific evidence exceeds its maximum allocation')
        blocks=total_games//GAMES_PER_BLOCK
        available=len(x);completed=[n for n in self.block_looks if n<=available]
        if not completed or blocks!=max(completed):raise ValueError('look is incomplete or a later completed planned look was hidden')
        x=x[:blocks];estimates={}
        for context_i,context in enumerate(CONTEXTS):
            for curriculum_i,curriculum in enumerate(CURRICULA):
                pair=x[:,context_i,curriculum_i]
                metrics={'u_G':pair[:,:,0],'u_E':pair[:,:,1],'interaction':pair[:,:,0]-pair[:,:,1]}
                for metric,per_seed in metrics.items():
                    support=(-2.,2.) if metric=='interaction' else (-1.,1.)
                    for cohort,vector in [('mean3',per_seed.mean(axis=1)),*((f'seed{s}',per_seed[:,i]) for i,s in enumerate(CONFIRMATION_SEEDS))]:
                        key=endpoint(context,curriculum,metric,cohort)
                        estimate=lower_bound(vector,minimum=support[0],maximum=support[1],delta=self.delta)
                        estimates[key]=dict(estimate,endpoint=key,blocks=blocks,total_physical_games=total_games,
                            protocol_sha256=self.sha256,simultaneous_alpha=self.alpha,comparisons=len(self.endpoints)*len(self.block_looks),
                            unit='matched 48-game block; seed-specific endpoints retain the same block grouping')
        decisions={}
        for curriculum in CURRICULA:
            selected={metric:estimates[endpoint('official-empty',curriculum,metric,'mean3')] for metric in ('u_G','interaction')}
            points={s:{metric:estimates[endpoint('official-empty',curriculum,metric,f'seed{s}')]['mean'] for metric in ('u_G','interaction')} for s in CONFIRMATION_SEEDS}
            positive=all(value>0 for values in points.values() for value in values.values())
            decisions[curriculum]={'numerical_pass':positive and all(e['lower']>.05 for e in selected.values()),
                'aggregate_bounds':selected,'seed_point_estimates':points,'all_seed_points_positive':positive}
        return {'version':'graygo/scientific-contrast-summary/1','protocol_sha256':self.sha256,'blocks':blocks,'available_blocks':available,
            'physical_games':total_games,'estimates':estimates,'primary':decisions['exposure'],'secondary':decisions['empty'],
            'numerical_performance_pass':decisions['exposure']['numerical_pass'],'purpose':self.purpose,
            'gray_specialization_demonstrated':False,'scope':'conditional fixed-cohort numerical contrast; A2, training provenance/scope, matched exposure and independent probe coverage remain separate prerequisites'}

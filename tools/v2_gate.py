"""Versioned v2 release gate: anytime paired tests, risk per actual attempt."""
from dataclasses import dataclass,asdict
import numpy as np
from graygo.betting import BettingFamily,estimate,VERSION

POLICY='paired-betting/v4'

@dataclass(frozen=True)
class V2GatePlan:
    rounds:int=2**31
    looks:tuple=(500,2000,4000)
    seed:int=11073
    alpha:float=.05
    regression:float=-.15
    floor:float=-.10
    pool_gain_policy:str=POLICY
    def __post_init__(self):
        object.__setattr__(self,'looks',tuple(self.looks))
        if not 0<self.alpha<=.05 or self.regression!=-.15 or self.floor!=-.10:raise ValueError('declared v2 release tolerances required')
        if not self.looks or tuple(sorted(set(self.looks)))!=self.looks or any(type(n) is not int or n<4 or n%2 for n in self.looks):raise ValueError('increasing even gate looks required')
    def identity(self):return {'version':POLICY,'method':VERSION,**asdict(self)}
    def family(self,round_index,labels):
        if type(round_index) is not int or round_index<0 or not labels:raise ValueError('actual attempt and learned opponents required')
        return BettingFamily(round_index+1,self.looks,('head',*[kind+'/'+k for k in labels for kind in ('delta','utility')]),self.alpha)
    def numeric(self,round_index,head,pool,incumbent,games):
        family=self.family(round_index,tuple(pool))
        if games not in self.looks or len(head)!=games//2 or any(len(v)!=games//2 for v in [*pool.values(),*incumbent.values()]):raise ValueError('paired evidence count differs')
        bound=lambda x,lo,hi:estimate(x,alpha=family.alpha,minimum=lo,maximum=hi)
        hb=bound(head,-1,1);opponents={}
        for key,values in pool.items():
            opponents[key]={'delta':bound(np.asarray(values)-np.asarray(incumbent[key]),-2,2),'utility':bound(values,-1,1)}
        criteria={'head_to_head_win_score_lower_above_half':hb['lower']>0,
                  'no_historical_regression':all(v['delta']['lower']>=self.regression for v in opponents.values()),
                  'absolute_floor':all(v['utility']['lower']>=self.floor for v in opponents.values())}
        gain=float(np.mean([np.mean(np.asarray(pool[k])-np.asarray(incumbent[k])) for k in pool]))
        return {'numeric_pass':all(criteria.values()),'criteria':criteria,'head':hb,
                'head_win_score':{'mean':(hb['mean']+1)/2,'lower':(hb['lower']+1)/2},'opponents':opponents,
                'promotion_policy':POLICY,'family_sha256':family.sha256,'alpha_per_attempt':family.alpha,
                'actual_attempt':family.attempt,'stop_for_futility':False,'pool_gain_required':False,
                'aggregate':{'mean':gain,'diagnostic_only':True},'minimum_pool_gain_lower':None}

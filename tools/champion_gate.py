"""Independent-seat candidate gates with frozen historical/baseline opponents."""
from dataclasses import asdict,dataclass
import json
from pathlib import Path
import time
import numpy as np
from graygo.confidence import ConfidenceFamily,canonical_sha
from graygo.evaluation import EvaluationPlan,EvaluationRun,EmptyStarts,seed_for
from graygo.evaluation_evidence import FrozenEvaluationEvidence
from graygo.manifest import write_report
from graygo.storage import DEFAULT_RESERVE,digest as _digest
from tools.nfsp_resources import require

def digest(path):return _digest(Path(path))

VERSION='graygo/champion-loop-gate/2'
POSITIVE_POOL_GAIN='positive-pool-gain/v2'
LEGACY_POOL_GAIN='minimum-pool-gain/v1'
HEAD_TO_HEAD_HISTORY='head-to-head-history/v3'


def gate_root(root,plan):
    if plan.pool_gain_policy=='paired-betting/v4':return Path(root)/'paired-betting-v4'
    if plan.pool_gain_policy==HEAD_TO_HEAD_HISTORY:return Path(root)/'head-to-head-history-v3'
    return Path(root)/'positive-pool-gain-v2' if plan.pool_gain_policy==POSITIVE_POOL_GAIN else Path(root)


def pool_gain_passes(lower,plan):
    return lower>0 if plan.pool_gain_policy!=LEGACY_POOL_GAIN else lower>=plan.improvement


@dataclass(frozen=True)
class GatePlan:
    rounds:int
    looks:tuple=(500,2000)
    seed:int=73
    alpha:float=.05
    improvement:float=0.
    regression:float=-.15
    floor:float=-.10
    pool_gain_policy:str=HEAD_TO_HEAD_HISTORY

    def __post_init__(self):
        object.__setattr__(self,'looks',tuple(self.looks))
        require(type(self.rounds) is int and self.rounds>0,'fixed finite round inventory required')
        require(type(self.seed) is int and self.seed>=0,'nonnegative gate seed required')
        require(self.looks and tuple(sorted(set(self.looks)))==self.looks and all(type(n) is int and n>=4 and n%2==0 for n in self.looks),'fixed increasing paired looks required')
        require(isinstance(self.alpha,(int,float)) and 0<self.alpha<=.05,'positive promotion alpha at most .05 required')
        require(self.pool_gain_policy in (POSITIVE_POOL_GAIN,LEGACY_POOL_GAIN,HEAD_TO_HEAD_HISTORY),'versioned pool-gain policy required')
        expected=.03 if self.pool_gain_policy==LEGACY_POOL_GAIN else 0.
        require((self.improvement,self.regression,self.floor)==(expected,-.15,-.10),'fixed promotion tolerances required')
    def identity(self):return asdict(self)

    def family(self,round_index,labels):
        require(0<=round_index<self.rounds and labels,'fixed round and nonempty pool required')
        endpoints=('head','aggregate',*[kind+'/'+label for label in labels for kind in ('delta','utility')])
        return ConfidenceFamily('champion-loop/'+str(round_index),endpoints,self.looks,self.alpha/self.rounds)

    def numeric(self,round_index,head,pool,incumbent,games):
        labels=tuple(pool);family=self.family(round_index,labels)
        head_bound=family.estimate('head',games,head,minimum=-1,maximum=1)
        deltas=np.stack([np.asarray(pool[k])-np.asarray(incumbent[k]) for k in labels],axis=1)
        aggregate=family.estimate('aggregate',games,deltas.mean(1),minimum=-2,maximum=2)
        opponents={}
        for k in labels:
            opponents[k]={'delta':family.estimate('delta/'+k,games,np.asarray(pool[k])-np.asarray(incumbent[k]),minimum=-2,maximum=2),
                          'utility':family.estimate('utility/'+k,games,pool[k],minimum=-1,maximum=1)}
        criteria={'head_to_head_win_score_lower_above_half':head_bound['lower']>0,
                  'pool_gain':pool_gain_passes(aggregate['lower'],self),
                  'no_historical_regression':all(v['delta']['lower']>=self.regression for v in opponents.values()),
                  'absolute_floor':all(v['utility']['lower']>=self.floor for v in opponents.values())}
        pool_required=self.pool_gain_policy!=HEAD_TO_HEAD_HISTORY
        pool_positive=criteria['pool_gain']
        if not pool_required:del criteria['pool_gain']
        return {'numeric_pass':all(criteria.values()),'criteria':criteria,'head':head_bound,'aggregate':aggregate,
                'promotion_policy':self.pool_gain_policy,'minimum_pool_gain_lower':self.improvement,
                'pool_gain_required':pool_required,'diagnostics':{'pool_gain':pool_positive},
                # Rejection-only futility checks cannot authorize a promotion.
                'stop_for_futility':(head_bound['mean']+head_bound['radius']<=0
                    or (pool_required and (aggregate['mean']+aggregate['radius']<=0 if self.pool_gain_policy==POSITIVE_POOL_GAIN
                        else aggregate['mean']+aggregate['radius']<self.improvement))
                    or any(v['delta']['mean']+v['delta']['radius']<self.regression
                        or v['utility']['mean']+v['utility']['radius']<self.floor for v in opponents.values())),
                'head_win_score':{'mean':(head_bound['mean']+1)/2,'lower':(head_bound['lower']+1)/2},
                'opponents':opponents,'family_sha256':family.sha256,'loop_family_alpha':self.alpha,
                'alpha_per_round':self.alpha/self.rounds}


class IndependentGate:
    def __init__(self,plan,*,reserve_bytes=DEFAULT_RESERVE,protected_authorizer=None):
        self.plan=plan;self.reserve=reserve_bytes;self.protected_authorizer=protected_authorizer
        self.identity={'version':VERSION,'plan':plan.identity(),'formal_protected_authorizer':protected_authorizer is not None,'implementation_sha256':digest(__file__)}
        self.sha256=canonical_sha(self.identity)

    def evaluate(self,candidate,champion,pool,root,*,round_index,time_cap_seconds,stop_requested=lambda:False):
        root=gate_root(root,self.plan);root.mkdir(parents=True,exist_ok=True);labels=tuple(pool)
        require(pool and 'head' not in pool and len(pool)<=10,'one to ten frozen baseline/historical opponents required')
        require(candidate.identity.get('origin',{}).get('rules') is not None,'candidate rules binding required')
        from graygo.engine import Rules
        rules=Rules(**candidate.identity['origin']['rules'])
        family=self.plan.family(round_index,labels);openings=EmptyStarts()
        contract={'version':VERSION,'candidate':candidate.sha256,'champion':champion.sha256,
                  'pool':{k:v.sha256 for k,v in pool.items()},'round_index':round_index,
                  'family':family.to_dict(),'rules':asdict(rules),'openings':openings.identity}
        if self.plan.pool_gain_policy==HEAD_TO_HEAD_HISTORY:contract['promotion_policy']=HEAD_TO_HEAD_HISTORY
        path=root/'gate-plan.json'
        if path.exists():require(json.loads(path.read_text())==contract,'gate identities/plan changed during resume')
        else:write_report(path,contract,reserve_bytes=self.reserve)
        seed=seed_for(self.plan.seed,'round',round_index);runs={}
        for label,opponent in [('head',champion),*pool.items()]:
            plan=EvaluationPlan('champion-loop',candidate.label,'head' if label=='head' else 'delta/'+label,
                rules,seed,self.plan.looks,family.sha256,'utility' if label=='head' else 'improvement')
            target=root/label
            runs[label]=EvaluationRun(plan,target,candidate,opponent,openings=openings,
                incumbent=None if label=='head' else champion,resume=(target/'plan.json').exists(),reserve_bytes=self.reserve)
        deadline=time.monotonic()+time_cap_seconds;decision=None
        # Previously collected later data cannot be hidden on recovery.
        available=max((len(run.records())*2 for run in runs.values()),default=0)
        for games in self.plan.looks:
            if games<available:continue
            for label,run in runs.items():
                remaining=deadline-time.monotonic()
                if remaining<=0 or stop_requested():return dict(status='budget_limited',numeric_pass=False,promotion_permitted=False)
                result=run.run(games,time_cap_seconds=remaining,stop_requested=stop_requested)
                if result['status']!='complete':return dict(status=result['status'],numeric_pass=False,promotion_permitted=False)
            rows={k:run.records()[:games//2] for k,run in runs.items()}
            decision=self.plan.numeric(round_index,[sum(float(np.sign(g['subject_utility'])) for g in r['subject'])/2 for r in rows['head']],
                {k:[r['utility'] for r in rows[k]] for k in labels},
                {k:[r['incumbent_utility'] for r in rows[k]] for k in labels},games)
            if decision['numeric_pass'] or decision['stop_for_futility']:break
        decision.update(status='complete',candidate_sha256=candidate.sha256,champion_sha256=champion.sha256,
            evidence={k:FrozenEvaluationEvidence.load(run.root).sha256 for k,run in runs.items()},
            promotion_permitted=False,experimental_promotion_permitted=decision['numeric_pass'])
        if decision['numeric_pass'] and self.protected_authorizer is not None:
            protected=self.protected_authorizer(candidate,champion,{k:run.root for k,run in runs.items()})
            allowed=(protected.get('promotion_permitted') is True and protected.get('prerequisites_proven') is True
                and protected.get('calibration_qualified') is True and protected.get('subject_sha256')==candidate.sha256
                and protected.get('incumbent_sha256')==champion.sha256)
            decision.update(protected_decision=protected,promotion_permitted=allowed)
        write_report(root/'decision.json',decision,reserve_bytes=self.reserve)
        return decision

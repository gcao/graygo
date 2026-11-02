"""Whole-gate pre-candidate simulation from actual paired pilot outcomes.

Draw whole cross-opponent opening/color units. Counterfactual candidate means
are formed by a declared mixture with +/-1 outcomes; incumbent outcomes and the
pilot's joint opening dependence remain paired. Infeasible scenarios do not pass.
"""
from dataclasses import dataclass
import math
import time
import numpy as np
from .confidence import canonical_sha,lower_bound
from .evaluation import seed_for
from .promotion import aligned_evidence,agent_signature,fingerprint

CALIBRATION_VERSION='graygo/promotion-calibration/1'


def scenario_definitions(protocol,incumbent_means):
    m=len(protocol.pool.opponents);w=np.asarray(protocol.pool.weights);zero=np.zeros(m);useful=np.full(m,.03)
    cases=[{'name':'zero-improvement','delta':zero.tolist(),'readiness':True},
           {'name':'uniform-plus-0.03','delta':useful.tolist(),'readiness':True},
           {'name':'missing-a2','delta':useful.tolist(),'readiness':False},
           {'name':'missing-board-smoke','delta':useful.tolist(),'readiness':False}]
    if m<2:cases.append({'name':'nontransitive','error':'nontransitive gains/losses require at least two protected opponents'})
    else:
        j=int(np.argmin(w));d=np.full(m,(.03+.05*w[j])/(1-w[j]));d[j]=-.05
        cases.append({'name':'nontransitive','delta':d.tolist(),'readiness':True})
    for j,(label,_) in enumerate(protocol.pool.opponents):
        for kind,loss in (('regression-at',-.15),('regression-beyond',-.20),('floor-violation',-.15-float(incumbent_means[j]))):
            name=kind+'/'+label
            if w[j]>=1:cases.append({'name':name,'error':'isolated protected comparison has no other evaluation weight'});continue
            d=np.full(m,(.03-w[j]*loss)/(1-w[j]));d[j]=loss
            cases.append({'name':name,'delta':d.tolist(),'readiness':True})
    return cases


@dataclass
class PilotPopulation:
    candidate: np.ndarray
    incumbent: np.ndarray
    subjects: tuple
    evidence: dict
    usable_for_promotion: bool
    agent_signatures: tuple=()

    @classmethod
    def from_evidence(cls,protocol,evidence):
        c,i,s,t=aligned_evidence(protocol,evidence,for_calibration=True)
        if len(c)<2:raise ValueError('paired calibration pilot has too few units')
        first=next(iter(evidence.values())).contract
        signatures=tuple(agent_signature(first[key]) for key in ('subject','incumbent'))
        return cls(c,i,(s,t),{key:value.sha256 for key,value in evidence.items()},True,signatures)
    @classmethod
    def synthetic(cls,candidate,incumbent):
        c,i=np.asarray(candidate,dtype=float),np.asarray(incumbent,dtype=float)
        if c.shape!=i.shape or c.ndim!=2 or len(c)<2:raise ValueError('matching synthetic paired matrices required')
        return cls(c,i,('synthetic-subject','synthetic-incumbent'),{},False)
    @property
    def sha256(self):
        return canonical_sha({'candidate':self.candidate.tolist(),'incumbent':self.incumbent.tolist(),'subjects':list(self.subjects),'evidence':self.evidence,'usable_for_promotion':self.usable_for_promotion,'agent_signatures':list(self.agent_signatures)})


def calibrate(protocol,pilot,*,simulations=2000,seed=0,time_cap_seconds=600,progress=None,save_progress=None,stop_requested=lambda:False):
    c,i=np.asarray(pilot.candidate,dtype=float),np.asarray(pilot.incumbent,dtype=float);m=len(protocol.pool.opponents)
    if c.shape!=i.shape or c.ndim!=2 or c.shape[1]!=m or len(c)<2 or not np.isfinite(c).all() or not np.isfinite(i).all() or (np.abs(c)>1).any() or (np.abs(i)>1).any():raise ValueError('bounded paired pilot data required')
    if type(simulations) is not int or simulations<2 or type(seed) is not int or seed<0 or not math.isfinite(time_cap_seconds) or time_cap_seconds<=0:raise ValueError('fixed simulation seed/count and time ceiling required')
    definitions=scenario_definitions(protocol,i.mean(0));plan={'version':CALIBRATION_VERSION,'protocol_sha256':protocol.sha256,'pilot_sha256':pilot.sha256,
        'simulations':simulations,'seed':seed,'definitions':definitions,'simulation':'whole paired rows + common cross-opponent uniform mixture; all planned looks; all criteria',
        'mc_family_alpha':.05,'mc_tail_budget':.05/(2*len(definitions))}
    plan_sha=canonical_sha(plan)
    if progress is None:progress={'plan_sha256':plan_sha,'scenarios':{},'elapsed_seconds':0.}
    if progress['plan_sha256']!=plan_sha:raise ValueError('calibration resume plan/pilot/protocol differs')
    previous_elapsed=progress.get('elapsed_seconds',0.)
    if not isinstance(previous_elapsed,(int,float)) or not math.isfinite(previous_elapsed) or previous_elapsed<0:raise ValueError('invalid calibration elapsed clock')
    started=time.monotonic();reports=[];complete=True
    def checkpoint():
        progress['elapsed_seconds']=previous_elapsed+time.monotonic()-started
        if save_progress is not None:save_progress(progress)
    checkpoint()
    for definition in definitions:
        name=definition['name'];prior=progress['scenarios'].setdefault(name,{'completed':0,'passes':0,'first_look_counts':{str(n):0 for n in protocol.looks}})
        if set(prior['first_look_counts'])!={str(n) for n in protocol.looks} or not 0<=prior['passes']<=prior['completed']<=simulations or sum(prior['first_look_counts'].values())!=prior['passes']:raise ValueError('invalid calibration counters')
        error=definition.get('error');delta=np.asarray(definition.get('delta',np.zeros(m)),dtype=float);target=i.mean(0)+delta
        if error is None and ((target<-1).any() or (target>1).any()):error='counterfactual utility mean is infeasible on [-1,1]'
        if error is not None:
            reports.append({'name':name,'status':'infeasible','error':error});continue
        target=np.clip(target,-1,1);base=c.mean(0);replacement=np.where(target>=base,1.,-1.);denominator=np.abs(replacement-base)
        theta=np.divide(np.abs(target-base),denominator,out=np.zeros(m),where=denominator>0)
        if (theta>1+1e-12).any():raise ValueError('invalid counterfactual mixture')
        theta=np.clip(theta,0,1)
        truth={'a2_and_smoke':definition['readiness'],'aggregate_improvement':float(delta@np.asarray(protocol.pool.weights))>0,
               'regression':bool((delta>=protocol.regression_tolerance).all()),'absolute_floor':bool((target>=protocol.absolute_floor).all())}
        violated=not all(truth.values())
        for trial in range(prior['completed'],simulations):
            if stop_requested() or previous_elapsed+time.monotonic()-started>=time_cap_seconds:complete=False;break
            rng=np.random.default_rng(seed_for(seed,'promotion-calibration',name,trial));n=max(protocol.looks)//2
            indices=rng.integers(0,len(c),n);candidate=c[indices].copy();incumbent=i[indices]
            choose=rng.random((n,1))<theta[None,:];candidate=np.where(choose,replacement[None,:],candidate)
            passed_at=None
            for games in protocol.looks:
                if definition['readiness'] and protocol.numerical_gate(candidate[:games//2],incumbent[:games//2],games)['numerical_pass']:
                    passed_at=games;break
            prior['completed']+=1
            if passed_at is not None:prior['passes']+=1;prior['first_look_counts'][str(passed_at)]+=1
            if prior['completed']%100==0:checkpoint()
        trials=prior['completed'];passes=prior['passes'];lower=upper=None
        if trials>=2:
            values=np.r_[np.ones(passes),np.zeros(trials-passes)]
            lower=lower_bound(values,minimum=0,maximum=1,delta=plan['mc_tail_budget'])['lower']
            upper=1-lower_bound(1-values,minimum=0,maximum=1,delta=plan['mc_tail_budget'])['lower']
        reports.append({'name':name,'status':'complete' if trials==simulations else 'budget_limited','trials':trials,'passes':passes,
            'pass_rate':None if not trials else passes/trials,'pass_rate_lower':lower,'pass_rate_upper':upper,'criterion_truth':truth,
            'criterion_violated':violated,'target_candidate_means':target.tolist(),'incumbent_means':i.mean(0).tolist(),
            'mixture_probability':theta.tolist(),'replacement':replacement.tolist(),'first_look_counts':dict(prior['first_look_counts'])})
        checkpoint()
        if not complete:break
    enough=pilot.usable_for_promotion and len(c)>=protocol.minimum_pilot_pairs and simulations>=protocol.minimum_simulations
    all_complete=complete and len(reports)==len(definitions) and all(r['status']=='complete' for r in reports)
    useful=next((r for r in reports if r['name']=='uniform-plus-0.03'),{})
    false_cases=[r for r in reports if r.get('criterion_violated')]
    power=bool(useful.get('status')=='complete' and not useful.get('criterion_violated',True) and useful['pass_rate_lower']>=protocol.target_power)
    errors=bool(false_cases and all(r['status']=='complete' and r['pass_rate_upper']<=protocol.maximum_false_promotion for r in false_cases))
    reasons=[]
    if not pilot.usable_for_promotion:reasons.append('synthetic pilot cannot qualify a real protected gate')
    if len(c)<protocol.minimum_pilot_pairs:reasons.append('insufficient separately designated pilot units')
    if simulations<protocol.minimum_simulations:reasons.append('insufficient Monte Carlo replications')
    if not all_complete:reasons.append('incomplete or infeasible required calibration scenario')
    if not power:reasons.append('uniform +0.03 power lower bound is below 0.80')
    if not errors:reasons.append('false-promotion upper bound exceeds 0.05 or is incomplete')
    checkpoint()
    report={'version':CALIBRATION_VERSION,'plan':plan,'protocol_sha256':protocol.sha256,'pilot_sha256':pilot.sha256,
        'pilot_subjects':list(pilot.subjects),'pilot_agent_signatures':list(pilot.agent_signatures),'pilot_evidence':pilot.evidence,'pilot_pairs':len(c),'usable_for_promotion':pilot.usable_for_promotion,
        'simulations':simulations,'complete':all_complete,'power_criterion_passed':power,'false_promotion_criterion_passed':errors,
        'passed':bool(enough and all_complete and power and errors),'status':'calibrated' if enough and all_complete and power and errors else 'not_calibrated',
        'reasons':reasons,'scenarios':reports,'paired_pilot_variance':np.var(c-i,axis=0,ddof=1).tolist(),
        'paired_pilot_covariance':np.atleast_2d(np.cov((c-i).T,ddof=1)).tolist(),'elapsed_this_segment_seconds':time.monotonic()-started,'cumulative_elapsed_seconds':progress['elapsed_seconds'],
        'scope':'conditional on preregistered pilot-derived outcome models; no transferable power guarantee for arbitrary candidates'}
    report['report_sha256']=canonical_sha(report)
    return report,progress


def qualified_report(protocol,report):
    if not isinstance(report,dict) or report.get('version')!=CALIBRATION_VERSION or report.get('protocol_sha256')!=protocol.sha256 or report.get('report_sha256')!=canonical_sha({k:v for k,v in report.items() if k!='report_sha256'}):return False
    if report.get('passed') is not True or report.get('complete') is not True or report.get('usable_for_promotion') is not True or report.get('pilot_pairs',0)<protocol.minimum_pilot_pairs or report.get('simulations',0)<protocol.minimum_simulations:return False
    if len(report.get('pilot_subjects',[]))!=2 or len(report.get('pilot_agent_signatures',[]))!=2 or not all(fingerprint(x) for x in report['pilot_subjects']+report['pilot_agent_signatures']):return False
    if set(report.get('pilot_evidence',{}))!={label for label,_ in protocol.pool.opponents} or not all(fingerprint(x) for x in report['pilot_evidence'].values()):return False
    expected={d['name'] for d in scenario_definitions(protocol,np.zeros(len(protocol.pool.opponents)))}
    scenarios=report.get('scenarios',[]);simulations=report['simulations'];mc_delta=.05/(2*len(expected))
    if len(scenarios)!=len(expected) or {r.get('name') for r in scenarios}!=expected or any(r.get('status')!='complete' for r in scenarios):return False
    if report.get('plan',{}).get('mc_tail_budget')!=mc_delta:return False
    for row in scenarios:
        n=row.get('trials');passes=row.get('passes')
        if type(n) is not int or type(passes) is not int or n!=simulations or not 0<=passes<=n:return False
        looks=row.get('first_look_counts',{})
        if set(looks)!={str(x) for x in protocol.looks} or sum(looks.values())!=passes:return False
        values=np.r_[np.ones(passes),np.zeros(n-passes)]
        low=lower_bound(values,minimum=0,maximum=1,delta=mc_delta)['lower']
        upper=1-lower_bound(1-values,minimum=0,maximum=1,delta=mc_delta)['lower']
        if not math.isclose(row.get('pass_rate_lower',-1),low,abs_tol=1e-12,rel_tol=0) or not math.isclose(row.get('pass_rate_upper',-1),upper,abs_tol=1e-12,rel_tol=0):return False
    useful=next(r for r in scenarios if r['name']=='uniform-plus-0.03')
    return bool(not useful.get('criterion_violated',True) and useful.get('pass_rate_lower',0)>=.8 and all(r.get('pass_rate_upper',1)<=.05 for r in scenarios if r.get('criterion_violated')))

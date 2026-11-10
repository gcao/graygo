"""Pre-launch power and null simulations of the new paired gate; no promotions."""
import argparse,json,math,sqlite3
from pathlib import Path
import numpy as np
from graygo.betting import stakes
from graygo.confidence import canonical_sha
from graygo.manifest import write_report,source_digest
from tools.v2_gate import V2GatePlan


def rows(path):
    db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
    try:return [json.loads(r[0]) for r in db.execute('SELECT payload FROM units ORDER BY idx')]
    finally:db.close()


def tilted(u,base,target):
    lam=np.zeros(len(target))
    for _ in range(100):
        z=np.log(base)+u@lam;z-=z.max();p=np.exp(z);p/=p.sum();mean=p@u;error=mean-target
        if max(abs(error))<1e-9:return p
        centered=u-mean;cov=centered.T@(p[:,None]*centered)
        lam-=.8*np.linalg.solve(cov+np.eye(len(target))*1e-12,error)
    raise ValueError('power model moment fitting failed')


def simulation(u,p,alpha,trials,rng):
    # Keep each whole color-pair endpoint vector together. Fresh independent
    # multinomial increments preserve dependence between the registered looks.
    counts=np.zeros((trials,len(u)),np.int32);passed=np.zeros(trials,bool);head_passed=np.zeros(trials,bool)
    hist_passed=np.zeros(trials,bool);previous=0
    support=[(-1,1),(-2,2),(-2,2),(-2,2)];thresholds=[0.,-.15,-.15,-.15]
    logs=[np.log1p((u[:,i,None]-thresholds[i])*stakes(*support[i])) for i in range(4)]
    for games in (500,2000,4000):
        count=games//2;counts+=rng.multinomial(count-previous,p,size=trials).astype(np.int32);previous=count
        tests=[]
        for factors in logs:
            capitals=counts@factors;top=capitals.max(1)
            e=top+np.log(np.exp(capitals-top[:,None]).mean(1));tests.append(e>=math.log(1/alpha))
        passed|=np.logical_and.reduce(tests);head_passed|=tests[0];hist_passed|=np.logical_and.reduce(tests[1:])
    return {'head_power':float(head_passed.mean()),'historical_power':float(hist_passed.mean()),'joint_power':float(passed.mean())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--trials',type=int,default=10000);a=ap.parse_args()
    h=rows(a.evidence/'head/evidence.sqlite');columns=[np.array([np.mean([np.sign(g['subject_utility']) for g in r['subject']]) for r in h])]
    for label in ('historical-0','historical-1','historical-2'):
        r=rows(a.evidence/label/'evidence.sqlite');assert [v['uid'] for v in h]==[v['uid'] for v in r]
        columns.append(np.array([v['utility']-v['incumbent_utility'] for v in r]))
    x=np.stack(columns,1);u,counts=np.unique(x,axis=0,return_counts=True);base=counts/counts.sum();rng=np.random.default_rng(1102026)
    # These simulations are power estimates under fitted paired-outcome models.
    # Error control comes from the martingale/IUT proof, not a fitted null alone.
    alternative=tilted(u,base,np.array([.1,0,0,0]));results={}
    for attempt in (1,10,100):
        alpha=V2GatePlan().family(attempt-1,['a','b','c']).alpha
        results[str(attempt)]={'alpha':alpha,**simulation(u,alternative,alpha,a.trials,rng)}
    null=tilted(u,base,np.array([0.,0,0,0]));alpha=V2GatePlan().family(0,['a','b','c']).alpha
    null_result=simulation(u,null,alpha,a.trials,rng)
    historical_nulls={}
    for i in range(3):
        target=np.array([.1,0.,0.,0.]);target[i+1]=-.15
        distribution=tilted(u,base,target)
        historical_nulls[str(i)]=simulation(u,distribution,alpha,a.trials,rng)['joint_power']
    power_ok=all(r['joint_power']>=.8 for r in results.values())
    report={'version':'graygo/v2-gate-calibration/1','source_sha256':source_digest(),'evidence':str(a.evidence.resolve()),
        'gate_plan_sha256':canonical_sha(V2GatePlan().identity()),'trials_per_cell':a.trials,'games_cap':4000,'alternative':'true head score 55%, all historical deltas zero; empirical tilted color-pair distribution',
        'power':results,'historical_boundary_false_promotion':historical_nulls,'head_null_crossing':null_result['head_power'],'null_alpha':alpha,'power_target':.8,
        'power_check_passed':power_ok,'null_check_passed':null_result['head_power']<=alpha and all(v<=alpha for v in historical_nulls.values()),'promotion_permitted':False,'error_control':'fixed-mixture supermartingales, Ville, conjunctive intersection-union; telescoping actual-attempt alpha'}
    write_report(a.output,report);print(json.dumps(report,indent=2))
    if not power_ok or not report['null_check_passed']:raise SystemExit('gate power target missed; do not launch')

if __name__=='__main__':main()

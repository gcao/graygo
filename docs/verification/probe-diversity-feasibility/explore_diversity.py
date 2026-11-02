"""Bounded candidate-independent feasibility search; no reachable-family claim."""
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
from graygo.engine import State, Rules, neighbors, groups
from graygo.manifest import source_digest, make_manifest, write_report
from graygo.probe_certification import certify_one_round, collision_range, _one_round_matrix, CertificateBudget
from graygo.probe_benchmark import strategy_family_id, rule_sensitive
from graygo.solving import SolveBudgetExceeded
from graygo.storage import digest

ROOT=Path(__file__).resolve().parent
SOURCE=source_digest()
PLAN={'purpose':'candidate-independent probe-construction feasibility', 'seed':974213,
      'time_ceiling_seconds':180., 'attempt_cap':10000, 'source_sha256':SOURCE,
      'script_sha256':digest(Path(__file__)), 'candidate_policy_used':False,
      'proposal':'1-4 permanent holes, multigroup 2/3-color growth with black/white count balancing; remove one occupied point',
      'scope':'prototype one-round utility/collision ranges only; no legal default-game setup, formal family coverage or learned policy evidence'}

def proposal(rng):
    adjacent=neighbors(9)
    holes=set(int(v) for v in rng.choice(81,int(rng.integers(1,5)),replace=False))
    board=[0 if p in holes else -1 for p in range(81)]
    for hole in sorted(holes):
        choices=[int(v) for v in rng.permutation(adjacent[hole]) if board[int(v)]==-1]
        colors=[1,2]
        if rng.random()<.4:colors.append(3)
        for point,color in zip(choices,colors):board[point]=color
    counts=Counter(board)
    while -1 in board:
        frontier=sorted({(p,board[q]) for p,c in enumerate(board) if c==-1 for q in adjacent[p] if board[q]>0})
        weights=[]
        for p,c in frontier:
            if c==3:weight=.2
            else:weight=float(np.exp(np.clip((counts[3-c]-counts[c])/8,-3,3)))
            weights.append(weight)
        weights=np.asarray(weights);weights/=weights.sum()
        point,color=frontier[int(rng.choice(len(frontier),p=weights))]
        board[point]=color;counts[color]+=1
    point=int(rng.choice([p for p,c in enumerate(board) if c>0]));other=board.copy();other[point]=0
    if any(not g.liberties for b in (board,other) for g in groups(b,9)):raise AssertionError('growth lost a permanent liberty')
    return State(Rules(9,1),board),State(Rules(9,1),other),point

@lru_cache(maxsize=8192)
def bounds(matrix):
    actions=tuple(range(len(matrix)))
    return collision_range(matrix,actions,actions,len(matrix)-1)

def main():
    assert not (ROOT/'plan.json').exists(), 'inspect original process/output, do not restart'
    write_report(ROOT/'plan.json',PLAN)
    write_report(ROOT/'run.json',make_manifest(seed=PLAN['seed'],command={'probe_diversity_feasibility':PLAN},rules={'size':9,'max_rounds':1,'collision':'gray'}))
    rng=np.random.default_rng(PLAN['seed']);counts=Counter();accepted=[];seen=set();started=time.monotonic();status='attempt_limited'
    for attempt in range(PLAN['attempt_cap']):
        if time.monotonic()-started>=PLAN['time_ceiling_seconds']:status='budget_limited';break
        a,b,point=proposal(rng);counts['proposals']+=1
        matrices=[_one_round_matrix(s,CertificateBudget())[0] for s in (a,b)]
        counts['holes_'+str(a.board.count(0))]+=1
        if any(len({v for row in matrix for v in row})==1 for matrix in matrices):counts['constant_payoff']+=1;continue
        try:ca,cb=(bounds(m) for m in matrices)
        except SolveBudgetExceeded:counts['uncertified']+=1;continue
        separation=max(cb.lower-ca.upper,ca.lower-cb.upper)
        if separation<.1:counts['overlap']+=1;continue
        counts['qualified_prototype_pairs']+=1
        first,second=certify_one_round(a),certify_one_round(b)
        if not (rule_sensitive(first) or rule_sensitive(second)):counts['not_gray_sensitive']+=1;continue
        identity=strategy_family_id(first,second)
        if identity in seen:counts['repeated_strategy']+=1;continue
        seen.add(identity)
        accepted.append({'attempt':attempt,'before':a.to_dict(),'after':b.to_dict(),'changed_point':point,
                         'strategy_family_id':identity,'before_range':[str(ca.lower),str(ca.upper)],
                         'after_range':[str(cb.lower),str(cb.upper)],'separation':str(separation),
                         'before_matrix':[[str(v) for v in row] for row in first.certificate.payoffs],
                         'after_matrix':[[str(v) for v in row] for row in second.certificate.payoffs],
                         'official_reachability_verified':False})
        write_report(ROOT/'prototypes.json',{'scope':PLAN['scope'],'distinct_prototype_classes':len(seen),'prototypes':accepted})
        print(json.dumps({'attempt':attempt,'distinct_prototype_classes':len(seen),'counts':dict(counts),'elapsed_seconds':time.monotonic()-started}),flush=True)
    assert source_digest()==SOURCE,'engine source changed during feasibility experiment'
    report={'status':status,'counts':dict(counts),'distinct_prototype_classes':len(seen),
            'elapsed_seconds':time.monotonic()-started,'scope':PLAN['scope'],'source_sha256':SOURCE,
            'script_sha256':digest(Path(__file__)),'formal_family_coverage_complete':False}
    write_report(ROOT/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()

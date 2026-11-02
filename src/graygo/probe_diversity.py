"""Independent multigroup proposals and exact reachable collision-probe search."""
from collections import Counter
from functools import lru_cache
import math
import time
import numpy as np
from .curriculum import canonical_board
from .confidence import canonical_sha
from .engine import Rules,State,neighbors,groups
from .probe_benchmark import ProbeBenchmark,ProbeFamily,FAMILIES,DIVERSE_GENERATOR_VERSION,MULTICELL_GENERATOR_VERSION,strategy_family_id,rule_sensitive
from .probe_certification import CertificateBudget,certify_one_round,collision_range,_one_round_matrix,SEPARATION
from .probe_setup import reachable_multigroup_pair
from .solving import SolveBudgetExceeded


def multigroup_proposal(rng):
    """Grow from permanent holes, then remove or recolor exactly one stone.

    Every seed and growth point has a same-faction path to a permanent liberty.
    Recolor proposals are retained only when all resulting groups are stable.
    No trained-policy output enters generation, filtering or orientation.
    """
    adjacent=neighbors(9)
    # Fixed stratification favors multiple liberties while retaining the dense
    # collision motifs. This schedule is part of the versioned generator.
    holes_count=int(rng.choice((1,1,2,2,2,3,3,4)))
    holes={int(v) for v in rng.choice(81,holes_count,replace=False)}
    board=[0 if p in holes else -1 for p in range(81)]
    for hole in sorted(holes):
        choices=[int(v) for v in rng.permutation(adjacent[hole]) if board[int(v)]==-1]
        colors=[1,2]
        if rng.random()<.7:colors.append(3)
        for point,color in zip(choices,colors):board[point]=color
    counts=Counter(board)
    while -1 in board:
        frontier=sorted({(p,board[q]) for p,c in enumerate(board) if c==-1 for q in adjacent[p] if board[q]>0})
        if not frontier:
            # Four permanent holes can isolate an unseeded intersection. It
            # still has a direct liberty and can safely start another group.
            seeds=[p for p,c in enumerate(board) if c==-1 and any(board[q]==0 for q in adjacent[p])]
            if not seeds:raise ValueError('unseeded component has no permanent liberty')
            point=int(rng.choice(seeds));color=int(rng.integers(1,4))
            board[point]=color;counts[color]+=1;continue
        weights=np.asarray([.5 if c==3 else float(np.exp(np.clip((counts[3-c]-counts[c])/8,-3,3))) for p,c in frontier])
        weights/=weights.sum()
        point,color=frontier[int(rng.choice(len(frontier),p=weights))]
        board[point]=color;counts[color]+=1
    point=int(rng.choice([p for p,c in enumerate(board) if c>0]));other=board.copy()
    operation='remove' if rng.random()<.5 else 'recolor'
    other[point]=0 if operation=='remove' else int(rng.choice([c for c in (1,2,3) if c!=board[point]]))
    if any(not g.liberties for b in (board,other) for g in groups(b,9)):
        return None,'unstable_recolor',operation
    return (State(Rules(9,1),board),State(Rules(9,1),other),point),None,operation


def multicell_proposal(rng):
    """Make exactly two occupied-cell removals/recolorings on a stable base.

    The first change follows the unchanged version-one proposal algorithm. A
    second distinct occupied point changes group/score interactions. Original
    holes stay empty; at most five points are empty in either root, so the
    complete six-action certificate budget remains unchanged.
    """
    candidate,reason,first=multigroup_proposal(rng)
    if candidate is None:return None,'unstable_first_change','base_'+first
    before,middle,point=candidate;other=list(middle.board)
    second_point=int(rng.choice([p for p,c in enumerate(other) if c and p!=point]))
    second='recolor' if other.count(0)>=5 else 'remove' if rng.random()<.5 else 'recolor'
    other[second_point]=0 if second=='remove' else int(rng.choice([c for c in (1,2,3) if c!=other[second_point]]))
    removals=int(first=='remove')+int(second=='remove')
    operation=('double_recolor','remove_recolor','double_remove')[removals]
    if any(not g.liberties for g in groups(other,9)):return None,'unstable_second_change',operation
    after=State(Rules(9,1),other)
    changed=[p for p,(a,b) in enumerate(zip(before.board,after.board)) if a!=b]
    if len(changed)!=2 or after.board.count(0)>5:raise ValueError('two-cell proposal violated its exact change/action budget')
    return (before,after,None),None,operation


@lru_cache(maxsize=8192)
def _normalized_range(matrix,budget):
    actions=tuple(range(len(matrix)))
    return collision_range(matrix,actions,actions,len(matrix)-1,budget=budget)


def _construct(seed,*,attempt_cap,time_cap_seconds,budget,stop_requested,on_progress,proposal_fn,generator):
    if (type(seed) is not int or seed<0 or type(attempt_cap) is not int or not 1<=attempt_cap<=1000000
            or type(time_cap_seconds) not in (int,float) or not math.isfinite(time_cap_seconds)
            or not 0<time_cap_seconds<=8*3600):
        raise ValueError('bounded candidate-independent diversity seed/attempt/time plan required')
    rng=np.random.default_rng(seed);counts=Counter();used_boards=set();used_classes=set();families=[]
    started=time.monotonic();status='running';attempts=0
    def progress():
        value={'generator':generator,'status':status,'attempts':attempts,'counts':dict(counts),
                'qualified_geometric_pairs':len(families),'distinct_strategic_families':len(used_classes),
                'elapsed_seconds':time.monotonic()-started,'candidate_policy_used':False,
                'benchmark_complete':False,'partial_families':[f.to_dict() for f in families]}
        if generator==MULTICELL_GENERATOR_VERSION:value['controlled_changes']={f.identity:list(f.changed_points) for f in families}
        return value
    for attempt in range(attempt_cap):
        if stop_requested():status='interrupted';break
        if time.monotonic()-started>=time_cap_seconds:status='budget_limited';break
        attempts=attempt+1;proposal,reason,operation=proposal_fn(rng);counts['proposed_'+operation]+=1
        if proposal is None:counts[reason]+=1;continue
        before,after,point=proposal;counts['holes_'+str(before.board.count(0))]+=1
        try:
            matrices=[]
            for state in (before,after):
                matrix,black,white=_one_round_matrix(state,budget)
                if black!=white:raise ValueError('proposal unexpectedly has asymmetric legal actions')
                matrices.append(matrix)
            # With a shared placement and pass, a constant utility game admits
            # collision probabilities zero and one, so cannot qualify a pair.
            if any(len({v for row in matrix for v in row})==1 for matrix in matrices):
                counts['constant_payoff']+=1;continue
            a_range,b_range=(_normalized_range(matrix,budget) for matrix in matrices)
            if max(b_range.lower-a_range.upper,a_range.lower-b_range.upper)<SEPARATION:
                counts['overlap']+=1;continue
            a,b=certify_one_round(before,budget=budget),certify_one_round(after,budget=budget)
        except SolveBudgetExceeded:counts['uncertified']+=1;continue
        if not (rule_sensitive(a) or rule_sensitive(b)):counts['not_gray_sensitive']+=1;continue
        strategy=strategy_family_id(a,b)
        if strategy in used_classes:counts['repeated_strategy']+=1;continue
        boards=tuple(canonical_sha({'kind':'probe-current-board/1','board':list(canonical_board(s.board))}) for s in (before,after))
        if boards[0]==boards[1] or set(boards)&used_boards:counts['root_reuse']+=1;continue
        try:roots=reachable_multigroup_pair(before.board,after.board)
        except ValueError:counts['setup_unavailable']+=1;continue
        (before,trace_a),(after,trace_b)=roots
        try:a,b=certify_one_round(before,budget=budget),certify_one_round(after,budget=budget)
        except SolveBudgetExceeded:counts['uncertified_reachable']+=1;continue
        if strategy_family_id(a,b)!=strategy:raise ValueError('actual setup changed the complete prototype game pair')
        direction=1 if b.certificate.lower-a.certificate.upper>=SEPARATION else -1
        # Alternate the orientation of accepted unique classes. Reversal never
        # manufactures another family; the class identity already quotients it.
        wanted=1 if len(families)%2==0 else -1
        if direction!=wanted:a,b,trace_a,trace_b=b,a,trace_b,trace_a
        family=ProbeFamily(a,b,point,attempt,trace_a,trace_b)
        families.append(family);used_classes.add(strategy);used_boards.update(boards)
        counts['accepted_'+operation]+=1;counts['increase' if wanted==1 else 'decrease']+=1
        if any(s.state.board.count(3) for s in (a,b)):counts['accepted_with_gray']+=1
        if any(sum(v>0 for v in distribution)>1 for s in (a,b) for distribution in s.certificate.equilibrium):counts['accepted_with_mixed_equilibrium']+=1
        on_progress(progress())
        if len(families)==FAMILIES:status='complete';break
    if status=='running':status='attempt_limited'
    benchmark=ProbeBenchmark(seed,tuple(families),generator) if len(families)==FAMILIES else None
    report=progress();report['case_catalog_complete']=benchmark is not None
    report['benchmark_complete']=benchmark is not None and benchmark.family_coverage_complete
    report['benchmark_sha256']=None if benchmark is None else benchmark.sha256
    if benchmark is not None:report['partial_families']=[]
    return benchmark,report


def construct_diverse_benchmark(seed,*,attempt_cap=100000,time_cap_seconds=900,budget=CertificateBudget(),
                                stop_requested=lambda:False,on_progress=lambda value:None):
    return _construct(seed,attempt_cap=attempt_cap,time_cap_seconds=time_cap_seconds,budget=budget,
        stop_requested=stop_requested,on_progress=on_progress,proposal_fn=multigroup_proposal,generator=DIVERSE_GENERATOR_VERSION)


def construct_multicell_benchmark(seed,*,attempt_cap=100000,time_cap_seconds=900,budget=CertificateBudget(),
                                  stop_requested=lambda:False,on_progress=lambda value:None):
    return _construct(seed,attempt_cap=attempt_cap,time_cap_seconds=time_cap_seconds,budget=budget,
        stop_requested=stop_requested,on_progress=on_progress,proposal_fn=multicell_proposal,generator=MULTICELL_GENERATOR_VERSION)

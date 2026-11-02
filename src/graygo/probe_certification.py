"""Exact near-equilibrium collision bounds for complete one-round subgames.

Floating LP output proposes an equilibrium only. Fraction arithmetic verifies
both security bounds, enumerates every polytope vertex, and checks every vertex
pair. No partial enumeration, restricted action set, or heuristic leaf is a
certificate. Longer complete-strategy games need a separate certifier.
"""
from dataclasses import dataclass,asdict
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
import math
import numpy as np
from .confidence import canonical_sha
from .engine import BLACK,WHITE,State,resolve,score,transform
from .solving import solve_matrix,SolveBudgetExceeded

CERTIFICATE_VERSION='graygo/exact-one-round-collision/1'
ROOT_VERSION='graygo/certified-one-round-probe/1'
GAP=Fraction(1,20)
SEPARATION=Fraction(1,10)
SHIFT=Fraction(1,20)


def rational(value):
    if isinstance(value,bool):raise ValueError('boolean is not a payoff/probability')
    if isinstance(value,Fraction):return value
    if isinstance(value,(int,np.integer)):return Fraction(int(value))
    if isinstance(value,(float,np.floating)):
        if not math.isfinite(value):raise ValueError('finite rational scalar required')
        return Fraction(str(float(value)))
    if isinstance(value,str):return Fraction(value)
    raise ValueError('rational scalar required')


def dot(a,b):return sum((x*y for x,y in zip(a,b)),Fraction(0))


@dataclass(frozen=True)
class CertificateBudget:
    max_actions:int=6
    max_vertex_bases:int=50000
    max_vertices:int=1024
    max_vertex_pairs:int=250000
    def __post_init__(self):
        if any(type(v) is not int or v<1 for v in asdict(self).values()):raise ValueError('positive exact-certificate budgets required')
    def dimensions(self,m,n):
        if max(m,n)>self.max_actions or max(math.comb(m+n,m-1),math.comb(m+n,n-1))>self.max_vertex_bases:
            raise SolveBudgetExceeded('complete near-equilibrium vertex budget exceeded')


def linear_solve(matrix,right):
    """Exact unique square solution, or None for a singular active basis."""
    n=len(right);a=[list(row)+[rhs] for row,rhs in zip(matrix,right)]
    for col in range(n):
        pivot=next((row for row in range(col,n) if a[row][col]),None)
        if pivot is None:return None
        a[col],a[pivot]=a[pivot],a[col]
        scale=a[col][col];a[col]=[value/scale for value in a[col]]
        for row in range(n):
            if row==col:continue
            scale=a[row][col]
            if scale:a[row]=[x-scale*y for x,y in zip(a[row],a[col])]
    return tuple(row[-1] for row in a)


def vertices(coefficients,bounds,dimension,budget):
    count=math.comb(len(bounds),dimension-1)
    if count>budget.max_vertex_bases:raise SolveBudgetExceeded('all vertex bases do not fit the declared budget')
    found=set()
    for selected in combinations(range(len(bounds)),dimension-1):
        solution=linear_solve([(Fraction(1),)*dimension]+[coefficients[i] for i in selected],
                              [Fraction(1)]+[bounds[i] for i in selected])
        if solution is not None and all(dot(row,solution)<=bound for row,bound in zip(coefficients,bounds)):
            found.add(solution)
            if len(found)>budget.max_vertices:raise SolveBudgetExceeded('complete vertex inventory exceeds budget')
    if not found:raise ValueError('near-equilibrium polytope has no verified vertex')
    return tuple(sorted(found))


def _equilibrium(matrix):
    seed=solve_matrix([[float(value) for value in row] for row in matrix])
    p=tuple(Fraction(float(value)).limit_denominator(1000000) for value in seed.row)
    q=tuple(Fraction(float(value)).limit_denominator(1000000) for value in seed.column)
    if min((*p,*q))<0 or sum(p)!=1 or sum(q)!=1:raise SolveBudgetExceeded('floating equilibrium could not be reconstructed exactly')
    lower=min(dot(p,column) for column in zip(*matrix));upper=max(dot(row,q) for row in matrix)
    if lower!=upper:raise SolveBudgetExceeded('rational primal/dual security bounds do not coincide')
    return lower,p,q


def _matrix(values):
    rows=tuple(tuple(rational(value) for value in row) for row in values)
    if not rows or not rows[0] or any(len(row)!=len(rows[0]) for row in rows):raise ValueError('nonempty rectangular exact matrix required')
    if any(abs(value)>1 for row in rows for value in row):raise ValueError('terminal utility matrix must lie in [-1,1]')
    return rows


def _actions(values,length,pass_action):
    values=tuple(values)
    if len(values)!=length or len(set(values))!=length or any(type(a) is not int or not 0<=a<=pass_action for a in values):
        raise ValueError('distinct complete legal action indices required')
    return values


@dataclass(frozen=True)
class CollisionRange:
    payoffs:tuple
    black_actions:tuple
    white_actions:tuple
    pass_action:int
    gap:Fraction
    value:Fraction
    equilibrium:tuple
    black_vertices:tuple
    white_vertices:tuple
    lower:Fraction
    upper:Fraction
    lower_witness:tuple
    upper_witness:tuple
    budget:CertificateBudget

    def to_dict(self):
        def vector(values):return [str(x) for x in values]
        def matrix(values):return [vector(row) for row in values]
        return {'version':CERTIFICATE_VERSION,'payoffs':matrix(self.payoffs),'black_actions':list(self.black_actions),
            'white_actions':list(self.white_actions),'pass_action':self.pass_action,'gap':str(self.gap),'value':str(self.value),
            'equilibrium':matrix(self.equilibrium),'black_vertices':matrix(self.black_vertices),'white_vertices':matrix(self.white_vertices),
            'lower':str(self.lower),'upper':str(self.upper),'lower_witness':matrix(self.lower_witness),'upper_witness':matrix(self.upper_witness),
            'budget':asdict(self.budget),'arithmetic':'exact rational; exhaustive vertex products'}
    @property
    def sha256(self):return canonical_sha(self.to_dict())
    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or value.get('version')!=CERTIFICATE_VERSION:raise ValueError('unknown exact collision certificate')
        try:
            matrix=_matrix(value['payoffs']);m,n=len(matrix),len(matrix[0]);budget=CertificateBudget(**value['budget'])
            budget.dimensions(m,n);passed=value['pass_action']
            if type(passed) is not int or passed<1:raise ValueError('invalid pass index')
            black=_actions(value['black_actions'],m,passed);white=_actions(value['white_actions'],n,passed)
            gap=rational(value['gap']);v=rational(value['value']);p,q=tuple(tuple(rational(x) for x in row) for row in value['equilibrium'])
            if not 0<=gap<=2 or len(p)!=m or len(q)!=n or min((*p,*q))<0 or sum(p)!=1 or sum(q)!=1:raise ValueError('invalid exact equilibrium witness')
            if min(dot(p,col) for col in zip(*matrix))!=v or max(dot(row,q) for row in matrix)!=v:raise ValueError('equilibrium witness does not certify the value')
            result=_range_from_exact(matrix,black,white,passed,gap,v,p,q,budget)
        except (KeyError,TypeError) as exc:raise ValueError('malformed collision proof schema') from exc
        if result.to_dict()!=value:raise ValueError('collision certificate does not reproduce its complete exact proof')
        return result


def collision_range(payoffs,black_actions,white_actions,pass_action,*,gap=GAP,budget=CertificateBudget()):
    matrix=_matrix(payoffs);m,n=len(matrix),len(matrix[0]);budget.dimensions(m,n)
    if type(pass_action) is not int or pass_action<1:raise ValueError('explicit positive pass index required')
    black_actions=_actions(black_actions,m,pass_action);white_actions=_actions(white_actions,n,pass_action)
    gap=rational(gap)
    if not 0<=gap<=2:raise ValueError('bounded near-equilibrium value gap required')
    value,p,q=_equilibrium(matrix)
    return _range_from_exact(matrix,black_actions,white_actions,pass_action,gap,value,p,q,budget)


@lru_cache(maxsize=2048)
def _range_from_exact(matrix,black_actions,white_actions,pass_action,gap,value,p,q,budget):
    m,n=len(matrix),len(matrix[0])
    black_coeff=[tuple(-Fraction(i==j) for j in range(m)) for i in range(m)]
    black_coeff.extend(tuple(-a for a in col) for col in zip(*matrix))
    black_bounds=[Fraction(0)]*m+[-(value-gap)]*n
    white_coeff=[tuple(-Fraction(i==j) for j in range(n)) for i in range(n)]+list(matrix)
    white_bounds=[Fraction(0)]*n+[value+gap]*m
    pv=vertices(black_coeff,black_bounds,m,budget);qv=vertices(white_coeff,white_bounds,n,budget)
    if len(pv)*len(qv)>budget.max_vertex_pairs:raise SolveBudgetExceeded('all collision vertex pairs do not fit the declared budget')
    positions=[(i,j) for i,a in enumerate(black_actions) for j,b in enumerate(white_actions) if a==b and a!=pass_action]
    lower=upper=None;lo=hi=None
    for a in pv:
        for b in qv:
            probability=sum((a[i]*b[j] for i,j in positions),Fraction(0))
            if lower is None or probability<lower:lower,lo=probability,(a,b)
            if upper is None or probability>upper:upper,hi=probability,(a,b)
    return CollisionRange(matrix,black_actions,white_actions,pass_action,gap,value,(p,q),pv,qv,lower,upper,lo,hi,budget)


@dataclass(frozen=True)
class CertifiedRoot:
    state:State
    certificate:CollisionRange
    def to_dict(self):return {'version':ROOT_VERSION,'root':self.state.to_dict(),'certificate':self.certificate.to_dict()}
    @property
    def sha256(self):return canonical_sha(self.to_dict())
    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or value.get('version')!=ROOT_VERSION:raise ValueError('unknown certified root')
        try:
            state=State.from_dict(value['root']);certificate=CollisionRange.from_dict(value['certificate'])
        except (KeyError,TypeError) as exc:raise ValueError('malformed certified root schema') from exc
        matrix,black,white=_one_round_matrix(state,certificate.budget)
        if matrix!=certificate.payoffs or black!=certificate.black_actions or white!=certificate.white_actions or state.rules.pass_action!=certificate.pass_action or certificate.gap!=GAP:
            raise ValueError('root/payoff/complete-strategy certificate differs')
        result=cls(state,certificate)
        if result.to_dict()!=value:raise ValueError('root certificate fields differ')
        return result


@lru_cache(maxsize=4096)
def _one_round_matrix(state,budget):
    if not isinstance(state,State) or state.done or state.rules.max_rounds-state.rounds_played!=1:
        raise ValueError('certifier requires a complete public root with exactly one round left')
    black=state.legal_actions(BLACK);white=state.legal_actions(WHITE);budget.dimensions(len(black),len(white))
    matrix=[]
    for b in black:
        row=[]
        for w in white:
            following=resolve(state,b,w).state
            if not following.done:raise ValueError('one-round leaf is not terminal')
            result=score(following.board,following.rules.size)
            row.append(Fraction(int(result.black_units>result.white_units)-int(result.black_units<result.white_units)))
        matrix.append(tuple(row))
    return tuple(matrix),black,white


@lru_cache(maxsize=1024)
def certify_one_round(state,*,budget=CertificateBudget()):
    matrix,black,white=_one_round_matrix(state,budget)
    return CertifiedRoot(state,collision_range(matrix,black,white,state.rules.pass_action,budget=budget))


def _distribution(values,length):
    values=tuple(rational(x) for x in values)
    if len(values)!=length or min(values,default=-1)<0 or abs(float(sum(values))-1)>1e-7:raise ValueError('complete normalized nonnegative root policy required')
    total=sum(values)
    return tuple(value/total for value in values)


def grade_root(root,black_policy,white_policy):
    certified=CertifiedRoot.from_dict(root.to_dict());c=certified.certificate
    p=_distribution(black_policy,len(c.black_actions));q=_distribution(white_policy,len(c.white_actions))
    lower=min(dot(p,col) for col in zip(*c.payoffs));upper=max(dot(row,q) for row in c.payoffs)
    collision=sum((p[i]*q[j] for i,a in enumerate(c.black_actions) for j,b in enumerate(c.white_actions) if a==b and a!=c.pass_action),Fraction(0))
    return {'black_gap':max(Fraction(0),c.value-lower),'white_gap':max(Fraction(0),upper-c.value),'collision':collision,
            'gap_passed':c.value-lower<=GAP and upper-c.value<=GAP}


def qualify_pair(before,after):
    a=CertifiedRoot.from_dict(before.to_dict());b=CertifiedRoot.from_dict(after.to_dict())
    if a.state.rules!=b.state.rules or a.state.rules.collision!='gray':raise ValueError('paired probes require the same base subgame rules')
    if a.certificate.gap!=GAP or b.certificate.gap!=GAP:raise ValueError('fixed 0.05 near-equilibrium qualification required')
    increase=b.certificate.lower-a.certificate.upper;decrease=a.certificate.lower-b.certificate.upper
    direction=1 if increase>=SEPARATION else (-1 if decrease>=SEPARATION else 0)
    return {'eligible':bool(direction),'direction':direction,'separation':max(increase,decrease),
            'before_range':(a.certificate.lower,a.certificate.upper),'after_range':(b.certificate.lower,b.certificate.upper)}


def grade_pair(before,after,before_policies,after_policies):
    qualification=qualify_pair(before,after)
    a=grade_root(before,*before_policies);b=grade_root(after,*after_policies)
    for root,graded in ((before,a),(after,b)):
        if graded['gap_passed'] and not root.certificate.lower<=graded['collision']<=root.certificate.upper:
            raise ValueError('near-equilibrium policy violates its exact certified collision range')
    change=b['collision']-a['collision'];direction=qualification['direction']
    return {'eligible':qualification['eligible'],'before':a,'after':b,'collision_change':change,'direction':direction,
            'passed':bool(qualification['eligible'] and a['gap_passed'] and b['gap_passed'] and direction*change>=SHIFT)}


@lru_cache(maxsize=8192)
def geometric_root_id(state):
    if state.rounds_played==0 and not state.history and not state.forbidden_black and not state.forbidden_white:
        from .board_augmentation import permutations
        _,inverse=permutations(state.rules.size)
        views=np.asarray(state.board,dtype=np.uint8)[inverse]
        swapped=np.array([0,2,1,3],dtype=np.uint8)[views]
        board=tuple(min(row.tobytes() for row in np.concatenate((views,swapped))))
        canonical=State(state.rules,board).canonical_json()
        return canonical_sha({'kind':'complete-root-geometric-family','state':canonical})
    from .board_augmentation import permutations
    import json
    permutation,inverse=permutations(state.rules.size);base=state.to_dict();points=state.rules.points
    boards=[np.asarray(state.board,dtype=np.uint8)[inverse]]+[np.asarray(record.before,dtype=np.uint8)[inverse] for record in state.history]
    best=None
    for swap in (False,True):
        views=[np.array([0,2,1,3],dtype=np.uint8)[board] for board in boards] if swap else boards
        for index in range(len(permutation)):
            value=dict(base);value['board']=views[0][index].tolist()
            for field,source in (('forbidden_black',state.forbidden_white if swap else state.forbidden_black),('forbidden_white',state.forbidden_black if swap else state.forbidden_white)):
                value[field]=sorted(int(permutation[index,p]) for p in source)
            history=[]
            for slot,record in enumerate(state.history):
                black,white=(record.white,record.black) if swap else (record.black,record.white)
                def move(action):return points if action==points else int(permutation[index,action])
                history.append({'before':views[slot+1][index].tolist(),'black':move(black),'white':move(white)})
            value['history']=history
            text=json.dumps(value,sort_keys=True,separators=(',',':'))
            if best is None or text<best:best=text
    return canonical_sha({'kind':'complete-root-geometric-family','state':best})

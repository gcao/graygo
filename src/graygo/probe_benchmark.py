"""Candidate-independent probe instances, exact grading and family-aware coverage."""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from dataclasses import replace
import math
import json
from itertools import permutations
import time
import numpy as np
from .confidence import canonical_sha
from .engine import Rules,State,neighbors,groups,resolve,score
from .pilot_gates import CONFIRMATION_SEEDS
from .probe_certification import CertifiedRoot,CertificateBudget,certify_one_round,qualify_pair,grade_pair,geometric_root_id,GAP,SEPARATION,SHIFT
from .solving import SolveBudgetExceeded
from .probe_setup import reachable_pair,replay_setup
from .curriculum import canonical_board

FAMILY_VERSION='graygo/paired-collision-family/1'
BENCHMARK_VERSION='graygo/collision-probe-catalog/1'
GENERATOR_VERSION='graygo/reachable-liberty-change-probes/1'
DIVERSE_GENERATOR_VERSION='graygo/reachable-multigroup-collision-probes/1'
FAMILIES=32
REQUIRED=24


def json_numbers(value):
    if isinstance(value,Fraction):return str(value)
    if isinstance(value,dict):return {key:json_numbers(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [json_numbers(item) for item in value]
    return value


@lru_cache(maxsize=2048)
def variant_payoffs(root):
    state=replace(root.state,rules=replace(root.state.rules,collision='remove'))
    return tuple(tuple(Fraction(score(resolve(state,b,w).state.board,state.rules.size).utility_black)
                       for w in root.certificate.white_actions) for b in root.certificate.black_actions)


@lru_cache(maxsize=2048)
def rule_sensitive(root):return variant_payoffs(root)!=root.certificate.payoffs


@lru_cache(maxsize=2048)
def strategy_family_id(before,after):
    # Same complete base/ablation payoff pair and collision incidence, modulo
    # common action labels/color exchange/pair reversal, is one strategic family.
    union=sorted({action for root in (before,after) for actions in (root.certificate.black_actions,root.certificate.white_actions)
                  for action in actions if action!=root.state.rules.pass_action})
    if len(union)>6:raise SolveBudgetExceeded('complete strategic-family isomorphism budget exceeded')
    best=None
    for order in permutations(range(len(union))):
        labels=dict(zip(union,order))
        for swap in (False,True):
            records=[]
            for root in (before,after):
                c=root.certificate;row_actions=c.white_actions if swap else c.black_actions;col_actions=c.black_actions if swap else c.white_actions
                def label(action):return len(union) if action==c.pass_action else labels[action]
                rows=sorted(range(len(row_actions)),key=lambda i:label(row_actions[i]));cols=sorted(range(len(col_actions)),key=lambda i:label(col_actions[i]))
                matrices=[]
                for matrix in (c.payoffs,variant_payoffs(root)):
                    matrices.append([[str(-matrix[j][i] if swap else matrix[i][j]) for j in cols] for i in rows])
                records.append({'rows':[label(row_actions[i]) for i in rows],'cols':[label(col_actions[i]) for i in cols],
                                'pass':len(union),'base_remove_payoffs':matrices})
            encoded=sorted(json.dumps(record,sort_keys=True,separators=(',',':')) for record in records)
            text=json.dumps(encoded,separators=(',',':'))
            if best is None or text<best:best=text
    return canonical_sha({'version':'graygo/complete-game-pair-isomorphism/1','pair':best})


@dataclass(frozen=True)
class ProbeFamily:
    before:CertifiedRoot
    after:CertifiedRoot
    changed_point:int
    attempt:int
    before_setup:tuple
    after_setup:tuple

    def __post_init__(self):
        object.__setattr__(self,'before_setup',tuple(tuple(pair) for pair in self.before_setup))
        object.__setattr__(self,'after_setup',tuple(tuple(pair) for pair in self.after_setup))
        q=qualify_pair(self.before,self.after)
        if not q['eligible']:raise ValueError('pair lacks the fixed complete near-equilibrium collision separation')
        a,b=self.before.state,self.after.state
        changed=tuple(p for p,(x,y) in enumerate(zip(a.board,b.board)) if x!=y)
        expected_point=changed[0] if len(changed)==1 else None
        if self.changed_point!=expected_point or isinstance(self.changed_point,bool):raise ValueError('declared single-cell change metadata differs')
        if any(s.rules!=Rules() or s.rounds_played!=199 or s.done for s in (a,b)):
            raise ValueError('default 9x9 round-199 public roots required')
        for root,trace in ((a,self.before_setup),(b,self.after_setup)):
            if len(trace)!=199 or replay_setup(trace)!=root:raise ValueError('probe root does not match its legal empty-board setup trace')
        if type(self.attempt) is not int or self.attempt<0:raise ValueError('nonnegative generator proposal index required')
        if self.root_ids[0]==self.root_ids[1]:raise ValueError('symmetry copies do not define a new family')
    @property
    def root_ids(self):return (geometric_root_id(self.before.state),geometric_root_id(self.after.state))
    @property
    def board_ids(self):return tuple(canonical_sha({'kind':'probe-current-board/1','board':list(canonical_board(root.state.board))}) for root in (self.before,self.after))
    @property
    def strategy_id(self):return strategy_family_id(self.before,self.after)
    @property
    def identity(self):return canonical_sha({'kind':FAMILY_VERSION,'roots':sorted(self.root_ids)})
    @property
    def direction(self):return qualify_pair(self.before,self.after)['direction']
    def to_dict(self):
        return {'version':FAMILY_VERSION,'family_id':self.identity,'strategy_family_id':self.strategy_id,'root_ids':list(self.root_ids),'board_ids':list(self.board_ids),'before':self.before.to_dict(),
                'after':self.after.to_dict(),'before_setup':[list(p) for p in self.before_setup],'after_setup':[list(p) for p in self.after_setup],
                'changed_point':self.changed_point,'attempt':self.attempt,'direction':self.direction,
                'qualification':json_numbers(qualify_pair(self.before,self.after)),'collision_rule_sensitive':rule_sensitive(self.before) or rule_sensitive(self.after)}
    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or value.get('version')!=FAMILY_VERSION:raise ValueError('unknown probe family')
        try:result=cls(CertifiedRoot.from_dict(value['before']),CertifiedRoot.from_dict(value['after']),value['changed_point'],value['attempt'],value['before_setup'],value['after_setup'])
        except (KeyError,TypeError) as exc:raise ValueError('malformed paired probe family') from exc
        if result.to_dict()!=value:raise ValueError('paired probe proof, direction or family identity differs')
        return result


@dataclass(frozen=True)
class ProbeBenchmark:
    seed:int
    families:tuple
    generator:str=GENERATOR_VERSION
    def __post_init__(self):
        if self.generator not in (GENERATOR_VERSION,DIVERSE_GENERATOR_VERSION):raise ValueError('unknown collision probe generator')
        object.__setattr__(self,'families',tuple(self.families))
        if type(self.seed) is not int or self.seed<0 or len(self.families)!=FAMILIES:raise ValueError('complete 32-case catalog and generation seed required')
        if any(not isinstance(f,ProbeFamily) for f in self.families):raise ValueError('certified paired families required')
        if any(f.before.state.rules!=Rules() or f.after.state.rules!=Rules() for f in self.families):raise ValueError('default 9x9/200-round rules required')
        ids=[f.identity for f in self.families];roots=[key for f in self.families for key in f.root_ids]
        if len(set(ids))!=FAMILIES or len(set(roots))!=2*FAMILIES:raise ValueError('root or family symmetry copies cannot be counted independently')
        if Counter(f.direction for f in self.families)!={-1:16,1:16}:raise ValueError('fixed 16-increase/16-decrease balance required')
        attempts=[f.attempt for f in self.families]
        if attempts!=sorted(set(attempts)):raise ValueError('original increasing construction proposal indices required')
    @property
    def strategic_family_count(self):return len({f.strategy_id for f in self.families})
    @property
    def family_coverage_complete(self):return self.strategic_family_count==FAMILIES
    def to_dict(self):
        return {'version':BENCHMARK_VERSION,'generator':self.generator,'generation_seed':self.seed,'cases':[f.to_dict() for f in self.families],
            'required_families':FAMILIES,'distinct_strategic_families':self.strategic_family_count,'family_coverage_complete':self.family_coverage_complete,
            'family_equivalence':'complete base/remove game pair up to shared placement labels, global color exchange and pair reversal',
            'required_passes_per_seed':REQUIRED,'seeds':list(CONFIRMATION_SEEDS),'individual_value_gap':str(GAP),
            'near_equilibrium_range_separation':str(SEPARATION),'required_collision_shift':str(SHIFT),
            'root_semantics':'reachable default 9x9 round-199 states; every probe episode starts at its declared root; setup traces certify reachability',
            'construction':('versioned independent public-root proposals; current generator uses one minority-stone liberty motif'
                            if self.generator==GENERATOR_VERSION else
                            'versioned independent multigroup three-faction growth; single-cell removal/recoloring; unique complete-game classes; legal default-game setup traces'),
            'inference':'descriptive finite benchmark; no binomial population confidence bound'}
    @property
    def sha256(self):return canonical_sha(self.to_dict())
    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or value.get('version')!=BENCHMARK_VERSION:raise ValueError('unknown collision benchmark')
        try:result=cls(value['generation_seed'],tuple(ProbeFamily.from_dict(f) for f in value['cases']),value['generator'])
        except (KeyError,TypeError) as exc:raise ValueError('malformed collision benchmark') from exc
        if result.to_dict()!=value:raise ValueError('benchmark requirements, proof inventory or inference scope differs')
        return result


def liberty_proposal(rng):
    rules=Rules(9,1);adjacent=neighbors(9)
    hole=int(rng.integers(81));board=[0 if p==hole else -1 for p in range(81)]
    for point,color in zip(rng.choice(adjacent[hole],2,replace=False),(1,2)):board[int(point)]=color
    while -1 in board:
        frontier=sorted({(p,board[q]) for p,c in enumerate(board) if c==-1 for q in adjacent[p] if board[q]>0})
        point,color=frontier[int(rng.integers(len(frontier)))];board[point]=color
    if board.count(1)==board.count(2):return None,'balanced'
    minority=1 if board.count(1)<board.count(2) else 2
    candidates=[p for p,c in enumerate(board) if c==minority and any(board[q]==3-minority for q in adjacent[p])]
    if not candidates:return None,'no_interface'
    point=int(rng.choice(candidates));other=board.copy();other[point]=0
    if any(not {1,2}<=set(other[q] for q in adjacent[p]) for p in (hole,point)):return None,'liberty_adjacency'
    before,after=State(rules,board),State(rules,other)
    if any(sum(g.color==color for g in groups(after.board,9))!=1 for color in (1,2)):return None,'split_group'
    if rng.random()<.5:before,after=after,before
    return (before,after,point),None


def construct_benchmark(seed,*,attempt_cap=10000,time_cap_seconds=120,budget=CertificateBudget(),stop_requested=lambda:False,on_progress=lambda value:None):
    if type(seed) is not int or seed<0 or type(attempt_cap) is not int or not 1<=attempt_cap<=1000000 or not math.isfinite(time_cap_seconds) or not 0<time_cap_seconds<=8*3600:
        raise ValueError('bounded independent probe-construction seed/attempt/time plan required')
    rng=np.random.default_rng(seed);counts=Counter();used=set();families=[];started=time.monotonic();status='running';attempts=0
    def progress():return {'status':status,'attempts':attempts,'counts':dict(counts),'qualified_geometric_pairs':len(families),
                          'elapsed_seconds':time.monotonic()-started,'candidate_policy_used':False,'benchmark_complete':False}
    for attempt in range(attempt_cap):
        if stop_requested():status='interrupted';break
        if time.monotonic()-started>=time_cap_seconds:status='budget_limited';break
        attempts=attempt+1;proposal,reason=liberty_proposal(rng)
        if proposal is None:counts[reason]+=1;continue
        before,after,point=proposal
        try:a,b=certify_one_round(before,budget=budget),certify_one_round(after,budget=budget)
        except SolveBudgetExceeded:counts['uncertified']+=1;continue
        q=qualify_pair(a,b)
        if not q['eligible']:counts['overlap']+=1;continue
        direction='increase' if q['direction']==1 else 'decrease'
        if counts[direction]>=16:counts['direction_full']+=1;continue
        boards=tuple(canonical_sha({'kind':'probe-current-board/1','board':list(canonical_board(root.board))}) for root in (before,after))
        if boards[0]==boards[1] or set(boards)&used:counts['root_reuse']+=1;continue
        try:setup=reachable_pair(before.board,after.board,point)
        except ValueError:counts['setup_unavailable']+=1;continue
        (before,trace_a),(after,trace_b)=setup
        a,b=certify_one_round(before,budget=budget),certify_one_round(after,budget=budget)
        family=ProbeFamily(a,b,point,attempt,trace_a,trace_b);families.append(family);used.update(boards);counts[direction]+=1
        on_progress(progress())
        if len(families)==FAMILIES:status='complete';break
    if status=='running':status='attempt_limited'
    benchmark=ProbeBenchmark(seed,tuple(families)) if len(families)==FAMILIES else None
    report=progress();report['case_catalog_complete']=benchmark is not None
    report['distinct_strategic_families']=0 if benchmark is None else benchmark.strategic_family_count
    report['benchmark_complete']=benchmark is not None and benchmark.family_coverage_complete
    if benchmark is not None and not benchmark.family_coverage_complete:report['status']='incomplete_family_coverage'
    report['benchmark_sha256']=None if benchmark is None else benchmark.sha256
    report['partial_families']=[f.to_dict() for f in families] if benchmark is None else []
    return benchmark,report


def assess_marginals(benchmark,by_seed):
    benchmark=ProbeBenchmark.from_dict(benchmark.to_dict())
    if set(by_seed)!=set(CONFIRMATION_SEEDS):raise ValueError('all three fixed confirmation seeds are required')
    ids={f.identity for f in benchmark.families};results=[]
    for seed in CONFIRMATION_SEEDS:
        if set(by_seed[seed])!=ids:raise ValueError('every original paired family is required for every seed')
        graded=[]
        for family in benchmark.families:
            pair=by_seed[seed][family.identity]
            if set(pair)!={'before','after'}:raise ValueError('both roots and both role marginals required')
            value=grade_pair(family.before,family.after,pair['before'],pair['after'])
            graded.append(dict(json_numbers(value),family_id=family.identity))
        passed=sum(row['passed'] for row in graded);groups={}
        for family,row in zip(benchmark.families,graded):groups.setdefault(family.strategy_id,[]).append(row)
        family_results=[{'strategy_family_id':key,'instances':len(rows),'passed':all(row['passed'] for row in rows)} for key,rows in sorted(groups.items())]
        passed_families=sum(row['passed'] for row in family_results)
        results.append({'seed':seed,'passed_instances':passed,'available_families':len(groups),'passed_families':passed_families,'required':REQUIRED,
                        'instance_threshold_passed':passed>=REQUIRED,'criterion_passed':passed_families>=REQUIRED and benchmark.family_coverage_complete,
                        'family_results':family_results,'cases':graded})
    return {'version':'graygo/collision-benchmark-grade/1','benchmark_sha256':benchmark.sha256,'seeds':results,
            'family_coverage_complete':benchmark.family_coverage_complete,'distinct_strategic_families':benchmark.strategic_family_count,
            'all_seed_instance_thresholds_passed':all(row['instance_threshold_passed'] for row in results),
            'all_seed_criterion_passed':all(row['criterion_passed'] for row in results),
            'scope':'fixed finite probe instances; complete-strategy gaps; action-isomorphic instances count as one strategic family',
            'trained_policy_provenance_verified':False,'gray_specialization_demonstrated':False}


def frozen_marginals(benchmark,policies):
    if set(policies)!=set(CONFIRMATION_SEEDS):raise ValueError('all three frozen confirmation policies required')
    output={};bindings={}
    for seed in CONFIRMATION_SEEDS:
        policy=policies[seed];policy.assert_frozen()
        if not hasattr(policy,'initial_distribution'):raise ValueError('exact deployed initial marginals required')
        bindings[seed]={'sha256':policy.sha256,'identity':policy.identity};output[seed]={}
        for family in benchmark.families:
            pair={}
            for name,root in (('before',family.before),('after',family.after)):
                state=root.state;actions=(root.certificate.black_actions,root.certificate.white_actions)
                values=[]
                for player,legal in zip((1,2),actions):
                    p=np.asarray(policy.initial_distribution(state,player,fresh_episode=True),dtype=np.float64)
                    if p.shape!=(state.rules.points+1,) or not np.isfinite(p).all() or (p<0).any() or abs(p.sum()-1)>1e-7:raise ValueError('invalid frozen public action distribution')
                    illegal=np.ones(len(p),dtype=bool);illegal[list(legal)]=False
                    if p[illegal].any():raise ValueError('probe policy assigns illegal probability')
                    values.append(p[list(legal)].tolist())
                pair[name]=values
            output[seed][family.identity]=pair
        policy.assert_frozen()
    return output,bindings

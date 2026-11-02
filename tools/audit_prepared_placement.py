"""Bounded CPU differential proof for the prepared single-placement resolver.

All 4**9 boards in both collision-rule contexts are enumerated. Resolved boards
are checked for every legal one-sided move and both-pass, complete transitions,
the placed group, and unchanged tactical private sampling. This is verification
work on hypothetical public states, never learning or game-strength evidence.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import random
import time

from graygo.agents import tactical,_tactical_reference,BASELINE_VERSION
from graygo.engine import BLACK,WHITE,Rules,State,PlacementResolver,groups,resolve_reference
from graygo.manifest import source_digest,make_manifest,write_report

REFERENCE_BODY='9163ad71918e6fb10f6ce9dc23777ad158aab6a57863f0b003daf4a73442f204'
TACTICAL_BODY='a4f9a96184c12423c8cf7e8edc739d4817948c345a044602e2d6444a7f5692a1'


def fingerprint(function,skip_doc=False):
    def normalized(value):
        if isinstance(value,ast.AST):
            return [type(value).__name__,{key:normalized(v) for key,v in ast.iter_fields(value)
                if (v is not None and v!=[]) or (isinstance(value,ast.Constant) and key=='value')}]
        if isinstance(value,list):return [normalized(x) for x in value]
        return value
    body=ast.parse(inspect.getsource(function)).body[0].body
    if skip_doc:body=body[1:]
    return hashlib.sha256(json.dumps(normalized(body),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify(*,time_cap_seconds,check_every=256):
    started=time.monotonic();source=source_digest()
    if fingerprint(resolve_reference)!=REFERENCE_BODY or fingerprint(_tactical_reference,True)!=TACTICAL_BODY:
        raise ValueError('literal frozen v1 oracle bodies changed')
    counters={'boards_enumerated':0,'resolved_public_states':0,'rejected_dead_boards':0,'complete_transitions':0,'placed_groups':0,'tactical_choices_and_rng_states':0}
    h=hashlib.sha256();status='verified';next_rule=next_board=0
    for ri,collision in enumerate(('gray','remove')):
        rules=Rules(3,3,collision)
        for code in range(4**9):
            next_rule,next_board=ri,code
            if code%check_every==0 and time.monotonic()-started>=time_cap_seconds:
                status='budget_limited';break
            board=tuple((code>>(2*i))&3 for i in range(9));counters['boards_enumerated']+=1
            try:state=State(rules,board)
            except ValueError:
                counters['rejected_dead_boards']+=1;continue
            counters['resolved_public_states']+=1
            for player in (BLACK,WHITE):
                prepared=PlacementResolver(state,player)
                for action in state.legal_actions(player):
                    black,white=(action,9) if player==BLACK else (9,action)
                    expected=resolve_reference(state,black,white);actual=prepared.resolve(action)
                    if actual.transition!=expected:raise ValueError(f'transition differs at rule={collision} board={code} player={player} action={action}')
                    own=next((g for g in groups(expected.state.board,3) if g.color==player and action in g.stones),None) if action!=9 else None
                    if own!=actual.placed_group:raise ValueError(f'placed group differs at {collision}/{code}/{player}/{action}')
                    counters['complete_transitions']+=1;counters['placed_groups']+=1
                seed=(ri<<24)+(code<<2)+player;a=random.Random(seed);b=random.Random(seed)
                x=tactical(state,player,a);y=_tactical_reference(state,player,b)
                if x!=y or a.getstate()!=b.getstate():raise ValueError(f'tactical/private RNG differs at {collision}/{code}/{player}')
                counters['tactical_choices_and_rng_states']+=1
                h.update(bytes((ri,player,x)));h.update(code.to_bytes(3,'little'))
        if status!='verified':break
    complete=status=='verified' and counters['boards_enumerated']==2*4**9
    if source_digest()!=source:raise ValueError('package source changed during verification')
    return {'version':'graygo/prepared-placement-exhaustive-audit/1','status':status,'complete_domain_checked':complete,
        'source_sha256':source,'reference_resolver_body_sha256':REFERENCE_BODY,'reference_tactical_body_sha256':TACTICAL_BODY,
        'baseline_version':BASELINE_VERSION,'counts':counters,'choice_digest_sha256':h.hexdigest(),'next_rule':next_rule,'next_board':next_board,
        'elapsed_seconds':time.monotonic()-started,'time_cap_seconds':time_cap_seconds,'new_learning_records':0,'a2_complete':False,'execution_authorized':False,
        'scope':'all resolved 3x3 boards, both rules and roles, zero-history roots; full history/restriction/terminal/large-board cases are separate tests; no trained-game or runtime-budget claim'}


def benchmark(*,seed=4301,iterations=3):
    rng=random.Random(seed);states=[]
    for _ in range(3):
        state=State()
        while not state.done:
            if state.rounds_played%8==0:states.append(state)
            from graygo.engine import resolve_reference
            state=resolve_reference(state,rng.choice(state.legal_actions(BLACK)),rng.choice(state.legal_actions(WHITE))).state
    cases=[(state,role,seed+2*i+role) for i,state in enumerate(states) for role in (BLACK,WHITE)]
    times={'prepared':[],'reference':[]};digests={}
    for repeat in range(iterations):
        order=(('prepared',tactical),('reference',_tactical_reference)) if repeat%2==0 else (('reference',_tactical_reference),('prepared',tactical))
        for name,function in order:
            started=time.perf_counter();h=hashlib.sha256()
            for state,player,private_seed in cases:
                action=function(state,player,random.Random(private_seed));h.update(action.to_bytes(2,'little'))
            times[name].append(time.perf_counter()-started);digests[name]=h.hexdigest()
    if digests['prepared']!=digests['reference']:raise ValueError('timed private choices differ')
    return {'corpus':'every eighth nonterminal state of three fixed-seed reference/uniform official games',
        'source_sha256':source_digest(),'seed':seed,'iterations':iterations,'tactical_calls_per_iteration':len(cases),'seconds':times,
        'private_choice_digest_sha256':digests['prepared'],'median_reference_over_prepared':sorted(times['reference'])[iterations//2]/sorted(times['prepared'])[iterations//2],
        'scope':'matched CPU implementation timing on this finite corpus; not full learner pricing, strength or a universal speedup'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--expected-source-sha',required=True)
    parser.add_argument('--time-cap-seconds',type=int,required=True);args=parser.parse_args()
    if not 0<args.time_cap_seconds<=1200 or source_digest()!=args.expected_source_sha:parser.error('bounded source-pinned verification required')
    root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():parser.error('verification exists; inspect original handle/result')
    import fcntl
    with (root/'.verification.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan={'kind':'exhaustive hypothetical-state verification and matched CPU timing','source_sha256':source_digest(),'time_cap_seconds':args.time_cap_seconds,
            'domain_board_count':2*4**9,'new_learning_records':0,'execution_authorized':False}
        write_report(root/'plan.json',plan);write_report(root/'run.json',make_manifest(seed=4301,command=plan,rules={'size':3,'max_rounds':3,'collision':'gray/remove'}))
        report=verify(time_cap_seconds=args.time_cap_seconds)
        if report['complete_domain_checked']:report['matched_cpu_timing']=benchmark()
        write_report(root/'report.json',report)
        print(json.dumps(report,indent=2,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())

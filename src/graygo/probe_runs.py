"""Bounded construction and read-only verification of the fixed probe benchmark."""
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import time
from .engine import BLACK,WHITE,Rules,State,resolve,score
from .manifest import make_manifest,source_digest,write_report
from .probe_benchmark import ProbeBenchmark,construct_benchmark,assess_marginals,GENERATOR_VERSION,DIVERSE_GENERATOR_VERSION,MULTICELL_GENERATOR_VERSION
from .probe_certification import CertificateBudget
from .storage import digest


def build_probes(root,*,seed,attempt_cap,time_cap_seconds,stop_requested=lambda:False,generator='liberty'):
    if generator not in ('liberty','multigroup','multicell'):raise ValueError('unknown probe construction generator')
    if generator=='multicell':
        from .probe_diversity import construct_multicell_benchmark
        constructor=construct_multicell_benchmark;version=MULTICELL_GENERATOR_VERSION
    elif generator=='multigroup':
        from .probe_diversity import construct_diverse_benchmark
        constructor=construct_diverse_benchmark;version=DIVERSE_GENERATOR_VERSION
    else:constructor=construct_benchmark;version=GENERATOR_VERSION
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'.construction.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'plan.json').exists():raise FileExistsError('construction exists; inspect its original process and outputs')
        budget=CertificateBudget()
        # Validate before publishing a nonsensical resource plan.
        import math
        if type(seed) is not int or seed<0 or type(attempt_cap) is not int or not 1<=attempt_cap<=1000000 or not math.isfinite(time_cap_seconds) or not 0<time_cap_seconds<=8*3600:
            raise ValueError('bounded probe-construction plan required')
        plan={'version':'graygo/collision-probe-construction/1','generator':version,'seed':seed,'attempt_cap':attempt_cap,
              'time_ceiling_seconds':time_cap_seconds,'certificate_budget':asdict(budget),'target_geometric_pairs':32,'required_strategic_families':32,
              'generator_scope':('one liberty-change motif; conservative complete-game family equivalence' if generator=='liberty' else
                                 'multigroup three-faction single-cell changes; one accepted pair per complete base/remove game class' if generator=='multigroup' else
                                 'multigroup three-faction two-cell changes; one accepted pair per complete base/remove game class'),
              'directions':{'increase':16,'decrease':16},
              'source_sha256':source_digest(),'rules':{'size':9,'max_rounds':200,'collision':'gray'},'candidate_policy_used':False,
              'scope':'reachable default-game round-199 benchmark construction; no learner or gray-specialization result'}
        write_report(root/'plan.json',plan)
        write_report(root/'run.json',make_manifest(seed=seed,command={'build_probes':plan},rules=plan['rules']))
        write_report(root/'process.json',{'pid':os.getpid(),'source_sha256':source_digest()})
        def progress(value):write_report(root/'progress.json',value)
        benchmark,report=constructor(seed,attempt_cap=attempt_cap,time_cap_seconds=time_cap_seconds,budget=budget,
                                              stop_requested=stop_requested,on_progress=progress)
        if benchmark is not None:write_report(root/'benchmark.json',benchmark.to_dict())
        report.update({'source_sha256':source_digest(),'plan_sha256':digest(root/'plan.json'),'training_started':False,'gray_specialization_demonstrated':False})
        write_report(root/'report.json',report);write_report(root/'progress.json',report)
        return {key:value for key,value in dict(report,benchmark_path=None if benchmark is None else str(root/'benchmark.json')).items() if key!='partial_families'}


def verify_probes(path,*,regenerate=False,time_cap_seconds=180):
    import math
    if not math.isfinite(time_cap_seconds) or not 0<time_cap_seconds<=8*3600:raise ValueError('bounded probe verification time required')
    path=Path(path);started=time.monotonic();benchmark=ProbeBenchmark.from_dict(json.loads(path.read_text()))
    def remaining():
        value=time_cap_seconds-(time.monotonic()-started)
        if value<=0:raise TimeoutError('probe verification time ceiling reached; no verified report')
        return value
    remaining()
    roots=[root for family in benchmark.families for root in (family.before,family.after)]
    sensitive_roots=0;sensitive_pairs=0;differences=0;sensitive_strategy_ids=set()
    for family in benchmark.families:
        remaining();family_sensitive=False
        for root in (family.before,family.after):
            from dataclasses import replace
            state=replace(root.state,rules=Rules(9,200,'remove'))
            count=0
            for i,black in enumerate(root.certificate.black_actions):
                for j,white in enumerate(root.certificate.white_actions):
                    following=resolve(state,black,white).state;value=score(following.board,9).utility_black
                    if value!=root.certificate.payoffs[i][j]:
                        if black!=white or black==state.rules.pass_action:raise ValueError('collision ablation changed a noncollision payoff')
                        count+=1
            sensitive_roots+=bool(count);family_sensitive|=bool(count);differences+=count
        sensitive_pairs+=family_sensitive
        if family_sensitive:sensitive_strategy_ids.add(family.strategy_id)
    marginals={seed:{family.identity:{'before':family.before.certificate.equilibrium,'after':family.after.certificate.equilibrium} for family in benchmark.families} for seed in (200,201,202)}
    oracle=assess_marginals(benchmark,marginals)
    if not oracle['all_seed_instance_thresholds_passed']:raise ValueError('exact equilibrium fixtures fail their certified directional benchmark')
    regenerated=False
    if regenerate:
        if benchmark.generator==MULTICELL_GENERATOR_VERSION:
            from .probe_diversity import construct_multicell_benchmark
            constructor=construct_multicell_benchmark
        elif benchmark.generator==DIVERSE_GENERATOR_VERSION:
            from .probe_diversity import construct_diverse_benchmark
            constructor=construct_diverse_benchmark
        else:constructor=construct_benchmark
        rebuilt,report=constructor(benchmark.seed,attempt_cap=benchmark.families[-1].attempt+1,time_cap_seconds=remaining())
        if rebuilt is None or rebuilt.to_dict()!=benchmark.to_dict():raise ValueError('benchmark does not reproduce its independent construction seed')
        regenerated=True
    remaining()
    return {'format_version':1,'verified':True,'status':'complete' if benchmark.family_coverage_complete else 'incomplete_family_coverage','benchmark_sha256':benchmark.sha256,'benchmark_file_sha256':digest(path),'source_sha256':source_digest(),
        'geometric_pairs':len(benchmark.families),'distinct_strategic_families':benchmark.strategic_family_count,
        'family_coverage_complete':benchmark.family_coverage_complete,'distinct_geometric_roots':len({key for f in benchmark.families for key in f.root_ids}),'distinct_current_boards':len({key for f in benchmark.families for key in f.board_ids}),
        'setup_rounds_replayed':sum(len(trace) for f in benchmark.families for trace in (f.before_setup,f.after_setup)),
        'directions':{'increase':sum(f.direction==1 for f in benchmark.families),'decrease':sum(f.direction==-1 for f in benchmark.families)},
        'minimum_range_separation':str(min(max(f.after.certificate.lower-f.before.certificate.upper,f.before.certificate.lower-f.after.certificate.upper) for f in benchmark.families)),
        'minimum_value':str(min(r.certificate.value for r in roots)),'maximum_value':str(max(r.certificate.value for r in roots)),
        'collision_rule_sensitive_roots':sensitive_roots,'collision_rule_sensitive_pairs':sensitive_pairs,'collision_rule_sensitive_strategic_families':len(sensitive_strategy_ids),'collision_rule_sensitive_payoff_cells':differences,
        'exact_equilibrium_fixture_instance_pass_counts':{str(r['seed']):r['passed_instances'] for r in oracle['seeds']},
        'generation_reproduced':regenerated,'elapsed_seconds':time.monotonic()-started,'trained_policy_evaluated':False,'gray_specialization_demonstrated':False,
        'scope':'exact complete one-round proof, source-seed reproducibility and solver fixtures; trained policy, exposure-controlled performance and interaction evidence remain separate'}

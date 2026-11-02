"""Independent exact proof and original-prefix audit, including partial catalogs."""
from collections import Counter
from dataclasses import asdict
import argparse
import json
import math
import os
from pathlib import Path
import signal
import time
from unittest.mock import patch
from graygo.manifest import make_manifest,source_digest,write_report
from graygo.probe_benchmark import ProbeBenchmark,ProbeFamily,DIVERSE_GENERATOR_VERSION,rule_sensitive
from graygo.probe_certification import CertificateBudget,grade_pair
from graygo.probe_diversity import construct_diverse_benchmark
from graygo.storage import digest


def require(condition,message):
    if not condition:raise ValueError(message)


def inventory(root):
    return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def audit(root,allocation_path,output,time_cap_seconds,regenerate):
    root=Path(root).resolve();output=Path(output).resolve();allocation_path=Path(allocation_path).resolve()
    require(not output.exists() and not output.is_relative_to(root),'one fresh separate audit directory required; inspect the original audit process before retrying')
    require(type(time_cap_seconds) in (int,float) and math.isfinite(time_cap_seconds) and time_cap_seconds>0,'finite audit ceiling required')
    allocation=json.loads(allocation_path.read_text());plan=json.loads((root/'plan.json').read_text())
    manifest=json.loads((root/'run.json').read_text());process=json.loads((root/'process.json').read_text())
    pid_path=Path('/proc')/str(process['pid'])/'cmdline'
    if pid_path.exists():
        command=pid_path.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        require(str(root) not in command,'original construction is still live; re-poll it without restarting')
    require((root/'report.json').is_file(),'construction has no terminal report; inspect its exact original process')
    report=json.loads((root/'report.json').read_text());source=source_digest()
    require(source==plan['source_sha256']==manifest['source_sha256']==process['source_sha256']==report['source_sha256']==allocation['source_sha256'],'use the original frozen package source')
    require(manifest['git_dirty'] is False and manifest['git_commit'].startswith(allocation['source_commit']),'source was not the declared clean frozen commit')
    require(plan['generator']==allocation['generator']==DIVERSE_GENERATOR_VERSION and plan['seed']==allocation['seed']
            and plan['attempt_cap']==allocation['attempt_cap'] and plan['time_ceiling_seconds']==allocation['construction_time_ceiling_seconds'],'construction allocation/plan differs')
    require(time_cap_seconds<=allocation['verification_time_ceiling_seconds'],'verification exceeds its selected allocation')
    require(report['plan_sha256']==digest(root/'plan.json') and report['training_started'] is False
            and report['gray_specialization_demonstrated'] is False,'terminal provenance or scope differs')
    require(report['status'] in ('complete','budget_limited','attempt_limited','interrupted'),'nonterminal construction status')
    require(type(report['attempts']) is int and 0<=report['attempts']<=plan['attempt_cap'],'invalid proposal prefix')
    before=inventory(root);stopped=[False]
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,lambda *args:stopped.__setitem__(0,True))
    output.mkdir(parents=True)
    audit_plan={'purpose':'independent full-proof and original-prefix audit','source_sha256':source,
                'input':str(root),'allocation_sha256':digest(allocation_path),'input_files':before,
                'auditor_sha256':digest(Path(__file__)),'time_ceiling_seconds':time_cap_seconds,'regenerate':bool(regenerate)}
    write_report(output/'plan.json',audit_plan)
    write_report(output/'run.json',make_manifest(seed=plan['seed'],command={'audit_probe_construction':audit_plan},rules=plan['rules']))
    write_report(output/'process.json',{'pid':os.getpid(),'source_sha256':source})
    started=time.monotonic()
    def remaining():
        if stopped[0]:raise InterruptedError('audit interrupted; no verified result')
        value=time_cap_seconds-(time.monotonic()-started)
        if value<=0:raise TimeoutError('audit allocation exhausted; no verified result')
        return value
    benchmark=None
    with patch('graygo.probe_certification.solve_matrix',side_effect=AssertionError('exact proof reading must not invoke a floating LP')):
        if (root/'benchmark.json').exists():
            benchmark=ProbeBenchmark.from_dict(json.loads((root/'benchmark.json').read_text()));families=benchmark.families
        else:families=tuple(ProbeFamily.from_dict(row) for row in report['partial_families'])
        ids=set();roots=set();boards=set();counts=Counter();teacher=0;proofs=[]
        for index,family in enumerate(families):
            remaining()
            require(family.strategy_id not in ids,'repeated complete-game class cannot count as another family');ids.add(family.strategy_id)
            require(not (set(family.root_ids)&roots) and not (set(family.board_ids)&boards),'reused geometric root/current board')
            roots.update(family.root_ids);boards.update(family.board_ids)
            require(family.direction==(1 if index%2==0 else -1),'fixed alternating direction schedule differs')
            require(rule_sensitive(family.before) or rule_sensitive(family.after),'pair has no Gray-rule utility effect')
            require(all(asdict(r.certificate.budget)==plan['certificate_budget'] for r in (family.before,family.after)),'certificate budget differs')
            grade=grade_pair(family.before,family.after,family.before.certificate.equilibrium,family.after.certificate.equilibrium)
            require(grade['passed'],'exact equilibrium fixture fails its complete paired criterion');teacher+=1
            a,b=family.before.state,family.after.state
            changed=[p for p,(x,y) in enumerate(zip(a.board,b.board)) if x!=y]
            require(changed==[family.changed_point],'proposal is not its declared single-cell change')
            operation='remove' if 0 in (a.board[changed[0]],b.board[changed[0]]) else 'recolor'
            counts['accepted_'+operation]+=1;counts['increase' if family.direction==1 else 'decrease']+=1
            if any(r.state.board.count(3) for r in (family.before,family.after)):counts['accepted_with_gray']+=1
            if any(sum(v>0 for v in p)>1 for r in (family.before,family.after) for p in r.certificate.equilibrium):counts['accepted_with_mixed_equilibrium']+=1
            proofs.append({'pair_id':family.identity,'strategy_family_id':family.strategy_id,'attempt':family.attempt,
                           'root_ids':list(family.root_ids),'board_ids':list(family.board_ids),'direction':family.direction})
    proof_seconds=time.monotonic()-started
    require(len(ids)==report['distinct_strategic_families']==report['qualified_geometric_pairs'],'reported unique counts differ')
    require([f.attempt for f in families]==sorted({f.attempt for f in families}) and all(f.attempt<report['attempts'] for f in families),'accepted proposal indices differ')
    for key in ('accepted_remove','accepted_recolor','increase','decrease','accepted_with_gray','accepted_with_mixed_equilibrium'):
        require(counts[key]==report['counts'].get(key,0),'reported accepted-case summary differs: '+key)
    complete=len(families)==32
    require(report['benchmark_complete']==report['case_catalog_complete']==complete,'partial/full completion flag differs')
    require((benchmark is not None)==complete,'benchmark inventory differs from coverage')
    if benchmark is not None:
        require(benchmark.seed==plan['seed'] and benchmark.generator==DIVERSE_GENERATOR_VERSION
                and benchmark.family_coverage_complete and benchmark.sha256==report['benchmark_sha256'],'full catalog identity differs')
    generation_seconds=0.
    if regenerate:
        def progress(value):
            write_report(output/'progress.json',{'status':'regenerating','attempts':value['attempts'],
                'distinct_strategic_families':value['distinct_strategic_families'],'elapsed_seconds':time.monotonic()-started})
        at=time.monotonic()
        if report['attempts']:
            rebuilt,reproduction=construct_diverse_benchmark(plan['seed'],attempt_cap=report['attempts'],time_cap_seconds=remaining(),
                budget=CertificateBudget(**plan['certificate_budget']),stop_requested=lambda:stopped[0],on_progress=progress)
            remaining()
            require(reproduction['attempts']==report['attempts'] and reproduction['counts']==report['counts'],'original proposal prefix/counts do not reproduce')
            actual=list(rebuilt.families) if rebuilt is not None else [ProbeFamily.from_dict(row) for row in reproduction['partial_families']]
            require([f.to_dict() for f in actual]==[f.to_dict() for f in families],'original exact families/setup traces do not reproduce')
            require((rebuilt is not None)==complete,'reproduced completion flag differs')
        else:require(not families,'zero proposal prefix contains accepted families')
        generation_seconds=time.monotonic()-at
    remaining()
    require(inventory(root)==before,'audit modified or raced its original evidence')
    result={'format_version':1,'verified':True,'status':'complete' if complete else 'incomplete_family_coverage',
        'purpose':'independent candidate-free construction audit','source_sha256':source,'auditor_sha256':audit_plan['auditor_sha256'],
        'input_report_sha256':digest(root/'report.json'),'input_plan_sha256':digest(root/'plan.json'),'allocation_sha256':digest(allocation_path),
        'distinct_strategic_families':len(ids),'qualified_pairs':len(families),'distinct_roots':len(roots),'distinct_current_boards':len(boards),
        'family_coverage_complete':complete,'required_families':32,'generation_reproduced':bool(regenerate),
        'proposal_prefix_reproduced':report['attempts'] if regenerate else None,'accepted_case_counts':dict(counts),
        'setup_rounds_replayed':sum(len(f.before_setup)+len(f.after_setup) for f in families),'exact_equilibrium_fixture_passed_pairs':teacher,
        'proofs':proofs,'proof_reading_seconds':proof_seconds,'regeneration_seconds':generation_seconds,
        'elapsed_seconds':time.monotonic()-started,'construction_elapsed_seconds':report['elapsed_seconds'],
        'construction_atomic_boundary_overrun_seconds':max(0.,report['elapsed_seconds']-plan['time_ceiling_seconds']),
        'original_evidence_unchanged':True,'trained_policy_evaluated':False,'gray_specialization_demonstrated':False,
        'scope':'exact complete-root/setup/class/near-equilibrium proofs and source-seed prefix; partial coverage stays incomplete; teacher fixtures are not trained-policy evidence'}
    write_report(output/'report.json',result);write_report(output/'progress.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_dir');parser.add_argument('--allocation',required=True)
    parser.add_argument('--output-dir',required=True);parser.add_argument('--time-cap-hours',type=float,required=True)
    parser.add_argument('--regenerate',action='store_true');args=parser.parse_args()
    result=audit(args.run_dir,args.allocation,args.output_dir,args.time_cap_hours*3600,args.regenerate)
    print(json.dumps({k:v for k,v in result.items() if k!='proofs'},sort_keys=True))

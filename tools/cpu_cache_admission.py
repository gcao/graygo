"""Qualify CPU 3x3 raw-forward reuse without changing original case identities."""
import argparse
from contextlib import ExitStack
import fcntl
from importlib import metadata
import json
from pathlib import Path
import time
from graygo.task_training import NFSPTaskConfig
from graygo.manifest import write_report
from tools.cpu_cache_structure import prove
from tools.cpu_cache_pipeline import compare_profiles
from tools.nfsp_resources import require
from tools.task_actor_boundary import file_sha

VERSION='graygo/cpu-3x3-raw-cache-admission/1'


def load(path):return json.loads(Path(path).read_text())

def files(root):
    root=Path(root).resolve();return {str(p.relative_to(root)):p.read_bytes() for p in [*root.glob('*.py'),*(root/'native').glob('*.cpp')]}

def package_sha(root):
    import hashlib
    h=hashlib.sha256()
    for name,value in sorted(files(root).items()):h.update(name.encode()+b'\0'+value+b'\0')
    return h.hexdigest()


def check_case(manifest,proof):
    require(proof.get('version')==VERSION and proof.get('status')=='qualified_cpu_3x3_raw_cache' and proof.get('source_compatibility_qualified') is True and proof.get('execution_authorized') is False,'qualified scoped raw-cache receipt required')
    require(manifest['source_sha256']==proof['candidate_source_sha256'],'raw-cache case is not its original candidate source')
    config=NFSPTaskConfig(**manifest['config']);require(config.task in ('gray-pure','gray-mixed') and config.learner in ('neural','tabular') and type(config.seed) is int and config.seed in (0,1,2),'case outside fixed bounded family/learner/seed scope')
    require(config.formal_protocol() and config.q_reuse==4 and config.device=='cpu' and config.storage=='features' and config.native_library is None,'case differs from primary CPU/features recipe')
    runtime=manifest['runtime'];require(runtime['device']=='cpu' and type(runtime['torch_threads']) is int and runtime['torch_threads']==1 and type(runtime['workers']) is int and runtime['workers']==1,'case CPU runtime differs')
    require(manifest['dependencies']==proof['dependencies'],'raw-cache numerical dependencies differ')
    return {'source_admission_applies':True,'original_source_identity_preserved':True,'primary_original_plan_and_independent_case_audit_required':True,'a2_complete':False,'execution_authorized':False}


def qualify(repo,reference,candidate,comparison_root,audit_root,suite_root,*,expected_reference,expected_candidate):
    started=time.monotonic();repo=Path(repo).resolve();reference=Path(reference).resolve();candidate=Path(candidate).resolve();comparison_root=Path(comparison_root).resolve();audit_root=Path(audit_root).resolve();suite_root=Path(suite_root).resolve()
    require(package_sha(reference)==expected_reference and package_sha(candidate)==expected_candidate,'frozen raw-cache source package changed')
    structure=prove(files(reference),files(candidate));job=load(comparison_root/'job.json');audits=load(audit_root/'job.json');suite=load(suite_root/'job.json')
    require(job['status']=='four_actual_cache_pipelines_exactly_equal' and job['reference_source_sha256']==expected_reference and job['candidate_source_sha256']==expected_candidate,'original actual comparison incomplete or changed source')
    allocation=Path(job['comparison_command'][job['comparison_command'].index('--allocation')+1]);plan=load(allocation);require(file_sha(allocation)==job['allocation_sha256'],'original comparison allocation changed')
    require(plan['reference_source_sha256']==expected_reference and plan['candidate_source_sha256']==expected_candidate,'comparison declared other source packages')
    require(job['measured_driver_wall_seconds']<=plan['outer_watchdog_seconds'],'original comparison exceeded its declared whole-job cap')
    require(audits['status']=='verified_all_four_actual_cache_prefixes' and audits['original_inputs_unchanged'] is True and audits['source_sha256']==expected_candidate and len(audits['cases'])==4 and all(c['verified'] for c in audits['cases'].values()),'four unchanged independent candidate audits required')
    require(suite['status']=='complete_suite_passed' and suite['source_sha256']==expected_candidate and suite['returncode']==0 and suite['reported_tests']>=533,'complete passing candidate suite required')
    require(file_sha(suite_root/'suite.stdout')==suite['suite_stdout_sha256'] and file_sha(suite_root/'suite.stderr')==suite['suite_stderr_sha256'],'original complete-suite output changed')
    inputs={str(allocation):file_sha(allocation),str(comparison_root/'job.json'):file_sha(comparison_root/'job.json'),str(audit_root/'job.json'):file_sha(audit_root/'job.json'),str(suite_root/'job.json'):file_sha(suite_root/'job.json'),str(suite_root/'suite.stdout'):file_sha(suite_root/'suite.stdout'),str(suite_root/'suite.stderr'):file_sha(suite_root/'suite.stderr')}
    for helper in ('cpu_cache_structure.py','terminal_batch_admission.py','cpu_cache_pipeline.py','task_pipeline_parity.py'):
        path=Path(__file__).with_name(helper);inputs[str(path.resolve())]=file_sha(path)
    supervisor=comparison_root/'outer-cap-supervisor.json'
    if supervisor.exists():
        budget=load(supervisor);require(budget['status']=='original_parent_terminal_under_declared_supervision' and budget['no_training_restart'] and budget['original_parent']['outer_cap_seconds']==plan['outer_watchdog_seconds'] and budget['observed_elapsed_from_parent_start_seconds']<=plan['outer_watchdog_seconds'],'declared outer budget supervision did not finish within its cap');inputs[str(supervisor)]=file_sha(supervisor);inputs[str(comparison_root/'outer-cap-supervisor.py')]=file_sha(comparison_root/'outer-cap-supervisor.py')
    dependencies=None;verified={}
    with ExitStack() as stack:
        roots=[]
        for case in plan['cases']:
            task=case['config']['task'];phase=case['phase'];roots.extend([(repo/case['reference_root']).resolve(),comparison_root/'profiles'/task/('neural-R4-'+phase)])
        for root in sorted(roots):fcntl.flock(stack.enter_context((root/'.trainer.lock').open('rb')),fcntl.LOCK_SH|fcntl.LOCK_NB)
        fresh=compare_profiles(repo/plan['reference_root'],comparison_root/'profiles',plan)
        stored=load(comparison_root/'comparison.json');require(fresh==stored and file_sha(comparison_root/'comparison.json')==job['comparison_sha256'],'fresh full pipeline partition equality differs')
        inputs[str(comparison_root/'comparison.json')]=file_sha(comparison_root/'comparison.json')
        require(all(file_sha(p)==h for p,h in audits['original_inputs_sha256'].items()),'independently audited data changed');inputs.update(audits['original_inputs_sha256'])
        audit_allocation=Path(audits['commands'][0]['command'][1]).parent
        for case in plan['cases']:
            task=case['config']['task'];phase=case['phase'];key=task+'/'+phase;root=comparison_root/'profiles'/task/('neural-R4-'+phase);manifest=load(root/'run.json');output=audit_root/(key.replace('/','-')+'.json');report=load(output)
            require(file_sha(output)==audits['cases'][key]['output_sha256'] and report['verified'] and report['source_sha256']==expected_candidate and report['training_manifest_sha256']==file_sha(root/'run.json') and report['result_sha256']==file_sha(root/'result.json'),'independent actual cache case binding differs')
            require(report['records']==32768 and not report['formal_protocol'],'cache diagnostic was rebranded as formal evidence')
            require(report['auditor_sha256']==audits['auditor_sha256']==file_sha(audit_allocation/'audit_nfsp_bounded.py'),'pinned independent bounded auditor changed')
            if dependencies is None:dependencies=manifest['dependencies']
            require(manifest['dependencies']==dependencies,'candidate numerical environments differ')
            inputs[str(output)]=file_sha(output);verified[key]={'audit_sha256':file_sha(output),'original_root':str(root),'original_manifest_sha256':file_sha(root/'run.json')}
        require(all(version is None or metadata.version(name)==version for name,version in dependencies.items()),'live numerical dependencies changed')
        require(all(file_sha(p)==h for p,h in inputs.items()),'cache qualification inputs changed')
    require(package_sha(reference)==expected_reference and package_sha(candidate)==expected_candidate,'source changed during qualification')
    return {'version':VERSION,'status':'qualified_cpu_3x3_raw_cache','reference_source_sha256':expected_reference,'candidate_source_sha256':expected_candidate,'families':['gray-pure','gray-mixed'],'learners':['neural','tabular'],'device':'cpu','storage':'features','scalar_control_path_literal_unchanged':True,'source_compatibility_qualified':True,'dependencies':dependencies,'verifier_sha256':file_sha(Path(__file__)),'structural_proof':structure,'actual_pipeline_sha256':file_sha(comparison_root/'comparison.json'),'independent_actual_prefix_audits':verified,'complete_suite_tests':suite['reported_tests'],'complete_suite_job_sha256':file_sha(suite_root/'job.json'),'input_files_sha256':inputs,'full_15M_equivalence_replay_claimed':False,'cross_source_resume_allowed':False,'original_case_identity_rewriting_allowed':False,'a2_complete':False,'execution_authorized':False,'qualification_seconds':time.monotonic()-started,'evidence':{'repo':str(repo),'reference':str(reference),'candidate':str(candidate),'comparison_root':str(comparison_root),'audit_root':str(audit_root),'suite_root':str(suite_root)},'scope':'CPU/features bounded 3x3 task source compatibility only. Known frozen neural raw forwards are exact; table paths remain literal and inactive for this cache. All masks/probabilities/epsilon/labels/RNG/learning/serialization code is preserved, full actual pipelines agree and unchanged independent audits pass. Each original source-specific plan/case/audit and separate current runtime price is still required; no admission, funding or strength follows for CUDA/native, other tasks or the full 9x9 runtime.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',required=True);parser.add_argument('--reference',required=True);parser.add_argument('--candidate',required=True);parser.add_argument('--comparison',required=True);parser.add_argument('--audits',required=True);parser.add_argument('--suite',required=True);parser.add_argument('--expected-reference',required=True);parser.add_argument('--expected-candidate',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();require(not Path(args.output).exists(),'preserve original cache qualification output')
    result=qualify(args.repo,args.reference,args.candidate,args.comparison,args.audits,args.suite,expected_reference=args.expected_reference,expected_candidate=args.expected_candidate);write_report(args.output,result);print(json.dumps({'status':result['status'],'complete_suite_tests':result['complete_suite_tests']}))


if __name__=='__main__':main()

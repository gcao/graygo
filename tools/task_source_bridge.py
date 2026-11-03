"""Qualify a CPU/features A2 source bridge from the complete recorded checks.

Original cases retain their original source and cohort plan. This certificate
only connects that task implementation to the selected later package; it never
resumes a trainer, fills missing cohorts, prices work or authorizes collection.
"""
import argparse
import ast
from copy import deepcopy
from contextlib import ExitStack
import fcntl
from importlib import metadata
import json
from pathlib import Path
import time
from tools.task_actor_boundary import file_sha, require, verify_pair as verify_actors
from tools.task_default_specialization import prove, ast_sha
from tools.task_domain_boundary import verify_pair as verify_domains
from tools.task_pipeline_parity import compare_profiles
from tools.task_source_diff import compare, inventory

VERSION='graygo/cpu-features-A2-source-bridge/1'
FAMILIES=('asymmetric','rps','delayed','branching','gray-pure','gray-mixed')


def load(path):return json.loads(Path(path).read_text())


def reference_body(source,name):
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    fn=deepcopy(fn);fn.name='reference'
    if fn.body and isinstance(fn.body[0],ast.Expr) and isinstance(fn.body[0].value,ast.Constant) and isinstance(fn.body[0].value.value,str):fn.body.pop(0)
    return ast_sha(fn)


def check_case(manifest,bridge,*,selected_candidate_source):
    """Source/config eligibility only; the original independent audit is separate."""
    from graygo.nfsp_training import NFSPMatrixConfig
    from graygo.task_training import NFSPTaskConfig
    require(bridge['version']==VERSION and bridge['status']=='qualified_cpu_features_A2_source_bridge'
            and bridge['source_compatibility_qualified'] is True,'qualified scoped bridge required')
    require(bridge['candidate_source_sha256']==selected_candidate_source
            and manifest['source_sha256']==bridge['reference_source_sha256'],'original/selected source binding differs')
    config=manifest['config'];task=config['task'];matrix=task in ('asymmetric','rps')
    require(task in FAMILIES and type(config['seed']) is int and config['seed'] in (0,1,2)
            and config['learner'] in ('neural','tabular') and config['device']=='cpu','case outside fixed CPU A2 family/learner/seed scope')
    parsed=NFSPMatrixConfig(**config) if matrix else NFSPTaskConfig(**config)
    require(parsed.formal_protocol() and parsed.q_reuse==(1 if matrix else 4),'case protocol differs from the selected primary recipe')
    if not matrix:
        require(parsed.storage=='features' and parsed.native_library is None,'native/rule-state tasks are outside this bridge')
        runtime=manifest.get('runtime',{})
        require(runtime.get('device')=='cpu' and type(runtime.get('torch_threads')) is int and runtime['torch_threads']==1,'task runtime differs')
    require(manifest['dependencies']==bridge['dependencies'],'case numerical dependencies differ')
    return {'source_bridge_applies':True,'original_source_identity_preserved':True,'original_case_plan_and_audit_required':True,
            'a2_complete':False,'execution_authorized':False}


def _qualify_locked(repo,domain_run,allocation,*,expected_allocation_sha):
    started=time.monotonic();repo=Path(repo).resolve();domain_run=Path(domain_run).resolve();allocation=Path(allocation).resolve()
    require(file_sha(allocation)==expected_allocation_sha,'pinned domain allocation differs')
    plan=load(allocation);reference=repo/plan['reference_package_path']/'graygo';candidate=repo/plan['candidate_package_path']/'graygo'
    old,new=plan['reference_source_sha256'],plan['candidate_source_sha256'];domain_job=load(domain_run/'job.json')
    require(domain_job['status']=='verified_domain_and_default_specialization' and domain_job['source_packages_unchanged'] is True
            and domain_job['allocation_sha256']==expected_allocation_sha,'original complete domain job is not verified')
    require(domain_job['new_training_records']==domain_job['new_scored_evaluation_games']==0 and not domain_job['new_accelerator_job'],'domain work changed scope')
    require([row['label'] for row in domain_job['commands']]==['specialization','reference','candidate'] and all(row['returncode']==0 for row in domain_job['commands']),'original domain phases incomplete')
    frozen=repo/'.worktrees/task-source-bridge-446a058'
    require(file_sha(frozen/plan['domain_producer'])==domain_job['domain_producer_sha256']==plan['domain_producer_sha256']
            and file_sha(frozen/plan['specialization_producer'])==domain_job['specialization_producer_sha256']==plan['specialization_producer_sha256'],'domain/specialization producer binding differs')
    inputs={}
    def sealed(path):
        path=Path(path);inputs[str(path)]=file_sha(path);return load(path)
    sealed(allocation);sealed(domain_run/'job.json')
    for name in ('tools/task_domain_boundary.py','tools/task_default_specialization.py','tools/task_source_diff.py','tools/task_actor_boundary.py','tools/task_pipeline_parity.py'):
        inputs[str(repo/name)]=file_sha(repo/name)
    require(inputs[str(repo/'tools/task_domain_boundary.py')]==plan['domain_producer_sha256']
            and inputs[str(repo/'tools/task_default_specialization.py')]==plan['specialization_producer_sha256']
            and inputs[str(repo/'tools/task_source_diff.py')]==plan['source_inventory_helper_sha256']
            and inputs[str(repo/'tools/task_actor_boundary.py')]==plan['transcript_helper_sha256'],'fresh verifier helpers differ from their pinned domain protocol')
    differences=compare(reference,candidate,expected_reference=old,expected_candidate=new)
    require(set(differences['changed_dependency_modules'])=={'engine','agents','nfsp_collection','frozen_forward'},'unreviewed task dependency changed')
    _,old_files,_=inventory(reference);_,new_files,_=inventory(candidate)
    # The changed tactical implementation is imported only for its literal version.
    agent_imports=[]
    for module in differences['candidate_dependencies']['modules']:
        if module=='agents':continue
        for node in ast.walk(ast.parse(new_files[module+'.py'])):
            if isinstance(node,ast.ImportFrom) and node.module in ('agents','graygo.agents'):
                require([(a.name,a.asname) for a in node.names]==[('BASELINE_VERSION',None)],'task dependency imports a changed tactical operation')
                agent_imports.append(module)
    require(agent_imports==['manifest'],'baseline import boundary differs')
    def baseline(source):
        values=[n.value for n in ast.parse(source).body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='BASELINE_VERSION']
        require(len(values)==1 and isinstance(values[0],ast.Constant) and isinstance(values[0].value,str),'literal baseline version required');return values[0].value
    require(baseline(old_files['agents.py'])==baseline(new_files['agents.py']),'baseline metadata version changed')
    resolver=reference_body(old_files['engine.py'],'resolve');require(resolver==reference_body(new_files['engine.py'],'resolve_reference'),'literal original rule resolver was changed')
    specialized=prove(reference,candidate,expected_reference=old,expected_candidate=new)
    stored=sealed(domain_run/'specialization.json');require(specialized==stored and file_sha(domain_run/'specialization.json')==domain_job['specialization_sha256'],'fresh default specialization differs')
    domains=verify_domains(sealed(domain_run/'reference.json'),sealed(domain_run/'candidate.json'),reference_source=old,candidate_source=new)
    require(domains==sealed(domain_run/'comparison.json') and file_sha(domain_run/'comparison.json')==domain_job['comparison_sha256'],'complete domain comparison differs')
    actor=repo/'runs/task-actor-boundary-53aa73c';actor_job=sealed(actor/'job.json');actor_plan=sealed(repo/'docs/experiments/task-actor-boundary-allocation.json')
    require(actor_job['status']=='verified' and actor_job['original_inputs_unchanged'] is True
            and actor_job['allocation_sha256']==file_sha(repo/'docs/experiments/task-actor-boundary-allocation.json')
            and actor_job['tool_sha256']==inputs[str(repo/'tools/task_actor_boundary.py')],'trained actor job binding differs')
    actor_results={}
    for learner,archive in actor_plan['trained_archives'].items():
        actor_results[learner]=verify_actors(sealed(actor/(learner+'-reference.json')),sealed(actor/(learner+'-candidate.json')),
            reference_source=old,candidate_source=new,archive_manifest_sha=archive['manifest_sha256'])
        bundle=repo/archive['path'];require(file_sha(bundle/'manifest.json')==archive['manifest_sha256'] and file_sha(bundle/'models.pt')==archive['models_sha256'],'original trained actor archive changed')
        inputs[str(bundle/'manifest.json')]=file_sha(bundle/'manifest.json');inputs[str(bundle/'models.pt')]=file_sha(bundle/'models.pt')
    require(actor_results==sealed(actor/'comparison.json')['comparisons'] and file_sha(actor/'comparison.json')==actor_job['comparison_sha256'],'fresh trained actor comparison differs')
    actor_inventory=sealed(actor/'original-input-snapshot.json')
    current_snapshot={'files':{},'directories':[]}
    for archive in actor_plan['trained_archives'].values():
        original=(repo/archive['path']).parent.parent
        for path in sorted(original.rglob('*')):
            require(not path.is_symlink(),'original learning inputs contain a symlink')
            name=str(path.relative_to(repo))
            if path.is_file():current_snapshot['files'][name]=file_sha(path)
            elif path.is_dir():current_snapshot['directories'].append(name)
    require(current_snapshot==actor_inventory,'original learning input inventory changed')
    for name,digest in actor_inventory['files'].items():inputs[str(repo/name)]=digest
    pipeline=repo/'runs/task-pipeline-parity-d724e06';corrected=repo/'runs/task-pipeline-corrected-71c404e';pipeline_job=sealed(pipeline/'job.json');corrected_job=sealed(corrected/'job.json')
    require(pipeline_job['status']=='failed' and corrected_job['status']=='verified' and corrected_job['original_inputs_unchanged'] is True
            and corrected_job['original_failed_job_sha256']==file_sha(pipeline/'job.json')
            and corrected_job['tool_sha256']==inputs[str(repo/'tools/task_pipeline_parity.py')],'pipeline provenance/corrected checker differs')
    pipeline_allocation=repo/'docs/experiments/task-pipeline-parity-allocation.json';pipeline_plan=sealed(pipeline_allocation)
    require(file_sha(pipeline_allocation)==pipeline_job['allocation_sha256'],'pipeline allocation differs')
    actual=compare_profiles(pipeline/'reference',pipeline/'candidate',pipeline_plan)
    require(actual==sealed(corrected/'comparison.json') and file_sha(corrected/'comparison.json')==corrected_job['comparison_sha256'],'fresh complete update/state comparison differs')
    for body in actual['comparisons'].values():
        for root,entries in body['input_snapshots'].items():
            for name,digest in entries.items():inputs[str(Path(root)/name)]=digest
    dependencies={name:metadata.version(name) for name in ('numpy','scipy','torch')}
    original_manifest=sealed(repo/'runs/nfsp-eligibility-R4-primary-ca45a53/neural/run.json')
    require(original_manifest['dependencies']==dependencies,'current numerical packages differ from original full A2 run')
    require(inventory(reference)[1]==old_files and inventory(candidate)[1]==new_files and all(file_sha(p)==digest for p,digest in inputs.items()),'source/evidence changed during source qualification')
    return {'version':VERSION,'status':'qualified_cpu_features_A2_source_bridge','source_compatibility_qualified':True,
        'reference_source_sha256':old,'candidate_source_sha256':new,'families':list(FAMILIES),'seeds':[0,1,2],
        'selected_multiround_reuse':4,'matrix_reuse':1,'target':'one-step','storage':'features','device':'cpu',
        'dependencies':dependencies,'default_collection_ast_sha256':specialized['reference_collection_ast_sha256'],
        'bounded_domain_cases':domains['case_count'],'trained_actor_archives':len(actor_results),'exact_complete_pipeline_pairs':len(actual['comparisons']),
        'unchanged_dependency_modules':differences['unchanged_dependency_modules'],'baseline_imports':agent_imports,'preserved_rule_reference_ast_sha256':resolver,
        'input_files_sha256':inputs,'verifier_sha256':file_sha(__file__),'qualification_seconds':time.monotonic()-started,
        'original_source_identities_preserved':True,'cross_source_trainer_resume_allowed':False,'new_training_records':0,'new_scored_evaluation_games':0,
        'a2_complete':False,'pilot_authorized':False,'execution_authorized':False,
        'scope':'Compatibility of the fixed CPU/features A2 task APIs under the recorded numerical packages and primary recipes, preserving original source/cohort/seed identities. Original case audits, complete cohorts, control interpretation, resource/project selection, 9x9 smoke and production entry remain required. Excludes graph/CUDA task actors, native rule-state storage, unrestricted game domains and changed recipes.'}


def qualify(repo,domain_run,allocation,*,expected_allocation_sha):
    repo=Path(repo).resolve()
    roots=[repo/'runs/nfsp-eligibility-R4-primary-ca45a53'/learner for learner in ('neural','tabular')]
    roots.extend(repo/'runs/task-pipeline-parity-d724e06'/side/f'{learner}-R4-{phase}'
                 for side in ('reference','candidate') for learner in ('neural','tabular') for phase in ('pre','post'))
    with ExitStack() as stack:
        for root in roots:
            lock=stack.enter_context((root/'.trainer.lock').open('rb'))
            fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        return _qualify_locked(repo,domain_run,allocation,expected_allocation_sha=expected_allocation_sha)


def main(argv=None):
    parser=argparse.ArgumentParser()
    for name in ('repo','domain-run','allocation','expected-allocation-sha','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    try:
        out=Path(args.output).resolve();require(not out.exists() and not out.is_relative_to(Path(args.domain_run).resolve()),'new qualification output outside original domain evidence required')
        result=qualify(args.repo,args.domain_run,args.allocation,expected_allocation_sha=args.expected_allocation_sha);out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
        print(json.dumps({key:result[key] for key in ('status','bounded_domain_cases','trained_actor_archives','exact_complete_pipeline_pairs','qualification_seconds','a2_complete')},indent=2));return 0
    except (ValueError,RuntimeError,OSError,KeyError,StopIteration) as exc:parser.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())

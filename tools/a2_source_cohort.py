"""Same-protocol A2 aggregation across explicitly admitted implementations.

Original wrappers/plans/source identities are never rewritten. All numerical
rules remain the legacy fixed-cohort rules; this grants no project entry.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from tools.a2_cohort import (NFSPA2Plan,checked_case,canonical,digest,fingerprint,require,
    slots,label,thresholds,FAMILIES,SEEDS,LEARNERS,ASSESS_VERSION,input_snapshot,read_lock)
from tools import task_source_bridge,terminal_batch_admission

REGISTRY_VERSION='graygo/a2-source-registry/1'
ASSESSMENT_VERSION='graygo/source-aware-a2-assessment/1'


def protocol_key(plan):
    value=plan.to_dict();value.pop('name');value.pop('learner_source_sha256');return canonical(value)


def _bridge_metadata(proof,*,current=True):
    if proof.get('version')==task_source_bridge.VERSION:
        require(proof['status']=='qualified_cpu_features_A2_source_bridge' and proof['source_compatibility_qualified'] is True,'CPU/features bridge not qualified')
        families=set(proof['families']);verifier=Path(task_source_bridge.__file__)
    elif proof.get('version')==terminal_batch_admission.VERSION:
        require(proof['status']=='qualified_cpu_matrix_terminal_batch' and proof['collection_implementation']=='terminal-batch','terminal collector not admitted')
        families=set(proof['families']);require(families=={'asymmetric','rps'},'terminal admission expanded its task scope');verifier=Path(terminal_batch_admission.__file__)
    else:raise ValueError('unknown source admission verifier/version')
    require(families and families<=set(FAMILIES) and fingerprint(proof['reference_source_sha256']) and fingerprint(proof['candidate_source_sha256']),'invalid admitted source edge')
    if current:require(proof['verifier_sha256']==digest(verifier),'use the receipt-bound verifier implementation')
    require(proof['execution_authorized'] is False,'implementation admission cannot grant execution')
    return {'left':proof['reference_source_sha256'],'right':proof['candidate_source_sha256'],'families':families,'proof':proof}



def check_selected_cpu_case(manifest,proof):
    """Admit a case produced by the selected package under its own identity."""
    from graygo.nfsp_training import NFSPMatrixConfig
    from graygo.task_training import NFSPTaskConfig
    require(proof.get('version')==task_source_bridge.VERSION and proof.get('status')=='qualified_cpu_features_A2_source_bridge'
            and proof.get('source_compatibility_qualified') is True and proof.get('execution_authorized') is False,'qualified scoped CPU/features bridge required')
    require(manifest['source_sha256']==proof['candidate_source_sha256'],'own-source case is not the selected CPU package')
    config=manifest['config'];family=config['task'];matrix=family in ('asymmetric','rps')
    require(family in proof['families'] and family in FAMILIES and config['learner'] in ('neural','tabular')
            and type(config['seed']) is int and config['seed'] in (0,1,2) and config['device']=='cpu','case outside selected CPU family/learner/seed scope')
    parsed=NFSPMatrixConfig(**config) if matrix else NFSPTaskConfig(**config)
    require(parsed.formal_protocol() and parsed.q_reuse==(1 if matrix else 4),'selected-source case differs from primary recipe')
    if not matrix:
        require(parsed.storage=='features' and parsed.native_library is None,'native/rule-state task is outside selected CPU admission')
        runtime=manifest.get('runtime',{})
        require(runtime.get('device')=='cpu' and type(runtime.get('torch_threads')) is int and runtime['torch_threads']==1
                and type(runtime.get('workers')) is int and runtime['workers']==1,'selected CPU task runtime differs')
    require(manifest['dependencies']==proof['dependencies'],'selected CPU numerical dependencies differ')
    return {'source_admission_applies':True,'original_source_identity_preserved':True,'original_case_plan_and_audit_required':True,
            'a2_complete':False,'execution_authorized':False}


def path_for(edges,source,target,family):
    todo=[(source,[])];seen=set()
    while todo:
        current,path=todo.pop(0)
        if current==target:return path
        if current in seen:continue
        seen.add(current)
        for index,edge in enumerate(edges):
            if family not in edge['families']:continue
            other=edge['right'] if current==edge['left'] else edge['left'] if current==edge['right'] else None
            if other is not None and other not in seen:todo.append((other,path+[index]))
    return None


def _fresh_bridge(repo,proof):
    if proof['version']==task_source_bridge.VERSION:
        fresh=task_source_bridge.qualify(repo,Path(repo)/'runs/task-source-bridge-446a058',
            Path(repo)/'.worktrees/task-source-bridge-446a058/docs/experiments/task-source-bridge-domain-allocation.json',
            expected_allocation_sha='11b97d2adeb8e68ce78142c210a83f2f8f85c50975665b7126dc05be48082b32')
    else:
        e=proof['evidence'];fresh=terminal_batch_admission.qualify(e['reference'],e['candidate'],e['comparison'],e['audits'],e['suite'],
            expected_reference=proof['reference_source_sha256'],expected_candidate=proof['candidate_source_sha256'])
    left,right=dict(proof),dict(fresh);left.pop('qualification_seconds');right.pop('qualification_seconds')
    require(left==right,'fresh source admission differs from the registered receipt')


def read_registry(path,*,repo,current=True):
    wrapper=json.loads(Path(path).read_text());body=wrapper.get('body');require(wrapper.get('sha256')==canonical(body) and body.get('version')==REGISTRY_VERSION,'source registry checksum/version differs')
    require(body['original_case_and_source_identity_rewriting_allowed'] is False and body['a2_complete'] is False and body['execution_authorized'] is False,'source registry became execution authority')
    plans={}
    for row in body['plans']:
        plan=NFSPA2Plan.from_dict(row['plan']);require(plan.sha256==row['plan_sha256'] and plan.sha256 not in plans,'duplicate or changed original source plan')
        if current:require(digest(Path(row['path']))==row['file_sha256'] and json.loads(Path(row['path']).read_text())==plan.to_dict(),'original registered plan changed')
        plans[plan.sha256]=plan
    require(body['canonical_plan_sha256'] in plans,'canonical original plan missing');base=plans[body['canonical_plan_sha256']]
    require(all(protocol_key(plan)==body['protocol_sha256']==protocol_key(base) for plan in plans.values()),'source plans differ in learner protocol, gates, seeds or audit definitions')
    edges=[]
    for row in body['bridges']:
        path=Path(row['path']);proof=row['proof']
        if current:require(digest(path)==row['file_sha256'] and json.loads(path.read_text())==proof,'registered source bridge changed')
        edge=_bridge_metadata(proof,current=current)
        if current:_fresh_bridge(repo,proof)
        edges.append(edge)
    for family in FAMILIES:require(path_for(edges,base.learner_source_sha256,body['target_runtime_source_sha256'],family) is not None,'canonical task implementation has no compatible target runtime')
    supported=set()
    for edge in edges:
        if edge['proof']['version']==task_source_bridge.VERSION:supported.update((edge['left'],edge['right']))
        else:supported.add(edge['right'])
    for plan in plans.values():
        require(plan.learner_source_sha256 in supported,'source has no implemented primary-case admission checker')
        require(any(path_for(edges,plan.learner_source_sha256,base.learner_source_sha256,family) is not None for family in FAMILIES),'source plan has no admitted task path')
    return body,base,plans,edges


def create_registry(name,canonical_plan,plan_paths,bridge_paths,*,target_runtime_source,repo,output):
    require(isinstance(name,str) and name and fingerprint(target_runtime_source),'named canonical runtime registry required')
    plan_paths=[Path(p).resolve() for p in plan_paths];canonical_plan=Path(canonical_plan).resolve();require(canonical_plan in plan_paths,'include the canonical original plan')
    plans=[]
    for path in plan_paths:
        plan=NFSPA2Plan.from_dict(json.loads(path.read_text()));plans.append({'path':str(path),'file_sha256':digest(path),'plan_sha256':plan.sha256,'plan':plan.to_dict()})
    base=NFSPA2Plan.from_dict(json.loads(canonical_plan.read_text()))
    body={'version':REGISTRY_VERSION,'name':name,'canonical_plan_sha256':base.sha256,'protocol_sha256':protocol_key(base),
        'target_runtime_source_sha256':target_runtime_source,'plans':plans,
        'bridges':[{'path':str(Path(p).resolve()),'file_sha256':digest(Path(p)),'proof':json.loads(Path(p).read_text())} for p in bridge_paths],
        'original_case_and_source_identity_rewriting_allowed':False,'a2_complete':False,'execution_authorized':False,
        'scope':'registered same-protocol implementation compatibility only; fixed original slots/gates and original source-specific case audits remain required'}
    out=Path(output).resolve();require(not out.exists(),'preserve existing source registry')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as file:json.dump({'body':body,'sha256':canonical(body)},file,indent=2,sort_keys=True);file.write('\n');file.flush();os.fsync(file.fileno())
    try:read_registry(out,repo=repo,current=True)
    except BaseException:
        # The invalid immutable draft remains reviewable; never replace its bytes.
        raise
    return body


# Fixed numerical aggregation is copied from the preserved legacy implementation.
# Only the loop input changes from wrappers to individually checked original cases.
def _aggregate_verified(plan,verified):
    present={};bodies={};roots=set();manifests=set()
    for key,body,summary in verified:
        require(key not in present,'duplicate A2 case cannot replace an original seed')
        require(body['root'] not in roots and body['provenance']['training_manifest_sha256'] not in manifests,'different original A2 cases cannot alias one run')
        roots.add(body['root']);manifests.add(body['provenance']['training_manifest_sha256'])
        present[key]=summary;bodies[key]=body
    missing=[label(key) for key in slots() if key not in present]
    groups={}
    for learner in LEARNERS:
        keys=[key for key in slots() if key[1]==learner]
        complete=all(key in present and present[key]['status'] in ('numeric_pass','numeric_miss') for key in keys)
        passed=complete and all(present[key]['numeric_passed'] for key in keys)
        groups[learner]={'complete':complete,'status':'incomplete' if not complete else ('numeric_pass' if passed else 'numeric_miss'),
            'numeric_passed':passed,'missing_cases':[label(key) for key in keys if key not in present],
            'case_misses':[label(key) for key in keys if key in present and present[key]['status']=='numeric_miss']}
    extension={'numeric_eligible':False,'selected':False,'resource_authorization_verified':False,
        'budget_reference_sha256':plan.extension_budget_sha256,'reason':'complete original neural cohort required'}
    if groups['neural']['complete']:
        failures=[key for key in present if key[1]=='neural' and not present[key]['numeric_passed']]
        extension['reason']='exactly one original neural family/seed must miss'
        if len(failures)==1:
            key=failures[0];final=bodies[key]['evaluations'][-5:];limits=thresholds(key[0])
            last=all(final[-1][metric]<=limit for metric,limit in limits.items()) and final[-1]['average_ready']
            marginal=all(row['average_ready'] and all(row[metric]<=1.2*limit for metric,limit in limits.items()) for row in final[:-1])
            extension['reason']='last evaluation must pass and every earlier miss must be within twenty percent'
            if last and marginal:
                extension.update(numeric_eligible=True,reason='numerically marginal; budget/entry authority remains separate',family=key[0],failing_seed=key[2],
                    paired_seeds=list(SEEDS),maximum_additional_records_per_seed=2000000,required_endpoint=17000000,
                    original_case_sha256=[present[(key[0],'neural',seed)]['case_sha256'] for seed in SEEDS])
    delayed=[present.get(('delayed','neural',seed)) for seed in SEEDS]
    worst_delayed=None if not all(item and 'final_five_maxima' in item for item in delayed) else max(item['final_five_maxima']['max_average_gap'] for item in delayed)
    neural_band='incomplete' if not groups['neural']['complete'] else ('all_neural_numeric_gates_pass' if groups['neural']['numeric_passed'] else 'not_ready')
    if groups['neural']['complete'] and worst_delayed is not None and .10<worst_delayed<=.25:
        other_ok=all(summary['numeric_passed'] for key,summary in present.items() if key[1]=='neural' and key[0]!='delayed')
        if other_ok and all(item['q_credit_passed'] for item in delayed):neural_band='averaging_lag'
    return {'version':ASSESS_VERSION,'plan_sha256':plan.sha256,'learner_source_sha256':plan.learner_source_sha256,
        'required_cases':len(slots()),'present_cases':len(present),'missing_cases':missing,
        'cases':{label(key):value for key,value in sorted(present.items())},'neural':groups['neural'],'controls':groups['tabular'],
        'neural_deployment_band':neural_band,'worst_delayed_average_gap':worst_delayed,
        'control_interpretation_required':bool(groups['tabular']['case_misses']),'marginal_extension':extension,
        'a2_complete':False,'pilot_authorized':False,'extension_authorized':False,
        'independent_training_trajectory_verified':False,'original_cohort_registration_verified':False,'resource_scope_verified':False,
        'scope':'numerical fixed-cohort assessment of source-bound audited case snapshots; control outcomes remain separate; original registration/trajectory, comparator/fixture checks, 9x9 smoke and filled resource/entry decision remain required'}


def assess_sources(base,plans,allowed,wrappers):
    require(base.sha256 in plans and all(protocol_key(p)==protocol_key(base) for p in plans.values()),'incompatible source-specific A2 plans')
    verified=[];origins={}
    for wrapper in wrappers:
        plan_sha=wrapper['body']['plan_sha256'];require(plan_sha in plans,'case belongs to an unregistered original plan')
        plan=plans[plan_sha];key,body,score=checked_case(plan,wrapper)
        require(key[0] in allowed.get(plan.learner_source_sha256,set()),'case family is not admitted for its original source')
        require(body['primary_recipe'],'diagnostic snapshots cannot fill the registered primary cohort')
        verified.append((key,body,score));origins[label(key)]={'original_plan_sha256':plan.sha256,'original_source_sha256':plan.learner_source_sha256,
            'original_case_sha256':wrapper['sha256'],'original_root':body['root'],'original_training_manifest_sha256':body['provenance']['training_manifest_sha256']}
    result=_aggregate_verified(base,verified)
    result.pop('plan_sha256');result.pop('learner_source_sha256');result['version']=ASSESSMENT_VERSION
    result.update(protocol_sha256=protocol_key(base),canonical_original_plan_sha256=base.sha256,case_provenance=origins,
        source_identities_preserved=True,source_admission_grants_training=False)
    return result


def assess_registry_files(registry,case_paths,*,repo,current=True,output=None):
    body,base,plans,edges=read_registry(registry,repo=repo,current=current);paths=[Path(p).resolve() for p in case_paths];wrappers=[json.loads(p.read_text()) for p in paths]
    allowed={p.learner_source_sha256:{f for f in FAMILIES if path_for(edges,p.learner_source_sha256,base.learner_source_sha256,f) is not None} for p in plans.values()}
    for wrapper in wrappers:
        require(wrapper['body']['plan_sha256'] in plans,'unregistered original case plan');checked_case(plans[wrapper['body']['plan_sha256']],wrapper)
    roots=[Path(w['body']['root']).resolve() for w in wrappers]
    if output is not None:
        out=Path(output).resolve();require(not out.exists() and out not in paths and not any(out.is_relative_to(root) for root in roots),'new assessment outside original inputs required')
    with ExitStack() as stack:
        if current:
            for root in sorted(set(roots)):read_lock(stack,root)
            for wrapper,root in zip(wrappers,roots):
                require(input_snapshot(root)==wrapper['body']['input_snapshot'],'original case snapshot is stale')
                manifest=json.loads((root/'run.json').read_text());source=manifest['source_sha256'];family=wrapper['body']['case'][0]
                require(source==wrapper['body']['provenance']['learner_source_sha256'],'original manifest source differs')
                applied=False
                for edge in edges:
                    proof=edge['proof']
                    if proof['version']==task_source_bridge.VERSION and source==proof['reference_source_sha256']:
                        task_source_bridge.check_case(manifest,proof,selected_candidate_source=proof['candidate_source_sha256']);applied=True
                    elif proof['version']==task_source_bridge.VERSION and source==proof['candidate_source_sha256']:
                        check_selected_cpu_case(manifest,proof);applied=True
                    elif proof['version']==terminal_batch_admission.VERSION and source==proof['candidate_source_sha256']:
                        terminal_batch_admission.check_matrix_case(manifest,proof);applied=True
                require(applied,'original case execution/configuration has no verified admission')
        result=assess_sources(base,plans,allowed,wrappers)
        result.update(registry_sha256=canonical(body),target_runtime_source_sha256=body['target_runtime_source_sha256'],
            current_input_snapshots_verified=current,current_source_admission_inputs_verified=current,
            analysis_mode='current_inputs' if current else 'archived_snapshots',assessor_sha256=digest(Path(__file__)),
            original_case_files_sha256={str(path):digest(path) for path in paths})
        if output is not None:
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.open('x') as file:json.dump(result,file,indent=2,sort_keys=True,allow_nan=False);file.write('\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('register');p.add_argument('--name',required=True);p.add_argument('--canonical-plan',required=True);p.add_argument('--plan',action='append',required=True);p.add_argument('--bridge',action='append',required=True);p.add_argument('--target-runtime-source',required=True);p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('assess');p.add_argument('--registry',required=True);p.add_argument('--case',action='append',required=True);p.add_argument('--repo',required=True);p.add_argument('--archived',action='store_true');p.add_argument('--output',required=True)
    a=parser.parse_args(argv)
    try:
        if a.command=='register':result=create_registry(a.name,a.canonical_plan,a.plan,a.bridge,target_runtime_source=a.target_runtime_source,repo=a.repo,output=a.output)
        else:result=assess_registry_files(a.registry,a.case,repo=a.repo,current=not a.archived,output=a.output)
        print(json.dumps({'status':'registered' if a.command=='register' else result['neural_deployment_band'],'a2_complete':False},indent=2));return 0
    except (ValueError,RuntimeError,OSError,KeyError) as exc:parser.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())

"""Read-only profile prices and an explicit, unselected control-scope proposal."""
import argparse
from copy import deepcopy
from decimal import Decimal,ROUND_CEILING
import json
from pathlib import Path
from graygo.storage import digest as _digest
from graygo.task_training import NFSPTaskConfig
from tools.nfsp_delayed_cpu_prices import runtime_identity,IDENTITY_KEYS
from tools.nfsp_resources import canonical,number,require,measured_usage,Usage

VERSION='graygo/control-completion-budget-proposal/1'
SOURCE_PATHS={'a522fe224ccf6db1c528d3fea1c896869038b29e48dbda9e00174ca8116a24df':'.worktrees/frozen-cuda-c3b8b5f/src/graygo','db9db2eb532d0a2782486a486743701a0a1914277616ff5d5c0406f3b6501873':'.worktrees/nfsp-eligibility-ca45a53/src/graygo'}
def digest(path):return _digest(Path(path))
def load(path):return json.loads(Path(path).read_text())


def price_pair(task,learner,reuse,rows,configs):
    require(task in ('delayed','branching','gray-pure','gray-mixed') and learner in ('neural','tabular') and type(reuse) is int and reuse in (1,4),'named one-step task price required')
    require(task=='delayed' or reuse==4,'remaining short families use selected R4')
    require(len(rows)==2 and {r['phase'] for r in rows}=={'pre','post'} and set(configs)=={'pre','post'},'distinct complete pre/post profiles required')
    phases={r['phase']:r for r in rows};counts=set()
    for phase,row in phases.items():
        count=row['records'];require(type(count) is int and count>=25600 and count%4==0,'complete paired profile with two target refreshes required');counts.add(count)
        config=NFSPTaskConfig(**configs[phase]);expected=NFSPTaskConfig(time_cap_seconds=config.time_cap_seconds,task=task,learner=learner,q_reuse=reuse,records=count,eval_every=count//2,purpose='profile',reservoir_open=5000000 if phase=='pre' else 0,handoff_end=7000000 if phase=='pre' else 1000)
        require(config.plan()==expected.plan(),'profile changed primary learner/model/storage/schedule settings')
        require(row['status']=='complete' and row['learner']==learner and row['q_reuse']==reuse and not row['formal_protocol'] and number(row['seconds_including_setup'])>0,'completed diagnostic timing required')
        require(row['counters']['fresh_records']==count and row['counters']['q_updates']>0 and row['counters']['target_refreshes']>=2,'complete Q/update/target work required')
        if phase=='pre':require(row['counters']['supervised_updates']==0 and row['acting_policy_counts']['average']==0,'pre-opening reservoir must stay closed')
        else:require(row['counters']['supervised_updates']>0 and row['acting_policy_counts']['average']>0 and row['reservoir_size']>=config.batch,'post-opening actual average-policy work required')
    require(len(counts)==1,'matched pre/post record counts required')
    forecast=sum(Decimal(n)/Decimal(phases[p]['records'])*number(phases[p]['seconds_including_setup']) for p,n in [('pre',5000000),('post',10000000)])
    cap=max(Decimal(900),(2*forecast/900).to_integral_value(rounding=ROUND_CEILING)*900)
    return {'task':task,'learner':learner,'q_reuse':reuse,'device':'cpu','device_hours':'0','cpu_process_ceiling_seconds':str(cap),'phase_weighted_15M_seconds':str(forecast),'runtime_margin':'2','formal_record_cap':15000000,'original_seed_scope':[0,1,2]}


def approved_control_limit(proposal,decision):
    require(proposal.get('version')==VERSION and proposal.get('selected') is False,'original unselected scope proposal required')
    require(isinstance(decision,dict) and decision.get('approved') is True,'explicit owner approval required to exceed the original control ceiling')
    require(decision.get('proposal_sha256')==canonical(proposal),'owner decision belongs to another proposal')
    require(decision.get('control_ceiling_seconds')==proposal['proposed_control_ceiling_seconds'] and decision.get('case_ids')==[c['id'] for c in proposal['cases']],'approved ceiling/case scope changed')
    return number(proposal['proposed_control_ceiling_seconds'])


def usage_with_approved_control_scope(inventory,proposal,decision):
    limit=approved_control_limit(proposal,decision);result=measured_usage([Usage(**r) for r in inventory['measurements']]);control=result['cpu_categories']['control'];committed=number(control['committed_process_seconds'])
    control.update(ceiling_seconds=str(limit),within_ceiling=committed<=limit,unallocated_seconds=str(limit-committed))
    result['control_scope_decision']={'proposal_sha256':canonical(proposal),'decision_sha256':canonical(decision),'original_default_ceiling_seconds':'43200','explicit_selected_ceiling_seconds':str(limit)}
    return result


def build_proposal(repo,evidence):
    repo=Path(repo).resolve();spec=load(evidence);inputs={str(Path(evidence).resolve()):digest(evidence)}
    def sealed(path):
        path=Path(path).resolve();inputs[str(path)]=digest(path);return load(path)
    corpus=[]
    for key,allocation_key in [('cpu_audit_root','cpu_audit_allocation'),('legacy_audit_root','legacy_audit_allocation')]:
        root=repo/spec[key];job=sealed(root/'job.json');require(job['original_inputs_unchanged'] is True,'independent audit original inputs changed')
        declaration_path=repo/spec[allocation_key];declaration=sealed(declaration_path);require(digest(declaration_path)==spec[allocation_key+'_sha256']==job['allocation_sha256'],'independent audit allocation changed');inputs[str((root/'driver.py').resolve())]=digest(root/'driver.py');require(digest(root/'driver.py')==job['driver_sha256'],'independent audit driver changed')
        require(job['new_training_records']==job['new_scored_evaluation_games']==job['new_accelerator_jobs']==0,'read-only audit scope changed')
        require(all(digest(p)==h for p,h in job['original_inputs_sha256'].items()),'audited inputs changed after the original pass');inputs.update(job['original_inputs_sha256'])
        outputs={r['case']:Path(r['output']) for r in job['commands']};corpus.append((root,job,outputs,declaration))
    require(corpus[0][1]['status']=='verified_all_twelve_original_profiles' and len(corpus[0][1]['cases'])==12,'all twelve direct CPU profile audits required')
    identities=[];prices=[];selected_source=spec['selected_source_sha256'];legacy_source=spec['legacy_source_sha256']
    def checked_price(profile_root,task,learner,reuse,source,audit_index,labels):
        profile_root=repo/profile_root;profile=sealed(profile_root/'profile.json');plan=sealed(profile_root/'plan.json');budget_path=profile_root/'budgets'/f'{learner}-R{reuse}.json';budget=sealed(budget_path)
        require(sealed(profile_root/'state.json')['status']=='complete' and profile['plan']==plan and plan['source_sha256']==source,'original complete source-pinned profile required')
        rows=[r for r in profile['results'] if r['learner']==learner and r['q_reuse']==reuse];require(rows==budget['profile_cases'] and budget['source_sha256']==source and budget['complete_pipeline'] is True and budget['device']=='cpu' and budget['storage']=='features' and budget['native_library_sha256'] is None and budget['target']=='one-step','raw budget is not the complete selected CPU pipeline')
        require(budget['profile_plan_sha256']==digest(profile_root/'plan.json') and budget['margin']==2.,'original profile plan or twofold margin changed');configs={}
        _,job,outputs,declaration=corpus[audit_index]
        for phase in ('pre','post'):
            row=next(r for r in rows if r['phase']==phase);folder=profile_root/row['case'];manifest=sealed(folder/'run.json');result=sealed(folder/'result.json');label=labels(phase);require(job['cases'][label]['verified'] is True and job['cases'][label]['returncode']==0,'selected profile independent audit failed')
            output=outputs[label];audit=sealed(output);require(digest(output)==job['cases'][label]['output_sha256'] and audit['verified'] is True and audit['source_sha256']==source and audit['training_manifest_sha256']==digest(folder/'run.json'),'independent audit identity differs')
            declared=declaration['cases'][label];auditor_path=repo/declaration['auditor'] if audit_index==0 else repo/declared['auditor'];auditor_sha=declaration['auditor_sha256'] if audit_index==0 else declared['auditor_sha256'];inputs[str(auditor_path.resolve())]=digest(auditor_path);require(digest(auditor_path)==auditor_sha==audit['auditor_sha256'],'receipt-bound independent auditor changed');require((repo/declared['root']).resolve()==folder.resolve() and declared['manifest_sha256']==digest(folder/'run.json') and declared['result_sha256']==digest(folder/'result.json') and declared['config']==manifest['config'],'declared original profile differs')
            require(audit['task']==task and audit['learner']==learner and audit['seed']==0 and audit['records']==row['records'] and not audit['formal_protocol'],'independent diagnostic case differs')
            require(manifest['source_sha256']==result['source_sha256']==source and manifest['config']['seed']==0 and result['status']=='complete' and result['records']==row['records'] and result['counters']==row['counters'] and result['acting_policy_counts']==row['acting_policy_counts'],'actual completed profile work differs from price rows')
            require(abs(number(row['seconds_including_setup'])-number(result['elapsed_seconds'])-number(result['timings']['setup']))<Decimal('.00000001'),'timing omitted constructor or loop')
            identity={k:manifest[k] for k in IDENTITY_KEYS};require(identity['runtime']['device']=='cpu' and identity['runtime']['workers']==identity['runtime']['torch_threads']==1,'profile runtime outside CPU single-worker/thread scope');identities.append(identity);configs[phase]=manifest['config']
        price=price_pair(task,learner,reuse,rows,configs)
        require(number(price['cpu_process_ceiling_seconds'])==number(budget['proposed_time_ceiling_seconds']) and abs(number(price['phase_weighted_15M_seconds'])-number(budget['phase_weighted_15M_seconds']))<Decimal('.000001'),'independent price arithmetic differs')
        return price|{'source_sha256':source,'source_package_path':str(Path(SOURCE_PATHS[source]).parent),'raw_budget':str(budget_path.relative_to(repo)),'raw_budget_sha256':digest(budget_path),'forecast_limits':budget['limits']}
    for task in ('branching','gray-pure','gray-mixed'):
        for learner in ('tabular','neural'):
            prices.append(checked_price(Path(spec['cpu_profile_root'])/'profiles'/task/learner,task,learner,4,selected_source,0,lambda phase,t=task,l=learner:t+'/'+l+'/'+phase))
    for reuse in (1,4):prices.append(checked_price(Path(f'runs/nfsp-eligibility-R{reuse}-profiles-ca45a53'),'delayed','tabular',reuse,legacy_source,1,lambda phase,r=reuse:f'legacy-delayed-table-R{r}-'+phase))
    require(all(i==identities[0] for i in identities) and runtime_identity()==identities[0],'current numerical/host/runtime identity differs from measured CPU profiles')
    import hashlib
    for source,path in SOURCE_PATHS.items():
        p=repo/path;h=hashlib.sha256()
        for file in sorted([*p.glob('*.py'),*(p/'native').glob('*.cpp')]):h.update(file.relative_to(p).as_posix().encode()+b'\0'+file.read_bytes()+b'\0')
        require(h.hexdigest()==source,'priced frozen source package changed')
    usage=sealed(repo/spec['current_usage']);current=number(usage['review']['cpu_categories']['control']['committed_process_seconds']);require(current<=43200,'original current control ceiling already exceeded')
    cases=[]
    for price in prices:
        if price['learner']!='tabular':continue
        seeds=(1,2) if price['task']=='delayed' else (0,1,2);role='comparison' if price['q_reuse']==1 else 'primary'
        for seed in seeds:
            config=NFSPTaskConfig(time_cap_seconds=int(price['cpu_process_ceiling_seconds']),task=price['task'],learner='tabular',seed=seed,q_reuse=price['q_reuse'],budget_profile_sha256=price['raw_budget_sha256']);require(config.formal_protocol(),'full original control recipe required')
            cases.append({'id':f'{role}/{price["task"]}/tabular/{seed}/R{price["q_reuse"]}','cohort_role':role,'task':price['task'],'learner':'tabular','seed':seed,'q_reuse':price['q_reuse'],'source_sha256':price['source_sha256'],'source_package_path':price['source_package_path'],'raw_budget':price['raw_budget'],'raw_budget_sha256':price['raw_budget_sha256'],'complete_process_ceiling_seconds':price['cpu_process_ceiling_seconds'],'config':config.plan()})
    require(len(cases)==13 and len({c['id'] for c in cases})==13,'all eleven missing primary and two original R1 comparison controls required')
    additional=sum(number(c['complete_process_ceiling_seconds']) for c in cases);projected=current+additional;hours=(projected/3600).to_integral_value(rounding=ROUND_CEILING)+1;limit=hours*3600
    require(all(digest(p)==h for p,h in inputs.items()),'pricing evidence changed during review')
    return {'version':VERSION,'builder_sha256':digest(Path(__file__)),'selected':False,'owner_approved':False,'original_control_ceiling_seconds':'43200','current_control_commitment_seconds':str(current),'additional_control_reservation_seconds':str(additional),'projected_control_commitment_seconds':str(projected),'proposed_control_ceiling_seconds':str(limit),'proposed_control_ceiling_hours':str(hours),'headroom_after_named_reservations_seconds':str(limit-projected),'case_count':13,'primary_case_count':11,'comparison_case_count':2,'aggregate_record_cap':195000000,'cases':cases,'audited_cpu_prices':prices,'runtime_identity':identities[0],'evidence_files_sha256':inputs,'cuda_variant_admitted':False,'legacy_audit_job_gpu_failures_preserved':[k for k,v in corpus[1][1]['cases'].items() if not v['verified']],'new_training_selected':False,'a2_complete':False,'project_study_scope_selected':False,'execution_authorized':False,'approval_basis':'docs/design.md section 6.2 fixes twelve elapsed hours for tabular controls; the thirteen named original full controls exceed that ceiling. The proposed limit covers their complete process caps with the original twofold profile margins, retaining all previous costs and unknown overhead. No seed filtering, changed record/gap clocks, optional extension or other case is authorized. Partial-buffer/scaled checkpoint/startup/evaluation forecast limits remain.'}


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--repo',required=True);parser.add_argument('--evidence',required=True);parser.add_argument('--output',required=True);args=parser.parse_args(argv)
    if Path(args.output).exists():parser.error('proposal output must be new; preserve original evidence')
    proposal=build_proposal(args.repo,args.evidence);Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(proposal,indent=2,sort_keys=True)+'\n');print(json.dumps({k:proposal[k] for k in ['case_count','additional_control_reservation_seconds','projected_control_commitment_seconds','proposed_control_ceiling_hours','selected']},sort_keys=True))


if __name__=='__main__':main()

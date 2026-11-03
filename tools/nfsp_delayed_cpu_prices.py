"""Qualify prospective CPU-only delayed NFSP prices from original profiles.

A zero accelerator price requires CPU execution. CPU ceilings, original evidence,
A2 cohort completion and project authorization remain separate quantities.
"""
import argparse
from contextlib import ExitStack
from decimal import Decimal, ROUND_CEILING
import fcntl
from importlib import metadata
import json
from pathlib import Path
import platform
import socket
import sys
import time
import torch
from graygo.manifest import write_report
from graygo.storage import digest as _storage_digest, CheckpointStore
from graygo.task_training import NFSPTaskConfig
from tools.a2_source_cohort import _fresh_bridge
from tools import task_source_bridge
from tools.nfsp_resources import number, require

VERSION='graygo/delayed-cpu-runtime-prices/1'
PROPOSAL_VERSION='graygo/prospective-delayed-cpu-prices/1'
IDENTITY_KEYS=('hostname','cpu','python','platform','dependencies','runtime')


def digest(path):return _storage_digest(Path(path))

def inputs_match(inputs):return bool(inputs) and all(digest(path)==value for path,value in inputs.items())


def load(path):return json.loads(Path(path).read_text())


def profile_price(reuse,rows,configs):
    """Check the entire profile recipe and independently price the 15M mixture."""
    require(type(reuse) is int and reuse in (1,4),'one-step reuse one or four required')
    require(len(rows)==2 and {r['phase'] for r in rows}=={'pre','post'} and set(configs)=={'pre','post'},'complete distinct pre/post pair required')
    by_phase={r['phase']:r for r in rows};records=set()
    for phase,row in by_phase.items():
        count=row['records'];records.add(count)
        require(type(count) is int and count>=25600 and count%4==0,'at least two refreshes and paired profile reads required')
        config=NFSPTaskConfig(**configs[phase])
        expected=NFSPTaskConfig(time_cap_seconds=config.time_cap_seconds,learner='neural',q_reuse=reuse,records=count,eval_every=count//2,purpose='profile',
            reservoir_open=5000000 if phase=='pre' else 0,handoff_end=7000000 if phase=='pre' else 1000)
        require(config.plan()==expected.plan(),'profile differs from the exact CPU/features delayed neural recipe')
        require(row['status']=='complete' and row['learner']=='neural' and row['q_reuse']==reuse and not row['formal_protocol'],'completed diagnostic learner required')
        require(number(row['seconds_including_setup'])>0,'positive complete profile timing required')
        require(row['counters']['fresh_records']==count and row['counters']['q_updates']>0 and row['counters']['target_refreshes']>=2,'profile omitted learning or target work')
        if phase=='pre':require(row['counters']['supervised_updates']==0 and row['acting_policy_counts']['average']==0,'pre-opening profile must keep the reservoir closed')
        else:require(row['counters']['supervised_updates']>0 and row['acting_policy_counts']['average']>0 and row['reservoir_size']>=config.batch,'post-opening profile omitted learned averaging')
    require(len(records)==1,'matched profile record counts required')
    forecast=sum(Decimal(n)/Decimal(by_phase[p]['records'])*number(by_phase[p]['seconds_including_setup']) for p,n in [('pre',5000000),('post',10000000)])
    cap=max(Decimal(900),(2*forecast/900).to_integral_value(rounding=ROUND_CEILING)*900)
    return {'device_hours':'0','cpu_process_ceiling_seconds':str(cap),'phase_weighted_15M_seconds':str(forecast),'runtime_margin':'2','formal_record_cap':15000000,'q_reuse':reuse,'original_seed_scope':[0,1,2]}



def combine_source_prices(reference,selected):
    for price in (reference,selected):
        require(price['device_hours']=='0' and price['runtime_margin']=='2' and price['formal_record_cap']==15000000 and price['original_seed_scope']==[0,1,2],'complete twofold CPU price required')
    require(reference['q_reuse']==selected['q_reuse'],'source profiles changed Q reuse')
    return selected|{'cpu_process_ceiling_seconds':str(max(number(reference['cpu_process_ceiling_seconds']),number(selected['cpu_process_ceiling_seconds']))),
        'phase_weighted_15M_seconds':str(max(number(reference['phase_weighted_15M_seconds']),number(selected['phase_weighted_15M_seconds']))),
        'reference_forecast_seconds':reference['phase_weighted_15M_seconds'],'selected_package_forecast_seconds':selected['phase_weighted_15M_seconds']}


def selected_profile_prices(repo,root,proposal,reference_prices,inputs,identities):
    job_path=root/'job.json';job=load(job_path);allocation_path=Path(job['allocation']);allocation=load(allocation_path)
    require(job['status']=='verified_selected_package_profiles_and_audits' and job['original_inputs_unchanged'] is True and job['diagnostic_records']==262144,'direct selected-package profiles and audits required')
    require(job['new_formal_a2_records']==job['new_tabular_control_records']==job['new_accelerator_jobs']==0 and job['execution_authorized'] is False,'selected profiles exceeded diagnostic scope')
    require(digest(allocation_path)==job['allocation_sha256'] and digest(repo/allocation['runner'])==job['runner_sha256']==allocation['runner_sha256'] and digest(repo/allocation['auditor'])==job['auditor_sha256']==allocation['auditor_sha256'],'selected profile/auditor producer changed')
    require(job['source_sha256']==allocation['learner_source_sha256']==proposal['target_runtime_source_sha256'],'direct profiles used another package')
    require(len(job['commands'])==5 and job['commands'][0]['label']=='profiles' and all(c['returncode']==0 for c in job['commands']) and set(job['cases'])=={'C1/pre','C1/post','C4/pre','C4/post'},'direct profile/audit inventory differs')
    inputs.update({str(job_path):digest(job_path),str(allocation_path):digest(allocation_path)})
    require(inputs_match(job['original_inputs_sha256']),'direct profile original inputs changed');inputs.update(job['original_inputs_sha256'])
    profile_root=root/'profiles';profile=load(profile_root/'profile.json');plan=load(profile_root/'plan.json')
    require(load(profile_root/'state.json')['status']=='complete' and profile['plan']==plan and plan['source_sha256']==job['source_sha256'] and plan['driver_sha256']==job['runner_sha256'] and len(profile['results'])==4,'direct matched profile plan differs')
    prices={}
    for term,reuse in [('C1',1),('C4',4)]:
        budget_path=profile_root/'budgets'/f'neural-R{reuse}.json';budget=load(budget_path);rows=[r for r in profile['results'] if r['learner']=='neural' and r['q_reuse']==reuse];configs={}
        require(budget['profile_cases']==rows and budget['source_sha256']==job['source_sha256'] and budget['profile_plan_sha256']==digest(profile_root/'plan.json') and budget['complete_pipeline'] is True,'direct budget was not generated from its complete profiles')
        require(budget['device']=='cpu' and budget['storage']=='features' and budget['native_library_sha256'] is None and budget['target']=='one-step' and budget['q_reuse']==reuse and budget['margin']==2.,'direct budget differs from CPU/features one-step scope')
        for row in rows:
            phase=row['phase'];label=term+'/'+phase;folder=profile_root/row['case'];manifest=load(folder/'run.json');result=load(folder/'result.json');checked=job['cases'][label];output=Path(checked['output']);audit=load(output)
            require(Path(checked['root']).resolve()==folder.resolve() and digest(output)==checked['output_sha256'] and audit['auditor_sha256']==job['auditor_sha256'] and audit['verified'] is True and not audit['formal_protocol'],'direct independent audit binding differs');inputs[str(output)]=digest(output)
            require(manifest['source_sha256']==result['source_sha256']==audit['source_sha256']==job['source_sha256'] and audit['training_manifest_sha256']==digest(folder/'run.json') and audit['episode_audit_sha256']==digest(folder/'episodes.bin'),'direct source/manifest/trajectory binding differs')
            bundle=CheckpointStore(folder).latest();CheckpointStore.verify(bundle);require(audit['checkpoint_manifest_sha256']==digest(bundle/'manifest.json'),'direct checkpoint binding differs')
            require(audit['learner']=='neural' and audit['task']=='delayed' and audit['seed']==0 and audit['records']==result['records']==row['records'] and result['status']=='complete' and result['counters']==row['counters'] and result['acting_policy_counts']==row['acting_policy_counts'],'direct completed learner work differs')
            require(abs(number(row['seconds_including_setup'])-number(result['elapsed_seconds'])-number(result['timings']['setup']))<Decimal('.00000001'),'direct timing omitted setup or loop')
            configs[phase]=manifest['config'];identities.append({k:manifest[k] for k in IDENTITY_KEYS})
        selected=profile_price(reuse,rows,configs)
        require(number(budget['proposed_time_ceiling_seconds'])==number(selected['cpu_process_ceiling_seconds']) and abs(number(budget['phase_weighted_15M_seconds'])-number(selected['phase_weighted_15M_seconds']))<Decimal('.000001'),'direct twofold budget arithmetic differs')
        prices[term]=combine_source_prices(reference_prices[term],selected)|{'raw_budget_sha256':digest(budget_path),'reference_raw_budget_sha256':reference_prices[term]['raw_budget_sha256'],'forecast_limits':budget['limits'],'price_basis':'larger CPU ceiling across original and direct selected-package complete pre/post profiles'}
    return prices


def runtime_identity():
    torch.set_num_threads(1)
    cpu=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    return {'hostname':socket.gethostname(),'cpu':cpu,'python':sys.version,'platform':platform.platform(),
        'dependencies':{p:metadata.version(p) for p in ('numpy','scipy','torch')},
        'runtime':{'device':'cpu','workers':1,'torch_threads':torch.get_num_threads(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic}}


def check_case(manifest,proof,term):
    """Recipe applicability; a caller must freshly read/qualify the receipt first."""
    require(proof.get('version')==VERSION and proof.get('status')=='qualified_prospective_cpu_delayed_prices' and term in ('C1','C4'),'qualified delayed CPU price required')
    require(manifest['source_sha256']==proof['target_runtime_source_sha256'],'price applies only to its selected package')
    price=proof['prices'][term];require(price['q_reuse']==(1 if term=='C1' else 4) and price['device_hours']=='0','price term/backend binding differs');config=NFSPTaskConfig(**manifest['config'])
    require(type(config.seed) is int and config.seed in (0,1,2),'original fixed seed required')
    expected=NFSPTaskConfig(time_cap_seconds=float(price['cpu_process_ceiling_seconds']),seed=config.seed,q_reuse=price['q_reuse'],budget_profile_sha256=price['raw_budget_sha256'])
    require(config.formal_protocol() and config.plan()==expected.plan(),'formal CPU delayed recipe or funded ceiling differs')
    identity={k:manifest[k] for k in IDENTITY_KEYS}
    require(identity==proof['runtime_identity'],'price numerical/runtime/host identity differs')
    return {'price_applies':True,'term':term,'device_hours':'0','cpu_process_ceiling_seconds':price['cpu_process_ceiling_seconds'],'a2_complete':False,'execution_authorized':False}


def qualify(repo,proposal_path,audit_root,selected_root):
    started=time.monotonic();repo=Path(repo).resolve();proposal_path=Path(proposal_path).resolve();audit_root=Path(audit_root).resolve();selected_root=Path(selected_root).resolve()
    proposal=load(proposal_path);job_path=audit_root/'job.json';job=load(job_path);allocation_path=Path(job['allocation']);allocation=load(allocation_path)
    require(proposal.get('version')==PROPOSAL_VERSION and proposal.get('status')=='proposal_pending_independent_price_qualification','original prospective proposal required')
    require(proposal['target_device']=='cpu' and proposal['storage']=='features' and proposal['native_library'] is None and proposal['target']=='one-step' and proposal['workers']==proposal['torch_threads']==1,'proposal outside CPU/features one-step scope')
    require(job['status']=='verified_all_four_original_profiles' and job['original_inputs_unchanged'] is True and len(job['cases'])==4,'complete fresh original-profile audits required')
    require(job['new_training_records']==job['new_scored_evaluation_games']==job['new_accelerator_jobs']==0 and job['execution_authorized'] is False,'audit scope differs')
    require(digest(allocation_path)==job['allocation_sha256'] and digest(proposal_path)==allocation['proposal_sha256'] and (repo/allocation['proposal']).resolve()==proposal_path,'proposal/allocation changed')
    require(job['auditor_sha256']==allocation['auditor_sha256']==digest(repo/allocation['auditor_path']),'pinned independent auditor changed')
    require(len(job['commands'])==4 and {r['case'] for r in job['commands']}==set(allocation['cases']) and all(r['returncode']==0 for r in job['commands']),'incomplete original audit inventory')
    bridge_path=repo/proposal['source_bridge'];bridge=load(bridge_path)
    require(digest(bridge_path)==proposal['source_bridge_sha256'] and bridge['version']==task_source_bridge.VERSION and bridge['source_compatibility_qualified'] is True,'qualified original CPU/features bridge required')
    require(bridge['reference_source_sha256']==proposal['reference_source_sha256'] and bridge['candidate_source_sha256']==proposal['target_runtime_source_sha256'],'price source bridge differs')
    _fresh_bridge(repo,bridge)
    require(job['source_sha256']==allocation['learner_source_sha256']==proposal['reference_source_sha256'],'original profile source differs')
    inputs={str(proposal_path):digest(proposal_path),str(job_path):digest(job_path),str(allocation_path):digest(allocation_path),str(bridge_path):digest(bridge_path)}
    prices={};identities=[]
    with ExitStack() as stack:
        locks={repo/c['root']/'.trainer.lock' for c in allocation['cases'].values()}|{(repo/c['root']).parent/'.driver.lock' for c in allocation['cases'].values()}
        locks|={selected_root/'profiles/.driver.lock',*sorted((selected_root/'profiles').glob('neural-R*/.trainer.lock'))}
        for path in sorted(locks):fcntl.flock(stack.enter_context(path.open('rb')),fcntl.LOCK_SH|fcntl.LOCK_NB)
        original=job['original_inputs_sha256']
        require(inputs_match(original),'original profile/checkpoint/trajectory inputs changed')
        inputs.update(original)
        for term,reuse in [('C1',1),('C4',4)]:
            proposed=proposal['candidate_terms'][term];budget_path=repo/proposed['raw_budget'];budget=load(budget_path);root=budget_path.parent.parent
            require(digest(budget_path)==proposed['raw_budget_sha256'] and digest(root/'plan.json')==proposed['profile_plan_sha256']==budget['profile_plan_sha256'],'raw profile budget/plan changed')
            require(digest(root/'profile.json')==proposed['profile_results_sha256'] and load(root/'state.json')['status']=='complete','original profile results incomplete or changed')
            profile=load(root/'profile.json');require(profile['plan']==load(root/'plan.json') and profile['plan']['source_sha256']==proposal['reference_source_sha256'],'profile source/plan mismatch')
            rows=[r for r in profile['results'] if r['learner']=='neural' and r['q_reuse']==reuse];configs={}
            require(rows==budget['profile_cases'] and budget['source_sha256']==proposal['reference_source_sha256'] and budget['complete_pipeline'] is True,'raw budget differs from original complete learner rows')
            require(budget['device']=='cpu' and budget['storage']=='features' and budget['native_library_sha256'] is None and budget['target']=='one-step' and budget['q_reuse']==reuse and budget['margin']==2.,'budget outside selected price scope')
            for phase in ('pre','post'):
                label=term+'/'+phase;declared=allocation['cases'][label];case=proposed['original_profile_cases'][phase];folder=repo/declared['root'];manifest=load(folder/'run.json');result=load(folder/'result.json')
                require(folder.resolve()==(repo/case['original_root']).resolve() and declared['manifest_sha256']==case['original_manifest_sha256']==digest(folder/'run.json') and declared['result_sha256']==case['original_result_sha256']==digest(folder/'result.json'),'original profile identity changed')
                require(manifest['source_sha256']==result['source_sha256']==proposal['reference_source_sha256'] and manifest['config']==declared['config']==case['profile_config'],'profile source/config differs')
                configs[phase]=manifest['config'];identity={k:manifest[k] for k in IDENTITY_KEYS};require(identity==case['runtime_identity'],'profile runtime identity changed');identities.append(identity)
                output=Path(job['cases'][label]['output']);audit=load(output);inputs[str(output)]=digest(output)
                require(inputs[str(output)]==job['cases'][label]['output_sha256'] and audit['auditor_sha256']==job['auditor_sha256'] and audit['verified'] is True and not audit['formal_protocol'],'fresh diagnostic audit differs')
                require(audit['source_sha256']==manifest['source_sha256'] and audit['training_manifest_sha256']==digest(folder/'run.json') and audit['episode_audit_sha256']==digest(folder/'episodes.bin'),'audit source/trajectory binding differs')
                checkpoint=CheckpointStore(folder).latest();CheckpointStore.verify(checkpoint)
                require(audit['checkpoint_manifest_sha256']==digest(checkpoint/'manifest.json'),'audit latest checkpoint binding differs')
                require(audit['learner']=='neural' and audit['task']=='delayed' and audit['seed']==0 and audit['records']==result['records']==configs[phase]['records'],'audited case recipe/count differs')
                row=next(r for r in rows if r['phase']==phase)
                require(result['status']=='complete' and result['counters']==row['counters'] and result['acting_policy_counts']==row['acting_policy_counts'],'profile result work differs from pricing rows')
                require(abs(number(row['seconds_including_setup'])-number(result['elapsed_seconds'])-number(result['timings']['setup']))<Decimal('.00000001'),'profile timing omitted measured setup or loop')
            price=profile_price(reuse,rows,configs)
            require(number(budget['proposed_time_ceiling_seconds'])==number(price['cpu_process_ceiling_seconds']) and abs(number(budget['phase_weighted_15M_seconds'])-number(price['phase_weighted_15M_seconds']))<Decimal('.000001'),'original twofold budget arithmetic differs')
            for key,value in price.items():require(proposed[key]==value,'prospective price differs from independently checked originals: '+key)
            prices[term]=price|{'raw_budget_sha256':digest(budget_path),'forecast_limits':budget['limits']}
        prices=selected_profile_prices(repo,selected_root,proposal,prices,inputs,identities)
        require(all(i==identities[0] for i in identities) and runtime_identity()==identities[0],'current CPU/runtime/dependency identity differs from measured profiles')
        require(identities[0]['dependencies']==bridge['dependencies'],'source bridge numerical dependencies differ')
        require(inputs_match(inputs),'pricing inputs changed during qualification')
    return {'version':VERSION,'status':'qualified_prospective_cpu_delayed_prices','qualifier_sha256':digest(Path(__file__)),'reference_source_sha256':proposal['reference_source_sha256'],'target_runtime_source_sha256':proposal['target_runtime_source_sha256'],'runtime_identity':identities[0],'prices':prices,'original_inputs_sha256':inputs,'cpu_delayed_prices_independently_qualified':True,'direct_selected_package_profiles_verified':True,'current_inputs_verified':True,'current_runtime_identity_verified':True,'original_cases_rewritten':False,'cpu_control_ceiling_separate_seconds':43200,'new_training_selected':False,'a2_complete':False,'project_scope_selected':False,'execution_authorized':False,'qualification_seconds':time.monotonic()-started,'evidence':{'repo':str(repo),'proposal':str(proposal_path),'audit_root':str(audit_root),'selected_root':str(selected_root)},'scope':'Prospective CPU/features neural delayed one-step prices, directly profiled on the selected package with the larger original/current twofold CPU ceilings; exact original seeds/recipes/runtime/package only. Zero device-hours is conditional on CPU execution; CPU process ceilings and partial-buffer forecasting limits remain explicit. No control funding, convergence claim, existing-case relabeling, project scope or execution grant.'}


def read_qualified(path):
    stored=load(path);e=stored['evidence'];fresh=qualify(e['repo'],e['proposal'],e['audit_root'],e['selected_root']);left=dict(stored);right=dict(fresh)
    left.pop('qualification_seconds');right.pop('qualification_seconds');require(left==right,'fresh CPU-price qualification differs from stored receipt');return stored


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--repo',required=True);parser.add_argument('--proposal',required=True);parser.add_argument('--audits',required=True);parser.add_argument('--selected',required=True);parser.add_argument('--output',required=True);args=parser.parse_args(argv)
    if Path(args.output).exists():parser.error('qualification output must be a new file; existing inputs/evidence are immutable')
    result=qualify(args.repo,args.proposal,args.audits,args.selected);write_report(args.output,result);print(json.dumps({'status':result['status'],'prices':result['prices']},sort_keys=True))


if __name__=='__main__':main()

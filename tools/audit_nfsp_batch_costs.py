"""Independent arithmetic and retained-evidence audit for batched NFSP costs.

No producer is imported. Scalar-core equations/artifact checks come from the
independent original auditor; selected group/wave equations are rebuilt here.
"""
import argparse
from copy import deepcopy
from decimal import Decimal
import fcntl
import json
import math
from pathlib import Path
import time
from tools.audit_nfsp_cost import require,sha,load,positive,close,samples,recompute,verify_artifacts,production_shape

VERSION='graygo/batched-nfsp-cost-audit/1'
PRODUCER='graygo/batched-nfsp-cost-measurement/1'


def timing(data,*,max_pairs,limit):
    total=positive(data['seconds'],'source interval');groups=data['groups'];require(bool(groups),'no actual source groups')
    for g in groups:
        require(type(g['pairs']) is int and 1<=g['pairs']<=max_pairs and g['physical_games']==2*g['pairs'],'source pair/game count differs')
        waves=g['waves'];require(0<len(waves)<=limit,'source wave cap differs');active=g['physical_games'];rounds=0
        for wave in waves:
            require(type(wave['active_games']) is int and 0<wave['active_games']<=active,'source games grew or lost their complete-wave order')
            active=wave['active_games'];rounds+=active;positive(wave['seconds'],'wave')
        require(g['physical_rounds']==g['completed_rounds']==rounds,'source physical rounds differ')
        positive(g['setup_seconds'],'group setup',zero=True)
        close(g['seconds'],g['setup_seconds']+math.fsum(w['seconds'] for w in waves),'complete source group partition')
    for value in data['integrity_seconds']:positive(value,'integrity')
    positive(data['metadata_seconds'],'source metadata',zero=True)
    close(total,math.fsum(g['seconds'] for g in groups)+math.fsum(data['integrity_seconds'])+data['metadata_seconds'],'source metadata/model/group partition')
    return groups


def recalculate(report):
    c=report['core_report'];e=report['evaluation'];plan=report['plan'];runtime=plan['runtime'];limit=runtime['rules']['max_rounds']
    require(report['version']==plan['version']==PRODUCER and report['status']=='complete','unknown or incomplete batched profile')
    require(c['plan']['runtime']==runtime and c['plan']['source_sha256']==plan['source_sha256'],'core source/runtime differs')
    require(e['inference_batch_size']==plan['inference_batch_size'] and e['pair_batch_size']==plan['pair_batch_size'],'selected evaluator shape differs')
    require(e['formal_learning_run'] is False and e['strength_claim'] is False,'timing weights became learning claims')
    require(e['endpoint']['games']==2000 and e['endpoint']['looks']==[500,2000],'missing full two-look endpoint')
    require(e['science']['distinct_policy_objects']==12 and e['science']['collection']['committed_blocks']==2,'scientific registry/block coverage differs')
    pair_batch=plan['pair_batch_size'];require(type(pair_batch) is int and 2<=pair_batch<=16,'invalid selected concurrent pairs')
    base=[];seen=set()
    for case in e['cases']:
        token=(case['representation'],case['mode']);require(token not in seen,'duplicate timing case');seen.add(token)
        groups=timing(case['timing'],max_pairs=pair_batch,limit=limit)
        require(len(groups)==1 and groups[0]['pairs']==pair_batch and len(case['pairs'])==pair_batch,'case did not execute its full group')
        roots=case['root_rounds_played']
        expected=[0]*pair_batch if case['mode']=='empty' else [limit-1]*pair_batch
        if case['mode']=='mixed-tail':expected[0]=0
        require(roots==expected,'empty/dense/tail root stress changed')
        for pair,start in zip(case['pairs'],roots):
            require(len(pair['subject'])==2 and not pair['incumbent'],'case color pairing differs')
            for game in pair['subject']:
                require(0<game['rounds']<=limit-start,'case exceeded its complete root horizon')
                if game['termination']=='round_limit':require(game['rounds']==limit-start,'case round-limit ending differs from root countdown')
                if start==0 and game['termination']=='mutual_pass':require(game['rounds']>=2,'empty timing game ended before two mutual passes')
        require(sum(g['rounds'] for p in case['pairs'] for g in p['subject'])==groups[0]['physical_rounds'],'case summaries lost physical rounds')
        base+=groups
    require(seen=={(r,m) for r in ('q/random','q/tactical','average/random','average/tactical') for m in ('empty','dense','mixed-tail')},'baseline cost coverage incomplete')
    endpoint=timing(e['endpoint']['timing'],max_pairs=pair_batch,limit=limit)
    expected_groups=(500+2*pair_batch-1)//(2*pair_batch)+(1500+2*pair_batch-1)//(2*pair_batch)
    require(len(endpoint)==expected_groups and sum(g['physical_games'] for g in endpoint)==2000,'real endpoint group/look count differs')
    base+=endpoint;science=timing(e['science']['timing'],max_pairs=24,limit=limit)
    require(len(science)==2 and all(g['pairs']==24 for g in science),'science did not run two full 48-game blocks')
    require(e['physical_games']==sum(g['physical_games'] for g in base+science)
            and e['physical_rounds']==sum(g['physical_rounds'] for g in base+science),'collected evaluation actions/rounds differ')
    checks=[value for case in e['cases'] for value in case['timing']['integrity_seconds']]
    checks+=e['endpoint']['timing']['integrity_seconds']+e['science']['timing']['integrity_seconds']
    require(e['integrity_seconds']==checks,'integrity samples differ from disjoint measured source intervals')
    rates={'baseline_wave':max(w['seconds'] for g in base for w in g['waves']),
        'baseline_group_setup':max(g['setup_seconds'] for g in base),'endpoint_metadata':e['endpoint']['timing']['metadata_seconds'],
        'science_wave':max(w['seconds'] for g in science for w in g['waves']),
        'science_group_setup':max(g['setup_seconds'] for g in science),'science_block_metadata_bound':e['science']['timing']['metadata_seconds'],
        'policy_integrity':max(e['integrity_seconds']),'probe_forward':max(e['probe_forward_seconds'])}
    require(len(e['probe_forward_seconds'])==128,'all 64 probe roots and both roles required')
    require(set(rates)==set(e['rates']),'batched cost inventory differs')
    for key,value in rates.items():close(e['rates'][key],value,'rate '+key)
    cold=positive(report['cold_start']['seconds'],'cold process');loads=samples(report['selected_model_load_seconds'],3,'selected policy load')
    require(report['cold_start']['result']['source_sha256']==plan['source_sha256']
            and report['cold_start']['result']['device']==runtime['device'] and report['cold_start']['result']['cuda_initialized'] is True,'cold source/device initialization differs')
    margin=positive(plan['margin'],'margin');require(margin>=1,'runtime margin below one');f=report['forecast'];hours={}
    for stage,games in [('pilot',36000),('screening_cumulative',40000),('confirmation',44000)]:
        endpoints=games//2000;groups=endpoints*expected_groups;row=f['per_seed'][stage]
        require((row['evaluation_games'],row['evaluation_endpoints'],row['evaluation_groups'],row['model_loads_upper'],row['cold_process_starts_upper'])
                ==(games,endpoints,groups,2*endpoints,2*endpoints+1),'full stage evaluation/model/start counts differ')
        # These core components have already been independently recomputed.
        parts={k:v for k,v in c['forecast']['per_seed'][stage]['components_seconds'].items() if k not in ('evaluation','evaluation_policy_integrity')}
        parts.update(evaluation_groups=groups*(rates['baseline_group_setup']+limit*rates['baseline_wave']),
            evaluation_endpoint_metadata=endpoints*rates['endpoint_metadata'],evaluation_policy_integrity=2*games*rates['policy_integrity'],
            evaluation_model_loads=2*endpoints*loads,cold_process_startups=(2*endpoints+1)*cold)
        require(set(parts)==set(row['components_seconds']),'stage dropped/added a component')
        for key,value in parts.items():close(row['components_seconds'][key],value,'stage '+key)
        total=math.fsum(parts.values());hours[stage]=total*margin/3600
        for key,value in [('projected_seconds_before_margin',total),('ceiling_seconds',total*margin),('allocated_process_hours',hours[stage]),('allocated_single_device_hours',hours[stage])]:close(row[key],value,'stage '+key)
    parts={'scientific_blocks':2000*(rates['science_group_setup']+limit*rates['science_wave']+rates['science_block_metadata_bound']),
        'scientific_and_probe_integrity':192512*rates['policy_integrity'],'probe_forwards':768*rates['probe_forward'],
        'model_loads':60*loads,'cold_process_startups':4*cold}
    require(set(parts)==set(f['scientific_and_probe_components_seconds']),'scientific inventory differs')
    for key,value in parts.items():close(f['scientific_and_probe_components_seconds'][key],value,'science '+key)
    seconds=margin*math.fsum(parts.values());close(f['scientific_and_probe_seconds_with_margin'],seconds,'science total');close(f['scientific_and_probe_device_hours'],seconds/3600,'science hours')
    require((f['scientific_blocks'],f['scientific_games'],f['scientific_and_probe_model_loads_upper'],f['concurrency'],f['margin'])==(2000,96000,60,1,margin),'scientific counts or amortization changed')
    for term,stage in [('H_P','pilot'),('H_D','screening_cumulative'),('H_N','confirmation')]:close(f[term],hours[stage],term)
    return {'per_seed_device_hours':hours,'scientific_and_probe_device_hours':seconds/3600,'rates_seconds':rates,
            'cold_start_seconds':cold,'selected_model_load_seconds':loads,'groups_per_endpoint':expected_groups}


def audit(root,repo,allocation,producer,core_producer,*,expected_source,expected_producer,expected_allocation):
    from graygo.manifest import source_digest
    from graygo.evaluation_evidence import FrozenEvaluationEvidence
    from graygo.science_evaluation import ScientificEvidence
    from graygo.evaluation import FrozenNeuralPolicy,neural_inference_identity,BaselinePolicy,validate_pair
    from graygo.models import PolicyNet,ModelConfig
    import torch
    root=Path(root).resolve();repo=Path(repo).resolve();allocation=Path(allocation).resolve();producer=Path(producer).resolve();core_producer=Path(core_producer).resolve()
    require(source_digest()==expected_source and sha(allocation)==expected_allocation and sha(producer)==expected_producer,'caller-pinned source/producer/allocation differs')
    inventory={str(p):sha(p) for p in [allocation,producer,core_producer,root/'plan.json',root/'run.json',root/'report.json',root/'job.json',root/'driver.py']}
    a=load(allocation);job=load(root/'job.json');r=load(root/'report.json');plan=r['plan'];manifest=load(root/'run.json')
    require(job['status']=='complete_measurement_unreviewed' and job['returncode']==0 and job['original_inputs_unchanged'] is True,'original profile is not successful/terminal')
    require(job['report_sha256']==sha(root/'report.json') and job['driver_sha256']==sha(root/'driver.py') and job['tool_sha256']==expected_producer
            and job['allocation_sha256']==expected_allocation,'original process bindings differ')
    require(plan==load(root/'plan.json')==manifest['command'] and manifest['git_dirty'] is False
            and manifest['git_commit']==a['learner_source_commit'] and plan['source_sha256']==manifest['source_sha256']==job['learner_source_sha256']==a['learner_source_sha256']==expected_source,'clean source/plan provenance differs')
    require(sha(core_producer)==plan['core_producer_sha256']==a['core_producer_sha256'] and plan['tool_sha256']==expected_producer,'producer binding differs')
    for key in ('formal_learning_run','a2_complete','execution_authorized'):require(r[key] is False,'profile claims '+key)
    require(a['production_learning_selected'] is False and a['pilot_selected'] is False and a['confirmation_selected'] is False and job['production_training_selected'] is False,'timing allocation became training scope')
    for field in ('inference_batch_size','pair_batch_size'):require(plan[field]==a[field],'selected evaluator shape changed')
    require(plan['runtime']['q_reuse']==a['selected_reuse'] and plan['runtime']['device']==a['device'],'selected learner reuse/device changed')
    require(job['physical_device_uuid']==a['physical_device_uuid'] and r['elapsed_seconds']<=a['process_time_ceiling_seconds']==plan['time_ceiling_seconds'],'device/cap differs')
    span=Decimal(str(job['allocation_interval_end_epoch']))-Decimal(str(job['allocation_interval_start_epoch']))
    require(span>0 and abs(span-Decimal(str(job['measured_driver_seconds'])))<Decimal('.01') and job['measured_driver_seconds']<=a['outer_watchdog_seconds'],'invalid original allocation interval')
    for p,h in plan['input_files_sha256'].items():require(sha(p)==h,'profile input changed');inventory[p]=h
    catalog=repo/a['catalog'];require(sha(catalog)==a['catalog_file_sha256'],'frozen catalog differs')
    c=load(root/'core/report.json');require(c==r['core_report'] and sha(root/'core/report.json')==r['core_report_sha256'],'retained core report differs')
    require(c['plan']['tool_sha256']==plan['core_producer_sha256'] and c['plan']['source_sha256']==expected_source
            and c['plan']['target']==a['target']=='one-step','core source/producer/target differs')
    inventory[str(root/'core/report.json')]=sha(root/'core/report.json')
    for name in ('plan.json','run.json','pipeline/report.json'):
        p=root/'core'/name;inventory[str(p)]=sha(p)
    require(load(root/'core/plan.json')==c['plan'] and load(root/'core/run.json')['command']==c['plan'],'core plan/source manifest differs')
    pipeline=verify_artifacts(root/'core',repo,a,c,inventory);recompute(c,pipeline)
    require(production_shape(c) and c['candidate_price_forecast_available'] is True and r['candidate_price_forecast_available'] is True
            and r['fresh_learner_profile_records']==65792,'missing full learner/kernel/storage profile')
    require(r['evaluation_physical_games']==r['evaluation']['physical_games'] and r['evaluation_physical_rounds']==r['evaluation']['physical_rounds'],'evaluation physical usage differs')
    computed=recalculate(r)
    endpoint_root=root/'evaluation/endpoint';science_root=root/'evaluation/science'
    for directory in (endpoint_root,science_root):
        for p in directory.rglob('*'):
            if p.is_file():inventory[str(p)]=sha(p)
    endpoint=FrozenEvaluationEvidence.load(endpoint_root);science=ScientificEvidence(science_root);blocks=science.records()
    require(endpoint.available_games==2000 and endpoint.plan.looks==(500,2000) and endpoint.plan.pair_batch_size==plan['pair_batch_size'],'actual endpoint prefix differs')
    require(len(blocks)==2 and science.contract['pair_batching'] is True and science.protocol.purpose=='diagnostic','actual source scientific evidence differs')
    payload=torch.load(Path(c['io']['checkpoint'])/'cost-payload.pt',map_location='cpu',weights_only=True)
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):model=PolicyNet(ModelConfig(plan['runtime']['width'],plan['runtime']['blocks'],plan['runtime']['norm_groups']))
    model.load_state_dict(payload['average']);expected=deepcopy(FrozenNeuralPolicy(model).identity)
    if plan['inference_batch_size']>1:expected['inference']=neural_inference_identity(plan['inference_batch_size'],plan['runtime']['device'])
    require(endpoint.contract['subject']==expected and endpoint.contract['opponent']==BaselinePolicy('tactical').identity,'actual timing endpoint changed selected weights or opponent')
    require(all(binding['identity']==expected for binding in science.bindings.values()),'scientific timing registry changed selected weights')
    require(science.catalog.sha256==a['catalog_sha256'],'scientific source catalog differs')
    endpoint_rounds=sum(game['rounds'] for row in endpoint.rows for game in row['subject'])
    require(endpoint_rounds==sum(g['physical_rounds'] for g in r['evaluation']['endpoint']['timing']['groups']),'endpoint wave log differs from scored evidence')
    for case in r['evaluation']['cases']:
        for pair in case['pairs']:validate_pair(pair,endpoint.plan.rules,False)
    memory=r['memory'];require(0<memory['peak_cuda_allocated_bytes']<=memory['peak_cuda_reserved_bytes']<=a['reported_vram_mib']*1024**2,'memory exceeds declared GPU')
    require(source_digest()==expected_source and all(sha(p)==h for p,h in inventory.items()),'retained profile inputs changed during audit')
    violations=[]
    if computed['per_seed_device_hours']['pilot']>24:violations.append('H_P exceeds initial 24-device-hour per-seed pilot ceiling')
    if computed['scientific_and_probe_device_hours']>16:violations.append('scientific/probe forecast exceeds 16-device-hour component')
    return {'version':VERSION,'status':'verified_empirical_accounting','source_sha256':expected_source,'producer_sha256':expected_producer,
        'auditor_sha256':sha(__file__),'core_auditor_sha256':sha(Path(__file__).with_name('audit_nfsp_cost.py')),
        'input_files_sha256':inventory,'original_inputs_unchanged':True,'independent_calculation':computed,'component_limit_violations':violations,
        'original_driver_pid':job['original_driver_pid'],'original_profile_pid':job['profile_pid'],'physical_device_uuid':job['physical_device_uuid'],
        'physical_allocation_seconds':str(span),'physical_allocation_device_hours':str(span/3600),
        'empirical_component_forecast_verified':True,'selected_scope_qualified':False,'runtime_bound_proven':False,'a2_complete':False,'execution_authorized':False,
        'scope':'independently recomputed source-bound measured component forecast; no new execution/learning authority or whole-project consumption claim'}


def main():
    parser=argparse.ArgumentParser()
    for key in ('run-dir','repo','allocation','producer','core-producer','expected-source-sha','expected-producer-sha','expected-allocation-sha','output-dir'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();started=time.monotonic()
    try:
        root=Path(args.run_dir).resolve();out=Path(args.output_dir).resolve();require(not out.exists() and not out.is_relative_to(root),'new audit output outside original profile required')
        with (root/'.cost-profile.lock').open('rb') as lock:
            fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
            report=audit(root,args.repo,args.allocation,args.producer,args.core_producer,expected_source=args.expected_source_sha,
                         expected_producer=args.expected_producer_sha,expected_allocation=args.expected_allocation_sha)
            report['audit_process_seconds']=time.monotonic()-started;out.mkdir(parents=True,exist_ok=False)
            (out/'audit.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
        print(json.dumps({k:report[k] for k in ('status','independent_calculation','component_limit_violations','audit_process_seconds')},indent=2));return 0
    except (ValueError,RuntimeError,KeyError,TypeError,OSError) as exc:parser.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())

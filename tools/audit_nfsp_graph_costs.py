"""Read-only, independent cold/steady graph-cost equations and artifacts.

No producer is imported. Original core artifact/equation checks are reused;
selected graph event partitions and every new bounded work count are rebuilt.
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

VERSION='graygo/graph-nfsp-cost-audit/1'
PRODUCER='graygo/graph-nfsp-cost-measurement/1'


def span_seconds(events,span):
    require(isinstance(span,list) and len(span)==2 and all(type(x) is int for x in span)
            and 0<=span[0]<=span[1]<=len(events),'invalid graph event interval')
    return math.fsum(e['seconds'] for e in events[span[0]:span[1]])


def timing(data,events,*,max_pairs,limit,seen):
    groups=data['groups'];require(bool(groups),'no actual selected source groups')
    for group in groups:
        require(type(group['pairs']) is int and 0<group['pairs']<=max_pairs and group['physical_games']==2*group['pairs'],'group/game count differs')
        waves=group['waves'];require(0<len(waves)<=limit,'group wave cap differs');active=group['physical_games'];rounds=0
        for wave in waves:
            require(type(wave['active_games']) is int and 0<wave['active_games']<=active,'active game sequence grew')
            active=wave['active_games'];rounds+=active;positive(wave['seconds'],'steady source wave')
            span=wave['graph_event_span'];setup=span_seconds(events,span)
            for i in range(*span):require(i not in seen,'one cold setup event was subtracted twice');seen.add(i)
            close(wave['graph_setup_seconds'],setup,'wave cold setup')
            close(wave['raw_seconds'],wave['seconds']+setup,'raw/cold/steady wave')
        require(group['physical_rounds']==group['completed_rounds']==rounds,'physical game rounds differ')
        close(group['seconds'],group['setup_seconds']+math.fsum(w['raw_seconds'] for w in waves),'complete raw group partition')
    close(data['seconds'],data['metadata_seconds']+math.fsum(data['integrity_seconds'])+math.fsum(g['seconds'] for g in groups),'metadata/integrity/raw-group partition')
    return groups


def recalculate(r):
    require(r['version']==r['plan']['version']==PRODUCER and r['status']=='complete','unknown or incomplete graph profile')
    c=r['core_report'];plan=r['plan'];cfg=plan['runtime'];e=r['evaluation'];actors=r['actors'];io=r['graph_io'];setup=r['graph_setup'];events=setup['events']
    require(cfg==c['plan']['runtime'] and cfg['cuda_graphs'] is True and cfg['device'].split(':')[0]=='cuda','selected graph/core runtime differs')
    require(c['plan']['source_sha256']==plan['source_sha256'],'core package differs')
    require(bool(events) and e['version']=='graygo/batched-evaluation-cost-components/2','no selected graph measurement')
    for event in events:
        require(event['kind'] in ('capture','weight_pack') and event['model'] in ('QNet','PolicyNet')
                and type(event['batch']) is int and 1<=event['batch']<=32,'invalid source graph setup event')
        positive(event['seconds'],'cold graph event')
    cap=max(event['seconds'] for event in events if event['kind']=='capture');pack=max(event['seconds'] for event in events if event['kind']=='weight_pack')
    close(setup['capture_max_seconds'],cap,'largest source capture');close(setup['weight_pack_max_seconds'],pack,'largest source pack')
    for key,pool in setup['pools'].items():
        own=[event for event in events if str(event['pool'])==key]
        require(pool['eager_fallbacks']==0 and 0<pool['resident_templates']<=64,'unmeasured fallback or unbounded graph inventory')
        for kind,counter,seconds in [('capture','captures','capture_seconds'),('weight_pack','weight_packs','weight_pack_seconds')]:
            selected=[event for event in own if event['kind']==kind];require(len(selected)==pool[counter],'source setup counter differs')
            close(pool[seconds],math.fsum(event['seconds'] for event in selected),'source pool setup total')
    require(all(str(event['pool']) in setup['pools'] for event in events),'event lost its source pool')
    seen=set();net=[]
    require(set(actors['cases'])=={'shared_q','shared_average','distinct_mixed'},'selected actor cases missing')
    require(actors['requests_per_sample']==actors['distinct_mixed_groups']==2*cfg['envs'] and actors['snapshots']==cfg['envs'],'actor version/group count differs')
    for values in actors['cases'].values():
        require(len(values)>=4,'insufficient selected actor samples')
        for sample in values:
            span=sample['graph_event_span'];cold=span_seconds(events,span)
            for i in range(*span):require(i not in seen,'cold actor setup subtracted twice');seen.add(i)
            close(sample['graph_setup_seconds'],cold,'actor setup total');close(sample['raw_seconds'],sample['steady_seconds']+cold,'actor raw/cold/steady partition')
            net.append(positive(sample['steady_seconds'],'steady private actor'))
    actor_rate=max(net)/actors['requests_per_sample'];close(actors['steady_actor_seconds_per_record'],actor_rate,'steady actor rate')
    copies=samples(actors['snapshot_copy_seconds'],cfg['envs'],'selected snapshot copy');base=[];cases=set()
    for case in e['cases']:
        token=(case['representation'],case['mode']);require(token not in cases,'duplicate baseline timing case');cases.add(token)
        groups=timing(case['timing'],events,max_pairs=16,limit=200,seen=seen);require(len(groups)==1 and groups[0]['pairs']==16 and len(case['pairs'])==16,'full selected case group missing')
        starts=[0]*16 if case['mode']=='empty' else [199]*16
        if case['mode']=='mixed-tail':starts[0]=0
        require(case['root_rounds_played']==starts,'selected empty/dense/tail root countdown differs')
        for pair,start in zip(case['pairs'],starts):
            for game in pair['subject']:
                require(0<game['rounds']<=200-start,'selected case exceeded root horizon')
                if game['termination']=='round_limit':require(game['rounds']==200-start,'selected case terminal countdown differs')
        base+=groups
    require(cases=={(p,m) for p in ('q/random','q/tactical','average/random','average/tactical') for m in ('empty','dense','mixed-tail')},'baseline timing coverage differs')
    require(e['inference_batch_size']==plan['inference_batch_size']==32 and e['pair_batch_size']==plan['pair_batch_size']==16,'selected shape changed')
    require(e['endpoint']['games']==2000 and e['endpoint']['looks']==[500,2000],'complete two-look endpoint missing')
    endpoint=timing(e['endpoint']['timing'],events,max_pairs=16,limit=200,seen=seen)
    require(len(endpoint)==63 and sum(g['physical_games'] for g in endpoint)==2000,'endpoint group counts differ')
    require(len(e['endpoint']['collection'])==2,'two actual endpoint calls required')
    for row,games in zip(e['endpoint']['collection'],(500,2000)):
        require(row['status']=='complete' and row['physical_games']==games and row['committed_pairs']==games//2,'actual look prefix differs')
    base+=endpoint;science=timing(e['science']['timing'],events,max_pairs=24,limit=200,seen=seen)
    require(len(science)==2 and all(g['pairs']==24 for g in science) and e['science']['distinct_policy_objects']==12,'scientific registry/blocks differ')
    require(e['physical_games']==sum(g['physical_games'] for g in base+science) and e['physical_rounds']==sum(g['physical_rounds'] for g in base+science),'source evaluation work differs')
    checks=[x for case in e['cases'] for x in case['timing']['integrity_seconds']]+e['endpoint']['timing']['integrity_seconds']+e['science']['timing']['integrity_seconds']
    require(e['integrity_seconds']==checks and len(e['probe_forward_seconds'])==128,'source integrity/probe scope differs')
    rates={'baseline_wave':max(w['seconds'] for g in base for w in g['waves']),'baseline_group_setup':max(g['setup_seconds'] for g in base),
        'endpoint_metadata':e['endpoint']['timing']['metadata_seconds'],'science_wave':max(w['seconds'] for g in science for w in g['waves']),
        'science_group_setup':max(g['setup_seconds'] for g in science),'science_block_metadata_bound':e['science']['timing']['metadata_seconds'],
        'policy_integrity':max(checks),'probe_forward':max(e['probe_forward_seconds'])}
    require(set(rates)==set(e['rates']),'selected evaluation cost inventory differs')
    for key,value in rates.items():close(e['rates'][key],value,'selected rate '+key)
    cold=positive(r['cold_start']['seconds'],'cold process');load=samples(r['selected_model_load_seconds'],3,'selected graph model load')
    require(r['cold_start']['result']['cuda_initialized'] is True and r['cold_start']['result']['source_sha256']==plan['source_sha256']
            and r['cold_start']['result']['device']==cfg['device'],'cold child source/device differs')
    margin=positive(plan['margin'],'margin');require(margin>=1,'margin below one');f=r['forecast'];hours={}
    for stage,games,milestones in [('pilot',36000,7),('screening_cumulative',40000,8),('confirmation',44000,9)]:
        old=c['forecast']['per_seed'][stage];n=old['records'];endpoints=games//2000;groups=endpoints*63
        snapshots=(n+cfg['batch']-1)//cfg['batch']+1;captures=2*64+2*endpoints;packs=2*snapshots+2*16+2*endpoints
        row=f['per_seed'][stage]
        require((row['records'],row['evaluation_games'],row['evaluation_endpoints'],row['evaluation_groups'],row['snapshot_publication_upper'],row['graph_capture_upper'],row['weight_pack_upper'])
            ==(n,games,endpoints,groups,snapshots,captures,packs),'graph/source event upper counts differ')
        parts={k:v for k,v in old['components_seconds'].items() if k not in ('evaluation','evaluation_policy_integrity')}
        parts.update(private_actors=n*actor_rate,frozen_snapshots=snapshots*copies,
            checkpoints=old['full_checkpoint_count_upper']*io['full_checkpoint_write_seconds'],one_restore=io['full_checkpoint_restore_seconds'],
            public_archives=milestones*io['public_payload_seconds'],evaluation_groups=groups*(rates['baseline_group_setup']+200*rates['baseline_wave']),
            evaluation_endpoint_metadata=endpoints*rates['endpoint_metadata'],evaluation_policy_integrity=2*games*rates['policy_integrity'],
            evaluation_model_loads=2*endpoints*load,cold_process_startups=(2+2*endpoints)*cold,frozen_graph_captures=captures*cap,frozen_parameter_packs=packs*pack)
        require(set(parts)==set(row['components_seconds']),'selected stage dropped/added a component')
        for key,value in parts.items():close(row['components_seconds'][key],value,'selected stage '+key)
        total=math.fsum(parts.values());hours[stage]=margin*total/3600
        for key,value in [('projected_seconds_before_margin',total),('ceiling_seconds',margin*total),('allocated_process_hours',hours[stage]),('allocated_single_device_hours',hours[stage])]:close(row[key],value,'selected stage '+key)
    parts={'scientific_blocks':2000*(rates['science_group_setup']+200*rates['science_wave']+rates['science_block_metadata_bound']),
        'scientific_and_probe_integrity':192512*rates['policy_integrity'],'probe_forwards':768*rates['probe_forward'],
        'model_loads':60*load,'cold_process_startups':4*cold,'frozen_graph_captures':60*cap,'frozen_parameter_packs':60*pack}
    require(set(parts)==set(f['scientific_and_probe_components_seconds']),'scientific graph scope differs')
    for key,value in parts.items():close(f['scientific_and_probe_components_seconds'][key],value,'selected science '+key)
    seconds=margin*math.fsum(parts.values());close(f['scientific_and_probe_seconds_with_margin'],seconds,'selected science total');close(f['scientific_and_probe_device_hours'],seconds/3600,'selected science hours')
    for term,stage in [('H_P','pilot'),('H_D','screening_cumulative'),('H_N','confirmation')]:close(f[term],hours[stage],term)
    require((f['scientific_blocks'],f['scientific_games'],f['concurrency'],f['margin'])==(2000,96000,1,margin),'study counts or amortization changed')
    return {'per_seed_device_hours':hours,'scientific_and_probe_device_hours':seconds/3600,'steady_actor_seconds_per_record':actor_rate,
        'capture_max_seconds':cap,'weight_pack_max_seconds':pack,'rates_seconds':rates,'cold_start_seconds':cold,'model_load_seconds':load}


def audit(root,repo,allocation,producer,core_producer,components,*,expected_source,expected_producer,expected_allocation):
    from graygo.manifest import source_digest
    from graygo.frozen_forward import execution_identity
    from graygo.evaluation import FrozenNeuralPolicy,neural_inference_identity,BaselinePolicy,validate_pair
    from graygo.evaluation_evidence import FrozenEvaluationEvidence
    from graygo.science_evaluation import ScientificEvidence
    from graygo.models import PolicyNet
    import torch
    root=Path(root).resolve();repo=Path(repo).resolve();allocation=Path(allocation).resolve();producer=Path(producer).resolve();core_producer=Path(core_producer).resolve();components=Path(components).resolve()
    require(source_digest()==expected_source and sha(allocation)==expected_allocation and sha(producer)==expected_producer,'caller-pinned graph source/tool/allocation differs')
    inventory={str(p):sha(p) for p in (allocation,producer,core_producer,components,root/'plan.json',root/'run.json',root/'report.json',root/'job.json',root/'driver.py')}
    a=load(allocation);j=load(root/'job.json');r=load(root/'report.json');plan=r['plan'];manifest=load(root/'run.json')
    require(j['status']=='complete_measurement_unreviewed' and j['returncode']==0 and j['original_inputs_unchanged'] is True,'original graph measurement is not successful and terminal')
    require(j['report_sha256']==sha(root/'report.json') and j['driver_sha256']==sha(root/'driver.py') and j['tool_sha256']==plan['tool_sha256']==expected_producer and j['allocation_sha256']==expected_allocation,'original graph process binding differs')
    require(plan==load(root/'plan.json')==manifest['command'] and manifest['git_dirty'] is False and manifest['git_commit']==a['learner_source_commit']
        and plan['source_sha256']==manifest['source_sha256']==j['learner_source_sha256']==a['learner_source_sha256']==expected_source,'clean selected package/plan differs')
    require(sha(core_producer)==plan['core_producer_sha256']==a['core_producer_sha256'] and sha(components)==plan['components_sha256']==a['components_sha256'],'source component helper differs')
    require(a['production_learning_selected'] is False and a['pilot_selected'] is False and a['confirmation_selected'] is False and j['production_training_selected'] is False,'measurement became a training grant')
    require(plan['runtime']['cuda_graphs'] is True and plan['runtime']['q_reuse']==a['selected_reuse'] and plan['inference_batch_size']==a['inference_batch_size']==32
        and plan['pair_batch_size']==a['pair_batch_size']==16,'selected graph workload changed')
    for key in ('formal_learning_run','a2_complete','execution_authorized'):require(r[key] is False,'graph timing became '+key)
    require(j['physical_device_uuid']==a['physical_device_uuid'] and r['elapsed_seconds']<=plan['time_ceiling_seconds']==a['process_time_ceiling_seconds'],'device or time cap differs')
    span=Decimal(str(j['allocation_interval_end_epoch']))-Decimal(str(j['allocation_interval_start_epoch']))
    require(span>0 and abs(span-Decimal(str(j['measured_driver_seconds'])))<Decimal('.01') and j['measured_driver_seconds']<=a['outer_watchdog_seconds'],'invalid graph allocation interval')
    for p,h in plan['input_files_sha256'].items():require(sha(p)==h,'selected graph input changed');inventory[p]=h
    require(sha(repo/a['catalog'])==a['catalog_file_sha256'],'frozen exposure catalog differs')
    c=load(root/'core/report.json');require(c==r['core_report'] and sha(root/'core/report.json')==r['core_report_sha256'] and c['plan']['runtime']==plan['runtime'],'retained selected pipeline core differs')
    for name in ('report.json','plan.json','run.json','pipeline/report.json'):
        p=root/'core'/name;inventory[str(p)]=sha(p)
    require(c['plan']['tool_sha256']==plan['core_producer_sha256'] and c['plan']['source_sha256']==expected_source and c['plan']['target']==a['target']=='one-step','selected core implementation differs')
    pipeline=verify_artifacts(root/'core',repo,a,c,inventory);recompute(c,pipeline)
    require(production_shape(c) and r['fresh_learner_profile_records']==65792 and r['candidate_price_forecast_available'] is True,'incomplete selected core measurements')
    selected=deepcopy(c);selected['io']=r['graph_io'];io_path=root/'graph-io/weights/000000500256'
    verify_artifacts(root/'core',repo,a,selected,inventory,selected_io_path=io_path)
    close(r['graph_io']['full_checkpoint_restore_seconds'],r['graph_io']['tensor_decode_and_backend_restore_seconds']+
          r['graph_io']['frozen_snapshot_reconstruction_seconds'],'complete decoded/snapshot restore composition')
    payload=torch.load(io_path/'cost-payload.pt',map_location='cpu',weights_only=True)
    require(all(v['frozen_execution']==execution_identity(True) for v in payload['frozen_snapshot_stress'].values()),'full selected snapshot execution metadata missing')
    computed=recalculate(r)
    for directory in (root/'evaluation/endpoint',root/'evaluation/science'):
        for p in directory.rglob('*'):
            if p.is_file():inventory[str(p)]=sha(p)
    endpoint=FrozenEvaluationEvidence.load(root/'evaluation/endpoint');science=ScientificEvidence(root/'evaluation/science');blocks=science.records()
    require(endpoint.available_games==2000 and endpoint.plan.looks==(500,2000) and endpoint.plan.pair_batch_size==16 and len(blocks)==2 and science.contract['pair_batching'] is True,'actual selected score prefixes differ')
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):model=PolicyNet()
    model.load_state_dict(payload['average']);expected=deepcopy(FrozenNeuralPolicy(model).identity);expected['inference']=neural_inference_identity(32,'cuda');expected['frozen_execution']=execution_identity(True)
    require(endpoint.contract['subject']==expected and endpoint.contract['opponent']==BaselinePolicy('tactical').identity and all(v['identity']==expected for v in science.bindings.values()),'selected graph weight/execution identity differs')
    require(science.catalog.sha256==a['catalog_sha256'] and science.protocol.purpose=='diagnostic','source scientific scope changed')
    require(sum(game['rounds'] for row in endpoint.rows for game in row['subject'])==sum(g['physical_rounds'] for g in r['evaluation']['endpoint']['timing']['groups']),'actual endpoint rounds differ')
    for case in r['evaluation']['cases']:
        for pair in case['pairs']:validate_pair(pair,endpoint.plan.rules,False)
    require(r['evaluation_physical_games']==r['evaluation']['physical_games'] and r['evaluation_physical_rounds']==r['evaluation']['physical_rounds'],'reported physical work differs')
    memory=r['memory'];require(0<memory['peak_cuda_allocated_bytes']<=memory['peak_cuda_reserved_bytes']<=a['reported_vram_mib']*1024**2,'selected graph memory exceeds device')
    require(source_digest()==expected_source and all(sha(p)==h for p,h in inventory.items()),'selected graph inputs changed during audit')
    violations=[]
    if computed['per_seed_device_hours']['pilot']>24:violations.append('H_P exceeds initial 24-device-hour per-seed pilot ceiling')
    if computed['scientific_and_probe_device_hours']>16:violations.append('scientific/probe forecast exceeds 16-device-hour component')
    return {'version':VERSION,'status':'verified_empirical_accounting','source_sha256':expected_source,'producer_sha256':expected_producer,
        'auditor_sha256':sha(__file__),'core_auditor_sha256':sha(Path(__file__).with_name('audit_nfsp_cost.py')),'input_files_sha256':inventory,
        'original_inputs_unchanged':True,'independent_calculation':computed,'component_limit_violations':violations,
        'original_driver_pid':j['original_driver_pid'],'original_profile_pid':j['profile_pid'],'physical_device_uuid':j['physical_device_uuid'],
        'physical_allocation_seconds':str(span),'physical_allocation_device_hours':str(span/3600),
        'empirical_component_forecast_verified':True,'selected_scope_qualified':False,'runtime_bound_proven':False,'a2_complete':False,'execution_authorized':False,
        'scope':'source-bound complete measured components and explicit cold work; no project spending/learning grant or runtime theorem'}


def main():
    parser=argparse.ArgumentParser()
    for key in ('run-dir','repo','allocation','producer','core-producer','components','expected-source-sha','expected-producer-sha','expected-allocation-sha','output-dir'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();started=time.monotonic()
    try:
        root=Path(args.run_dir).resolve();out=Path(args.output_dir).resolve();require(not out.exists() and not out.is_relative_to(root),'new independent output outside original graph profile required')
        with (root/'.cost-profile.lock').open('rb') as lock:
            fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
            r=audit(root,args.repo,args.allocation,args.producer,args.core_producer,args.components,expected_source=args.expected_source_sha,
                    expected_producer=args.expected_producer_sha,expected_allocation=args.expected_allocation_sha)
            r['audit_process_seconds']=time.monotonic()-started;out.mkdir(parents=True,exist_ok=False);(out/'audit.json').write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n')
        print(json.dumps({k:r[k] for k in ('status','independent_calculation','component_limit_violations','audit_process_seconds')},indent=2));return 0
    except (ValueError,RuntimeError,KeyError,TypeError,OSError) as exc:parser.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())

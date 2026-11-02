"""Complete NFSP pricing with measured fixed-shape evaluation group costs.

The unchanged scalar producer measures the real learner/storage components.
This module measures the selected evaluator and composes a separate forecast;
it never changes a learner counter, substitutes an average round for a group,
or turns diagnostic weights/evaluation into learning or scope evidence.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import torch

from graygo.board_profile import _synchronize
from graygo.confidence import ConfidenceFamily,canonical_sha
from graygo.curriculum import ExposureCatalog
from graygo.engine import State
from graygo.evaluation import BaselinePolicy,FrozenNeuralPolicy,EvaluationPlan,EvaluationRun,seed_for
from graygo.evaluation_batch import PairRequest
from graygo.manifest import make_manifest,source_digest,write_report
from graygo.models import QNet,PolicyNet
from graygo.science_evaluation import ScientificRun
from graygo.science_protocol import ScienceProtocol,policy_keys
from graygo.storage import DEFAULT_RESERVE,digest
from tools import nfsp_cost_profile as core

VERSION='graygo/batched-nfsp-cost-measurement/1'
CORE_SHA='e8c9f6721242b0ebc880a6b7207f0faa4b7582c05149632e08658789c920b73a'


def require(condition,message):
    if not condition:raise ValueError(message)


class TimingStarts:
    """Explicit timing cases, rejected by formal read-only opening loaders."""
    def __init__(self,seed,roots):
        self.mapping={seed_for(seed_for(seed,'unit',i),'opening'):root for i,root in enumerate(roots)}
        self.identity={'kind':'cost-only-complete-public-timing-roots','roots':[x.to_dict() for x in roots]}
        self.sha256=canonical_sha(self.identity)
    def sample(self,seed,rules):
        result=self.mapping[seed]
        require(result.rules==rules,'timing root rules differ');return result


class Observer:
    """Time source groups, complete waves and real immutable-model checks."""
    def __init__(self,device,check_budget=lambda:None):
        self.device=device;self.check_budget=check_budget;self.groups=[];self.integrity=[];self.assertions=0
    @contextmanager
    def installed(self):
        import graygo.evaluation_batch as module
        original_group=module.evaluate_pairs_batched;original_commit=module.commit_evaluation_actions
        original_frozen=FrozenNeuralPolicy.assert_frozen
        current=[None]
        def commit(requests):
            self.check_budget();_synchronize(self.device);now=time.perf_counter();group=current[0]
            require(group is not None,'wave escaped source group')
            if group['waves']:group['waves'][-1]['seconds']=now-group['waves'][-1]['started']
            group['waves'].append({'started':now,'active_games':len(requests)//2})
            return original_commit(requests)
        def group(requests):
            self.check_budget();_synchronize(self.device);started=time.perf_counter()
            entry={'pairs':len(requests),'physical_games':sum(4 if r.incumbent is not None else 2 for r in requests),'waves':[]}
            current[0]=entry
            try:result=original_group(requests)
            finally:current[0]=None
            _synchronize(self.device);ended=time.perf_counter();require(bool(entry['waves']),'source group produced no waves')
            entry['waves'][-1]['seconds']=ended-entry['waves'][-1]['started']
            entry['setup_seconds']=entry['waves'][0]['started']-started;entry['seconds']=ended-started
            for wave in entry['waves']:wave.pop('started')
            entry['physical_rounds']=sum(wave['active_games'] for wave in entry['waves'])
            entry['completed_rounds']=sum(game['rounds'] for pair in result for name in ('subject','incumbent') for game in pair[name])
            require(entry['physical_rounds']==entry['completed_rounds'],'source wave/game round accounting differs')
            self.groups.append(entry);return result
        def frozen(policy):
            self.check_budget();_synchronize(self.device);started=time.perf_counter();result=original_frozen(policy);_synchronize(self.device)
            self.integrity.append(time.perf_counter()-started);self.assertions+=1;return result
        module.evaluate_pairs_batched=group;module.commit_evaluation_actions=commit;FrozenNeuralPolicy.assert_frozen=frozen
        try:yield self
        finally:
            module.evaluate_pairs_batched=original_group;module.commit_evaluation_actions=original_commit;FrozenNeuralPolicy.assert_frozen=original_frozen


def run_timed(observer,function):
    group_at=len(observer.groups);integrity_at=len(observer.integrity)
    _synchronize(observer.device);started=time.perf_counter();result=function();_synchronize(observer.device);elapsed=time.perf_counter()-started
    groups=observer.groups[group_at:];checks=observer.integrity[integrity_at:]
    metadata=elapsed-sum(x['seconds'] for x in groups)-sum(checks)
    require(metadata>=-.001,'nested timing scopes exceed original source call')
    return result,{'seconds':elapsed,'groups':groups,'integrity_seconds':checks,'metadata_seconds':max(0.,metadata)}


def cold_start(device,expected_source,timeout):
    code="""import json,sys,torch
from graygo.board_runtime import configure_runtime
from graygo.manifest import source_digest
configure_runtime(sys.argv[1],True);torch.set_num_threads(1)
if sys.argv[1].startswith('cuda'):torch.cuda.init();torch.cuda.synchronize()
print(json.dumps({'source_sha256':source_digest(),'device':sys.argv[1],'cuda_initialized':torch.cuda.is_initialized()}))
"""
    started=time.perf_counter();child=subprocess.run([sys.executable,'-c',code,str(device)],capture_output=True,text=True,timeout=timeout)
    require(child.returncode==0,'cold interpreter/context probe failed: '+child.stderr[-2000:]);record=json.loads(child.stdout)
    require(record['source_sha256']==expected_source,'cold child package source differs')
    return {'seconds':time.perf_counter()-started,'command':[sys.executable,'-c','<retained source in producer>',str(device)],'result':record}


def evaluation_components(config,q_model,average_model,states,catalog,root,*,inference_batch_size,pair_batch_size,
                          full_look_games=2000,check_budget=lambda:None,reserve_bytes=DEFAULT_RESERVE):
    require(pair_batch_size>=2,'cost observer needs at least two pairs per source group')
    require(full_look_games>=16 and full_look_games%4==0,'complete two-look timing endpoint required')
    device=torch.device(config.device);root=Path(root);root.mkdir(parents=True,exist_ok=False)
    q=FrozenNeuralPolicy(q_model,representation='greedy_q',inference_batch_size=inference_batch_size)
    average=FrozenNeuralPolicy(average_model,inference_batch_size=inference_batch_size)
    observer=Observer(device,check_budget);cases=[]
    with observer.installed():
        import graygo.evaluation_batch as module
        for label,subject,opponent in [('q/random',q,BaselinePolicy('random')),('q/tactical',q,BaselinePolicy('tactical')),
                                      ('average/random',average,BaselinePolicy('random')),('average/tactical',average,BaselinePolicy('tactical'))]:
            for mode in ('empty','dense','mixed-tail'):
                roots=[State(config.rules) if mode=='empty' or (mode=='mixed-tail' and i==0) else states[i%len(states)] for i in range(pair_batch_size)]
                requests=[PairRequest(subject,opponent,state,seed_for(config.seed,'cost-group',label,mode,i)) for i,state in enumerate(roots)]
                results,timing=run_timed(observer,lambda:module.evaluate_pairs_batched(requests))
                cases.append({'representation':label,'mode':mode,'root_rounds_played':[x.rounds_played for x in roots],
                              'timing':timing,'pairs':results})
        # Run a real two-look endpoint through SQLite, score checks and bounds.
        looks=(full_look_games//4,full_look_games);family=ConfidenceFamily('cost-only-full-endpoint',('utility',),looks)
        plan=EvaluationPlan('cost-only-full-endpoint','benchmark-weights','utility',config.rules,config.seed,looks,family.sha256,pair_batch_size=pair_batch_size)
        def endpoint():
            run=EvaluationRun(plan,root/'endpoint',average,BaselinePolicy('tactical'),reserve_bytes=reserve_bytes)
            results=[]
            for games in looks:
                check_budget();results.append(run.run(games,time_cap_seconds=3600));run.estimate(family,games)
            return results
        collected,endpoint_timing=run_timed(observer,endpoint)
        require(all(x['status']=='complete' for x in collected),'incomplete real timing endpoint')
        policies={key:FrozenNeuralPolicy(average_model,inference_batch_size=inference_batch_size) for key in policy_keys()}
        storage={next(p._model.parameters()).untyped_storage().data_ptr() for p in policies.values()}
        require(len(storage)==12,'scientific timing registry must retain twelve distinct model objects')
        protocol=ScienceProtocol('cost-only-science',catalog.sha256,{key:p.sha256 for key,p in policies.items()},3600,
                                 seed=config.seed,block_looks=(2,4),purpose='diagnostic')
        def science():
            run=ScientificRun(protocol,root/'science',policies,catalog,reserve_bytes=reserve_bytes,pair_batching=True)
            result=run.run(96);run.estimate(96);require(result['status']=='complete','incomplete real scientific timing blocks')
            return result
        scientific,science_timing=run_timed(observer,science)
        probes=[];probability_hash=hashlib.sha256()
        for state in states:
            for role in (1,2):
                check_budget();_synchronize(device);started=time.perf_counter();probabilities=average.initial_distribution(state,role);_synchronize(device)
                probes.append(time.perf_counter()-started);probability_hash.update(probabilities.tobytes())
    all_base=[g for case in cases for g in case['timing']['groups']]+endpoint_timing['groups']
    return {'version':'graygo/batched-evaluation-cost-components/1','inference_batch_size':inference_batch_size,'pair_batch_size':pair_batch_size,
        'cases':cases,'endpoint':{'games':full_look_games,'looks':list(looks),'collection':collected,'timing':endpoint_timing},
        'science':{'collection':scientific,'timing':science_timing,'distinct_policy_objects':len(storage)},
        'probe_forward_seconds':probes,'probe_probability_sha256':probability_hash.hexdigest(),'integrity_seconds':observer.integrity,
        'rates':{'baseline_wave':max(w['seconds'] for g in all_base for w in g['waves']),
                 'baseline_group_setup':max(g['setup_seconds'] for g in all_base),'endpoint_metadata':endpoint_timing['metadata_seconds'],
                 'science_wave':max(w['seconds'] for g in science_timing['groups'] for w in g['waves']),
                 'science_group_setup':max(g['setup_seconds'] for g in science_timing['groups']),
                 'science_block_metadata_bound':science_timing['metadata_seconds'],'policy_integrity':max(observer.integrity),
                 'probe_forward':max(probes)},
        'physical_games':sum(g['physical_games'] for g in all_base)+96,
        'physical_rounds':sum(g['physical_rounds'] for g in all_base+science_timing['groups']),
        'formal_learning_run':False,'strength_claim':False,'scope':'actual source game groups, two-look evidence IO, twelve-model scientific blocks and selected-policy probe calls; no optimizer/fresh clock updates'}


def forecast(core_report,evaluation,*,margin,cold_seconds,model_load_seconds):
    require(math.isfinite(margin) and margin>=1,'stated finite runtime margin required')
    rates=evaluation['rates'];pair_batch=evaluation['pair_batch_size'];limit=core_report['plan']['runtime']['rules']['max_rounds']
    require(evaluation['endpoint']['games']==2000 and evaluation['endpoint']['looks']==[500,2000],'full registered endpoint timing required')
    groups_per_endpoint=math.ceil(500/(2*pair_batch))+math.ceil(1500/(2*pair_batch))
    result={};cuda=core_report['plan']['runtime']['device'].split(':')[0]=='cuda'
    for name,n_games in [('pilot',36000),('screening_cumulative',40000),('confirmation',44000)]:
        row=core_report['forecast']['per_seed'][name];endpoints=n_games//2000;groups=endpoints*groups_per_endpoint
        parts={k:v for k,v in row['components_seconds'].items() if k not in ('evaluation','evaluation_policy_integrity')}
        parts.update(evaluation_groups=groups*(rates['baseline_group_setup']+limit*rates['baseline_wave']),
                     evaluation_endpoint_metadata=endpoints*rates['endpoint_metadata'],evaluation_policy_integrity=2*n_games*rates['policy_integrity'],
                     evaluation_model_loads=2*endpoints*model_load_seconds,cold_process_startups=(1+2*endpoints)*cold_seconds)
        total=math.fsum(parts.values())
        result[name]={'records':row['records'],'evaluation_games':n_games,'evaluation_endpoints':endpoints,'evaluation_groups':groups,
                     'model_loads_upper':2*endpoints,'cold_process_starts_upper':1+2*endpoints,'components_seconds':parts,
                     'projected_seconds_before_margin':total,'ceiling_seconds':margin*total,
                     'allocated_process_hours':margin*total/3600,'allocated_single_device_hours':margin*total/3600 if cuda else None}
    science_parts={'scientific_blocks':2000*(rates['science_group_setup']+limit*rates['science_wave']+rates['science_block_metadata_bound']),
                   'scientific_and_probe_integrity':192512*rates['policy_integrity'],'probe_forwards':768*rates['probe_forward'],
                   'model_loads':60*model_load_seconds,'cold_process_startups':4*cold_seconds}
    science=margin*math.fsum(science_parts.values())
    return {'per_seed':result,'H_P':result['pilot']['allocated_single_device_hours'],'H_D':result['screening_cumulative']['allocated_single_device_hours'],
        'H_N':result['confirmation']['allocated_single_device_hours'],'scientific_and_probe_components_seconds':science_parts,
        'scientific_and_probe_seconds_with_margin':science,'scientific_and_probe_device_hours':science/3600 if cuda else None,
        'scientific_blocks':2000,'scientific_games':96000,'scientific_and_probe_model_loads_upper':60,
        'margin':margin,'concurrency':1,'scope':'empirical complete-group/wave forecast; full-stage endpoint counts, model loads and cold starts; no runtime theorem or scope grant'}


def main():
    parser=argparse.ArgumentParser()
    for name in ('run-dir','buffer-checkpoint','benchmark','catalog','native-library','expected-source-sha'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--reuse',type=int,choices=(1,4),required=True);parser.add_argument('--device',default='cuda')
    parser.add_argument('--time-cap-seconds',type=int,required=True);parser.add_argument('--inference-batch-size',type=int,required=True)
    parser.add_argument('--pair-batch-size',type=int,required=True);args=parser.parse_args()
    import fcntl,os,graygo
    from graygo.board_runtime import BoardRuntimeConfig,configure_runtime
    from graygo.probe_benchmark import ProbeBenchmark
    root=Path(args.run_dir).resolve();paths={k:Path(getattr(args,k)).resolve() for k in ('buffer_checkpoint','benchmark','catalog','native_library')}
    require(0<args.time_cap_seconds<=3600 and 2<=args.pair_batch_size<=16,'bounded full profile allocation required')
    require(source_digest()==args.expected_source_sha and digest(Path(core.__file__))==CORE_SHA,'exact selected package and unchanged core producer required')
    require(1<=args.inference_batch_size<=64,'declared padded inference shape required')
    os.chdir(Path(graygo.__file__).resolve().parents[2]);root.mkdir(parents=True,exist_ok=True)
    require(not (root/'plan.json').exists(),'profile exists; inspect original process/output')
    started=time.monotonic()
    def check_budget():
        if time.monotonic()-started>=args.time_cap_seconds:raise TimeoutError('original batched profile ceiling exhausted')
    with (root/'.cost-profile.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=BoardRuntimeConfig(q_reuse=args.reuse,device=args.device,native_library=str(paths['native_library']))
        configure_runtime(config.device,True);torch.set_num_threads(1)
        benchmark=ProbeBenchmark.from_dict(json.loads(paths['benchmark'].read_text()));catalog=ExposureCatalog.from_dict(json.loads(paths['catalog'].read_text()))
        plan={'version':VERSION,'source_sha256':source_digest(),'tool_sha256':digest(Path(__file__)),'core_producer_sha256':CORE_SHA,'runtime':config.plan(),
              'input_files_sha256':{str(p):digest(p) for p in [paths['buffer_checkpoint']/'manifest.json',paths['benchmark'],paths['catalog'],paths['native_library']]},
              'inference_batch_size':args.inference_batch_size,'pair_batch_size':args.pair_batch_size,'time_ceiling_seconds':args.time_cap_seconds,
              'margin':2.,'core_records_per_phase':32768,'dense_records':256,'endpoint_looks':[500,2000],'scientific_timing_blocks':2,
              'scope':'complete selected-source hardware costs only; no A2, pilot, confirmation or study allocation'}
        write_report(root/'plan.json',plan);write_report(root/'run.json',make_manifest(seed=config.seed,command=plan,rules=config.plan()['rules']))
        c=core.profile(config,paths['buffer_checkpoint'],benchmark,root/'core',records=32768,iterations=64,warmup=4,actor_iterations=4,dense_records=256,
                       margin=2.,time_cap_seconds=max(.001,args.time_cap_seconds-(time.monotonic()-started)))
        check_budget();payload=torch.load(Path(c['io']['checkpoint'])/'cost-payload.pt',map_location='cpu',weights_only=True)
        with torch.random.fork_rng(devices=[]),torch.device('cpu'):
            q=QNet(config.model_config());average=PolicyNet(config.model_config())
        q.load_state_dict(payload['q']);average.load_state_dict(payload['average']);q=q.to(config.device);average=average.to(config.device);del payload
        states=tuple(r.state for family in benchmark.families for r in (family.before,family.after))
        ev=evaluation_components(config,q,average,states,catalog,root/'evaluation',inference_batch_size=args.inference_batch_size,pair_batch_size=args.pair_batch_size,check_budget=check_budget)
        check_budget();cold=cold_start(config.device,source_digest(),min(60,max(1,args.time_cap_seconds-(time.monotonic()-started))))
        loads=[];public=Path(c['io']['checkpoint']).parents[1]/'cost-public-payload.pt'
        for _ in range(3):
            check_budget()
            def load_selected():
                digest(public);data=torch.load(public,map_location='cpu',weights_only=True)
                with torch.random.fork_rng(devices=[]),torch.device('cpu'):model=PolicyNet(config.model_config()).to(config.device)
                model.load_state_dict(data['average']);return FrozenNeuralPolicy(model,inference_batch_size=args.inference_batch_size)
            _,elapsed=core.timed(torch.device(config.device),load_selected);loads.append(elapsed)
        projected=forecast(c,ev,margin=2.,cold_seconds=cold['seconds'],model_load_seconds=max(loads));check_budget()
        require(source_digest()==plan['source_sha256'] and all(digest(Path(p))==s for p,s in plan['input_files_sha256'].items()),'profile source or inputs changed')
        result={'version':VERSION,'status':'complete','plan':plan,'core_report_sha256':digest(root/'core/report.json'),'core_report':c,
                'evaluation':ev,'cold_start':cold,'selected_model_load_seconds':loads,'forecast':projected,
                'elapsed_seconds':time.monotonic()-started,'memory':core._memory(torch.device(config.device)),
                'fresh_learner_profile_records':c['fresh_profile_records'],'evaluation_physical_games':ev['physical_games'],
                'evaluation_physical_rounds':ev['physical_rounds'],'candidate_price_forecast_available':bool(c['candidate_price_forecast_available'] and len(states)==64),
                'formal_learning_run':False,'a2_complete':False,'execution_authorized':False,'scope':'retained scalar core plus selected batched evaluator; counters and data scopes remain distinct'}
        write_report(root/'report.json',result)
        print(json.dumps({k:result[k] for k in ('status','forecast','elapsed_seconds','candidate_price_forecast_available')},indent=2,sort_keys=True))


if __name__=='__main__':main()

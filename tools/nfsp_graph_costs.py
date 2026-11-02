"""Complete graph-runtime costs with disjoint cold and steady work.

The source-pinned original core measures kernels, selected runtime pipeline and
reference eager actor/IO work. Actual graph actors and full graph-descriptor IO
replace those forecast components below. No cold setup is discarded.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import json
import math
import time
from weakref import WeakKeyDictionary
import numpy as np
import torch

from graygo.board_runtime import BoardRuntimeConfig,configure_runtime
from graygo.buffers import ReplayBuffer,ReservoirBuffer
from graygo.curriculum import ExposureCatalog
from graygo.evaluation import FrozenNeuralPolicy,seed_for
from graygo.frozen_forward import FrozenForwardPool,execution_identity
from graygo.manifest import source_digest,make_manifest,write_report
from graygo.models import QNet,PolicyNet
from graygo.nfsp import NFSPBackend
from graygo.nfsp_collection import FrozenPolicies,EpisodeActor,commit_many
from graygo.probe_benchmark import ProbeBenchmark
from graygo.rule_state import RuleObservation
from graygo.schedules import EpisodeChoice
from graygo.storage import digest
from tools import nfsp_cost_profile as core
from tools import nfsp_batch_costs as batch

VERSION='graygo/graph-nfsp-cost-measurement/1'
CORE_SHA='e8c9f6721242b0ebc880a6b7207f0faa4b7582c05149632e08658789c920b73a'


def require(condition,message):
    if not condition:raise ValueError(message)


class GraphSetupObserver:
    """Observe source metrics; never replace a model computation or counter."""
    def __init__(self):self.events=[];self.keys=WeakKeyDictionary();self.pools={};self.next_key=0
    def mark(self):return len(self.events)
    def seconds_between(self,first,last):return math.fsum(x['seconds'] for x in self.events[first:last])
    @contextmanager
    def installed(self):
        original=FrozenForwardPool.forward
        def observed(pool,model,spatial,globals,**kwargs):
            if pool not in self.keys:self.keys[pool]=self.next_key;self.next_key+=1
            key=self.keys[pool];before=dict(pool.metrics);result=original(pool,model,spatial,globals,**kwargs)
            for counter,seconds,kind in [('weight_packs','weight_pack_seconds','weight_pack'),('captures','capture_seconds','capture')]:
                count=pool.metrics[counter]-before[counter];require(count in (0,1),'source forward created multiple setup events')
                if count:self.events.append({'kind':kind,'seconds':pool.metrics[seconds]-before[seconds],
                    'pool':key,'model':type(model).__name__,'batch':int(spatial.shape[0])})
            self.pools[key]=pool.stats();return result
        FrozenForwardPool.forward=observed
        try:yield self
        finally:FrozenForwardPool.forward=original

    def report(self):
        return {'events':self.events,'pools':{str(k):v for k,v in self.pools.items()},
            'capture_max_seconds':max((x['seconds'] for x in self.events if x['kind']=='capture'),default=0.),
            'weight_pack_max_seconds':max((x['seconds'] for x in self.events if x['kind']=='weight_pack'),default=0.),
            'execution':execution_identity(True)}


def actor_measurements(config,backend,states,tracker,*,iterations=4,check_budget=lambda:None):
    pool=FrozenForwardPool(config.device);snapshots=[];copies=[]
    for i in range(config.envs):
        check_budget();snapshot,elapsed=core.timed(backend.device,lambda i=i:FrozenPolicies(backend.q,backend.average,
            version=1000000+i,average_ready=True,forward_pool=pool));snapshots.append(snapshot);copies.append(elapsed)
    cases={}
    for layout in ('shared_q','shared_average','distinct_mixed'):
        actors=[];samples=[]
        for env in range(config.envs):
            for role in (1,2):
                kind='q' if layout=='shared_q' or (layout=='distinct_mixed' and role==1) else 'average'
                snapshot=snapshots[env] if layout=='distinct_mixed' else snapshots[0]
                choice=EpisodeChoice(5000000,kind=='q',kind,.08875,1.,0. if kind=='q' else .5,.5)
                actors.append(EpisodeActor(snapshot,choice,np.random.default_rng(seed_for(config.seed,'cost-actor',env,role))))
        for _ in range(iterations):
            check_budget();first=tracker.mark()
            requests=[(actor,RuleObservation(states[i//2],i%2+1),actor.choice.best_response_mode) for i,actor in enumerate(actors)]
            values,raw=core.timed(backend.device,lambda:commit_many(requests));last=tracker.mark();setup=tracker.seconds_between(first,last)
            require(raw>=setup-.000001 and len(values)==2*config.envs,'actor setup/count partition differs')
            for i,value in enumerate(values):require(value.action in states[i//2].legal_actions(i%2+1),'graph actor chose illegal move')
            samples.append({'raw_seconds':raw,'steady_seconds':max(0.,raw-setup),'graph_setup_seconds':setup,'graph_event_span':[first,last]})
        cases[layout]=samples
    return {'cases':cases,'requests_per_sample':2*config.envs,'distinct_mixed_groups':2*config.envs,
        'snapshot_copy_seconds':copies,'snapshots':config.envs,'pool_stats':pool.stats(),
        'steady_actor_seconds_per_record':max(x['steady_seconds'] for values in cases.values() for x in values)/(2*config.envs),
        'scope':'actual selected graph commitments; raw intervals retained and source-reported one-time setup separated, not removed from forecasts'}


def graph_io(config,backend,replay,reservoir,root,*,reserve_bytes=core.DEFAULT_RESERVE):
    """Use original sealed IO with explicit selected snapshot/load factories."""
    original_snapshot=core.FrozenPolicies;original_policy=core.FrozenNeuralPolicy;pool=FrozenForwardPool(config.device)
    def snapshot(q,average,**kwargs):return FrozenPolicies(q,average,forward_pool=pool,**kwargs)
    def policy(model,**kwargs):return FrozenNeuralPolicy(model,cuda_graphs=True,**kwargs)
    core.FrozenPolicies=snapshot;core.FrozenNeuralPolicy=policy
    try:
        result=core.io_measurements(config,backend,replay,reservoir,root,reserve_bytes=reserve_bytes)
        payload=torch.load(Path(result['checkpoint'])/'cost-payload.pt',map_location='cpu',weights_only=True)
        def reconstruct():
            restored=[]
            for value in payload['frozen_snapshot_stress'].values():
                item=FrozenPolicies(backend.q,backend.average,version=value['version'],average_ready=value['average_ready'],forward_pool=pool)
                item.load_state_dict(value);restored.append(item)
            return restored
        _,seconds=core.timed(backend.device,reconstruct)
        result['tensor_decode_and_backend_restore_seconds']=result['full_checkpoint_restore_seconds']
        result['frozen_snapshot_reconstruction_seconds']=seconds
        result['full_checkpoint_restore_seconds']+=seconds
        result['restore_composition']='measured tensor/buffer/backend restore plus decoded frozen-module reconstruction; graph caches separately rebuilt'
        return result
    finally:core.FrozenPolicies=original_snapshot;core.FrozenNeuralPolicy=original_policy


def forecast(c,e,actors,io,setup,*,cold_seconds,model_load_seconds,margin=2.):
    require(e['endpoint']['games']==2000 and e['endpoint']['looks']==[500,2000],'complete fixed endpoint timing required')
    require(math.isfinite(margin) and margin>=1,'declared finite runtime margin required')
    cap=setup['capture_max_seconds'];pack=setup['weight_pack_max_seconds'];require(cap>0 and pack>0,'actual cold graph samples required')
    r=e['rates'];cfg=c['plan']['runtime'];limit=cfg['rules']['max_rounds'];batch_size=cfg['batch'];p=e['pair_batch_size']
    result={};cuda=cfg['device'].split(':')[0]=='cuda'
    for name,games,milestones in [('pilot',36000,7),('screening_cumulative',40000,8),('confirmation',44000,9)]:
        old=c['forecast']['per_seed'][name];n=old['records'];endpoints=games//2000
        groups=endpoints*(math.ceil(500/(2*p))+math.ceil(1500/(2*p)))
        snapshots=math.ceil(n/batch_size)+1;captures=128+2*endpoints;packs=2*snapshots+32+2*endpoints
        parts={key:value for key,value in old['components_seconds'].items() if key not in ('evaluation','evaluation_policy_integrity')}
        parts.update(private_actors=n*actors['steady_actor_seconds_per_record'],frozen_snapshots=snapshots*max(actors['snapshot_copy_seconds']),
            checkpoints=old['full_checkpoint_count_upper']*io['full_checkpoint_write_seconds'],one_restore=io['full_checkpoint_restore_seconds'],
            public_archives=milestones*io['public_payload_seconds'],evaluation_groups=groups*(r['baseline_group_setup']+limit*r['baseline_wave']),
            evaluation_endpoint_metadata=endpoints*r['endpoint_metadata'],evaluation_policy_integrity=2*games*r['policy_integrity'],
            evaluation_model_loads=2*endpoints*model_load_seconds,cold_process_startups=(2+2*endpoints)*cold_seconds,
            frozen_graph_captures=captures*cap,frozen_parameter_packs=packs*pack)
        total=math.fsum(parts.values());result[name]={'records':n,'evaluation_games':games,'evaluation_endpoints':endpoints,'evaluation_groups':groups,
            'snapshot_publication_upper':snapshots,'graph_capture_upper':captures,'weight_pack_upper':packs,
            'components_seconds':parts,'projected_seconds_before_margin':total,'ceiling_seconds':margin*total,
            'allocated_process_hours':margin*total/3600,'allocated_single_device_hours':margin*total/3600 if cuda else None}
    sci={'scientific_blocks':2000*(r['science_group_setup']+200*r['science_wave']+r['science_block_metadata_bound']),
         'scientific_and_probe_integrity':192512*r['policy_integrity'],'probe_forwards':768*r['probe_forward'],
         'model_loads':60*model_load_seconds,'cold_process_startups':4*cold_seconds,'frozen_graph_captures':60*cap,'frozen_parameter_packs':60*pack}
    seconds=margin*math.fsum(sci.values())
    return {'per_seed':result,'H_P':result['pilot']['allocated_single_device_hours'],'H_D':result['screening_cumulative']['allocated_single_device_hours'],
        'H_N':result['confirmation']['allocated_single_device_hours'],'scientific_and_probe_components_seconds':sci,
        'scientific_and_probe_seconds_with_margin':seconds,'scientific_and_probe_device_hours':seconds/3600 if cuda else None,
        'scientific_blocks':2000,'scientific_games':96000,'margin':margin,'concurrency':1,
        'graph_count_basis':'two full 64-template worker inventories (startup and one restore); two model packs per snapshot plus sixteen restored contexts; one shape per evaluation load',
        'scope':'source-bound empirical complete-component ceilings with cold work explicit; no runtime theorem, repeated-restart budget or execution grant'}


def main():
    parser=argparse.ArgumentParser()
    for key in ('run-dir','buffer-checkpoint','benchmark','catalog','native-library','expected-source-sha','expected-components-sha'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--reuse',type=int,choices=(1,4),required=True);parser.add_argument('--time-cap-seconds',type=int,required=True)
    args=parser.parse_args();import os,fcntl,graygo
    root=Path(args.run_dir).resolve();paths={key:Path(getattr(args,key)).resolve() for key in ('buffer_checkpoint','benchmark','catalog','native_library')}
    require(source_digest()==args.expected_source_sha and digest(core.__file__)==CORE_SHA and digest(batch.__file__)==args.expected_components_sha,'exact package/core/components required')
    require(0<args.time_cap_seconds<=3600,'bounded original time allocation required')
    os.chdir(Path(graygo.__file__).resolve().parents[2]);root.mkdir(parents=True,exist_ok=True);require(not (root/'plan.json').exists(),'profile exists; inspect original handle')
    started=time.monotonic()
    def check():
        if time.monotonic()-started>=args.time_cap_seconds:raise TimeoutError('original graph cost ceiling exhausted')
    with (root/'.cost-profile.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=BoardRuntimeConfig(device='cuda',q_reuse=args.reuse,native_library=str(paths['native_library']),cuda_graphs=True)
        configure_runtime('cuda',True);torch.set_num_threads(1)
        benchmark=ProbeBenchmark.from_dict(json.loads(paths['benchmark'].read_text()));catalog=ExposureCatalog.from_dict(json.loads(paths['catalog'].read_text()))
        plan={'version':VERSION,'source_sha256':source_digest(),'tool_sha256':digest(__file__),'core_producer_sha256':CORE_SHA,'components_sha256':digest(batch.__file__),
            'runtime':config.plan(),'inference_batch_size':32,'pair_batch_size':16,'time_ceiling_seconds':args.time_cap_seconds,'margin':2.,
            'input_files_sha256':{str(p):digest(p) for p in [paths['buffer_checkpoint']/'manifest.json',paths['benchmark'],paths['catalog'],paths['native_library']]},
            'core_protocol':'selected graph runtime pipeline plus eager reference standalone actor/IO; explicitly replaced by selected stages',
            'graph_io_factories':'FrozenPolicies selected execution descriptor and FrozenNeuralPolicy(cuda_graphs=True); original sealed IO body unchanged',
            'scope':'complete measured graph candidate only; no A2, pilot, confirmation or project budget selection'}
        write_report(root/'plan.json',plan);write_report(root/'run.json',make_manifest(seed=config.seed,command=plan,rules=config.plan()['rules']))
        with GraphSetupObserver().installed() as tracker:
            c=core.profile(config,paths['buffer_checkpoint'],benchmark,root/'core',records=32768,iterations=64,warmup=4,actor_iterations=4,dense_records=256,
                margin=2.,time_cap_seconds=max(.001,args.time_cap_seconds-(time.monotonic()-started)))
            check();payload=torch.load(Path(c['io']['checkpoint'])/'cost-payload.pt',map_location='cpu',weights_only=True)
            with torch.random.fork_rng(devices=[]),torch.device('cpu'):backend=NFSPBackend(QNet(),PolicyNet(),config.backend_config(),device='cuda')
            for key in ('q','average','target'):getattr(backend,key).load_state_dict(payload[key])
            backend.q_optimizer.load_state_dict(payload['q_optimizer']);backend.average_optimizer.load_state_dict(payload['average_optimizer']);del payload
            replay=ReplayBuffer.load(Path(c['io']['checkpoint'])/'replay.npz');reservoir=ReservoirBuffer.load(Path(c['io']['checkpoint'])/'reservoir.npz')
            states=core.sample_states(replay,config.envs,np.random.default_rng(config.seed+901))
            actors=actor_measurements(config,backend,states,tracker,check_budget=check)
            check();io=graph_io(config,backend,replay,reservoir,root/'graph-io')
            dense=tuple(node.state for family in benchmark.families for node in (family.before,family.after))
            e=batch.evaluation_components(config,backend.q,backend.average,dense,catalog,root/'evaluation',inference_batch_size=32,pair_batch_size=16,
                check_budget=check,cuda_graphs=True,setup_tracker=tracker)
        check();cold=batch.cold_start('cuda',source_digest(),min(60,max(1,args.time_cap_seconds-(time.monotonic()-started))))
        public=root/'graph-io/cost-public-payload.pt';loads=[]
        for _ in range(3):
            check()
            def load_model():
                digest(public);data=torch.load(public,map_location='cpu',weights_only=True)
                with torch.random.fork_rng(devices=[]),torch.device('cpu'):model=PolicyNet().to('cuda')
                model.load_state_dict(data['average']);return FrozenNeuralPolicy(model,inference_batch_size=32,cuda_graphs=True)
            _,seconds=core.timed(torch.device('cuda'),load_model);loads.append(seconds)
        setup=tracker.report();f=forecast(c,e,actors,io,setup,cold_seconds=cold['seconds'],model_load_seconds=max(loads));check()
        require(source_digest()==plan['source_sha256'] and all(digest(p)==h for p,h in plan['input_files_sha256'].items()),'source/input changed')
        result={'version':VERSION,'status':'complete','plan':plan,'core_report':c,'core_report_sha256':digest(root/'core/report.json'),
            'actors':actors,'graph_io':io,'evaluation':e,'graph_setup':setup,'cold_start':cold,'selected_model_load_seconds':loads,'forecast':f,
            'fresh_learner_profile_records':c['fresh_profile_records'],'evaluation_physical_games':e['physical_games'],'evaluation_physical_rounds':e['physical_rounds'],
            'elapsed_seconds':time.monotonic()-started,'memory':core._memory(torch.device('cuda')),
            'candidate_price_forecast_available':bool(c['candidate_price_forecast_available'] and len(dense)==64 and all(v['eager_fallbacks']==0 for v in setup['pools'].values())),
            'formal_learning_run':False,'a2_complete':False,'execution_authorized':False}
        write_report(root/'report.json',result);print(json.dumps({'status':result['status'],'forecast':f,'elapsed_seconds':result['elapsed_seconds']},indent=2))


if __name__=='__main__':main()

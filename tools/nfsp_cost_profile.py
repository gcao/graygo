"""Full-buffer one-step NFSP cost measurements, separate from learning gates.

Materialized offline rows never become invented optimizer/fresh clocks. Kernel
updates are counted as profiling work; collected append records have local,
explicitly offset namespaces. Profiling weights cannot become public agents.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from graygo.board_augmentation import augment
from graygo.board_collection import BoardCollector
from graygo.board_profile import profile_board,_memory,_synchronize
from graygo.board_runtime import BoardNFSPRuntime,BoardRuntimeConfig,configure_runtime
from graygo.buffers import ReplayBuffer,ReservoirBuffer
from graygo.engine import State
from graygo.evaluation import BaselinePolicy,FixedStarts,FrozenNeuralPolicy,evaluate_pair,seed_for
from graygo.manifest import make_manifest,source_digest,write_report
from graygo.models import QNet,PolicyNet
from graygo.nfsp import NFSPBackend,OneStepBatch,update_q,update_average
from graygo.nfsp_collection import FrozenPolicies,EpisodeActor,commit_many
from graygo.rule_state import RuleObservation,RuleStateSpec
from graygo.schedules import LearnerSchedule,EpisodeChoice
from graygo.storage import CheckpointStore,DEFAULT_RESERVE,digest

VERSION='graygo/full-buffer-nfsp-cost-measurement/1'


def require(condition,message):
    if not condition:raise ValueError(message)
def timed(device,function):
    _synchronize(device);at=time.perf_counter();result=function();_synchronize(device)
    return result,time.perf_counter()-at
def weights(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        x=tensor.detach().cpu().contiguous().numpy();h.update(name.encode()+str(x.dtype).encode()+str(x.shape).encode()+x.tobytes())
    return h.hexdigest()


def materialize(source):
    """Reindex retained valid rows as a new offline one-step sampling corpus."""
    require(source.size==source.capacity and source.spec.to_dict().get('storage')=='rule-state','full rule-state replay corpus required')
    require(bool(source.arrays['value_mask'][:source.size].all()),'cost corpus must contain only valid Q starts')
    order=np.argsort(source.arrays['ordinal'][:source.size]);replay=ReplayBuffer(source.capacity,source.spec)
    for name,array in replay.arrays.items():array[:source.size]=source.arrays[name][order]
    replay.arrays['ordinal'][:]=np.arange(source.size,dtype=np.uint64)
    # Reindex valid retained links as well. This preserves realistic IO data;
    # only one-step kernels are measured, never a Retrace price.
    inverse=np.empty(source.size,dtype=np.int64);inverse[order]=np.arange(source.size)
    old_next=source.arrays['next_index'][order];old_ord=source.arrays['next_ordinal'][order]
    valid=(old_next>=0)&(old_next<source.size);safe=np.maximum(0,np.minimum(source.size-1,old_next))
    valid&=source.arrays['ordinal'][safe]==old_ord
    replay.arrays['next_index'][:]=np.where(valid,inverse[safe],-1)
    replay.arrays['next_ordinal'][:]=np.where(valid,inverse[safe],0)
    replay.size=replay.insertions=replay.q_insertions=replay.q_size=source.size;replay.cursor=0;replay._tails={}
    reservoir=ReservoirBuffer(source.capacity,source.spec,opening_record=0)
    for name in ('spatial','globals','legal'):reservoir.arrays[name][:]=replay.arrays[name]
    reservoir.arrays['greedy'].fill(0);actions=replay.arrays['action'].astype(np.int64)
    reservoir.arrays['greedy'][np.arange(source.size),actions//8]=1<<(actions%8)
    reservoir.arrays['record'][:]=np.arange(source.size,dtype=np.int64);reservoir.arrays['policy_version'].fill(0)
    reservoir.size=reservoir.seen=reservoir.revisions=source.size
    return replay,reservoir,{'source_retained_rows':source.size,'source_original_insertions':source.insertions,
        'source_historical_q_count_known':source.q_insertions is not None,'materialized_rows':source.size,
        'new_environment_actions':0,'sequence_links_retained':True,'retained_links_remapped':int(valid.sum()),
        'label_semantics':'synthetic executed-action one-hot cost targets, not historical NFSP greedy labels',
        'record_semantics':'new offline materialization indices; no optimizer or aggregate learner clocks are set'}


class _ReplaySink:
    def __init__(self,replay):
        self.data=replay;self.base=replay.insertions;self.stream_base=2*(int(replay.arrays['stream'][:replay.size].max())//2+1)
        self.spec=replay.spec
    @property
    def insertions(self):return self.data.insertions-self.base
    def add(self,*args,**kwargs):
        kwargs['stream']+=self.stream_base;return self.data.add(*args,**kwargs)


class _ReservoirSink:
    def __init__(self,reservoir,base):self.data=reservoir;self.base=base;self.spec=reservoir.spec;self.opening_record=0
    def offer_index(self,*args,**kwargs):
        kwargs['record']+=self.base;return self.data.offer_index(*args,**kwargs)


class _AliveRng:
    def __init__(self,replay,rng):
        self.indices=np.flatnonzero(~replay.arrays['terminated'][:replay.size]);self.rng=rng
        require(bool(len(self.indices)),'cost corpus needs real nonterminal transitions')
    def integers(self,low,high,size):return self.indices[self.rng.integers(0,len(self.indices),size)]


def kernel_measurements(config,replay,reservoir,*,iterations,warmup,check_budget=lambda:None):
    require(type(iterations) is int and iterations>=1 and type(warmup) is int and warmup>=0,'positive kernel iteration counts required')
    backend=NFSPBackend(QNet(config.model_config()),PolicyNet(config.model_config()),config.backend_config(),device=config.device)
    device=backend.device;rng=np.random.default_rng(config.seed+701);alive=_AliveRng(replay,rng);sl_rng=np.random.default_rng(config.seed+702)
    aug_q=np.random.default_rng(config.seed+703);aug_sl=np.random.default_rng(config.seed+704)
    q_times=[];sl_times=[];q_stats=[];sl_stats=[];copies=[]
    def q_step():
        data=augment(replay.sample_q(config.batch,alive),aug_q);require(not data['terminated'].any(),'nonterminal stress sampler changed transition flags')
        return update_q(backend.q,backend.target,backend.q_optimizer,OneStepBatch.from_replay(data,device),epsilon=.08875,max_grad_norm=config.max_grad_norm)
    def sl_step():return update_average(backend.average,backend.average_optimizer,augment(reservoir.sample(config.batch,sl_rng),aug_sl),device=device,max_grad_norm=config.max_grad_norm)
    for i in range(warmup+iterations):
        check_budget();stats,elapsed=timed(device,q_step)
        if i>=warmup:q_times.append(elapsed);q_stats.append(stats)
        if (i+1)%(config.target_every//config.batch*config.q_reuse)==0:backend.target.load_state_dict(backend.q.state_dict())
    for i in range(warmup+iterations):
        check_budget();stats,elapsed=timed(device,sl_step)
        if i>=warmup:sl_times.append(elapsed);sl_stats.append(stats)
    for _ in range(max(2,min(iterations,8))):
        check_budget();_,elapsed=timed(device,lambda:backend.target.load_state_dict(backend.q.state_dict()));copies.append(elapsed)
    for optimizer in (backend.q_optimizer,backend.average_optimizer):
        require(all(int(v['step'].item())==warmup+iterations for v in optimizer.state.values()),'measured optimizer steps differ')
    require(all(s['targets']==config.batch for s in q_stats+sl_stats),'measured logical target batch differs')
    return backend,{'q_seconds':q_times,'supervised_seconds':sl_times,'target_copy_seconds':copies,
        'warmup_updates_each':warmup,'measured_updates_each':iterations,'actual_optimizer_updates_each':warmup+iterations,
        'measured_q_loss_targets':iterations*config.batch,'measured_supervised_targets':iterations*config.batch,
        'q_starts':'actual full-corpus nonterminal-only sampling; source transitions/flags unchanged',
        'replay_and_augmentation_included':True,'new_fresh_learner_records':0,
        'counter_semantics':'functional optimization benchmark; no NFSP fresh clock is fabricated and no training resume is emitted'}


def actor_measurements(config,backend,states,*,iterations,check_budget=lambda:None):
    require(len(states)>=config.envs and iterations>=1,'one real public state per environment required')
    device=backend.device;snapshots=[];copy_times=[]
    for i in range(config.envs):
        check_budget();snapshot,elapsed=timed(device,lambda i=i:FrozenPolicies(backend.q,backend.average,version=1000000+i,average_ready=True))
        snapshots.append(snapshot);copy_times.append(elapsed)
    results={}
    for layout in ('shared_q','shared_average','distinct_mixed'):
        samples=[];actors=[]
        for env in range(config.envs):
            pair=[]
            for role in (1,2):
                kind='q' if layout=='shared_q' or (layout=='distinct_mixed' and role==1) else 'average'
                snapshot=snapshots[env] if layout=='distinct_mixed' else snapshots[0]
                choice=EpisodeChoice(5000000,kind=='q',kind,.08875,1.,0. if kind=='q' else .5,.5)
                pair.append(EpisodeActor(snapshot,choice,np.random.default_rng(seed_for(config.seed,'cost-actor',env,role))))
            actors.append(pair)
        for repeat in range(iterations):
            check_budget()
            def commit():
                requests=[]
                for env,state in enumerate(states[:config.envs]):
                    for role in (1,2):
                        actor=actors[env][role-1];requests.append((actor,RuleObservation(state,role),actor.choice.best_response_mode))
                commitments=commit_many(requests)
                require(len(commitments)==2*config.envs,'paired actor count differs')
                return commitments
            commitments,elapsed=timed(device,commit);samples.append(elapsed)
            for index,commitment in enumerate(commitments):require(commitment.action in states[index//2].legal_actions(index%2+1),'cost actor chose an illegal move')
        results[layout]=samples
    return {'samples_seconds':results,'requests_per_sample':2*config.envs,'distinct_mixed_groups':2*config.envs,
            'snapshot_copy_seconds':copy_times,'snapshots':config.envs,'paired_public_states':True,
            'scope':'actual private commitment path; distinct immutable versions stress scalar Q/average groups; measurement actors have received benchmark supervised updates, not deployment qualification'}


def dense_collection(config,backend,replay,reservoir,states,*,records,check_budget=lambda:None):
    require(records>=2 and records%2==0 and all(s.rules==config.rules and not s.done for s in states),'complete declared dense timing roots required')
    starts=FixedStarts(states,label='cost-only-complete-public-roots');collector=BoardCollector(config.rules,config.envs,config.seed+801,starts=starts)
    sink=_ReplaySink(replay);labels=_ReservoirSink(reservoir,sink.base);rng=np.random.default_rng(config.seed+802)
    schedule=LearnerSchedule(40000000,0,1);snapshot=FrozenPolicies(backend.q,backend.average,version=2000000,average_ready=True)
    import graygo.board_collection as module
    original=module.commit_many;actor_seconds=[0.]
    def observed(requests):
        result,seconds=timed(backend.device,lambda:original(requests));actor_seconds[0]+=seconds;return result
    _synchronize(backend.device);started=time.perf_counter();module.commit_many=observed
    try:
        while collector.records<records:
            check_budget();collector.collect(min(2*config.envs,records-collector.records),snapshot,sink,labels,rng,schedule)
    finally:module.commit_many=original
    _synchronize(backend.device);elapsed=time.perf_counter()-started
    require(sink.insertions==collector.records==records and replay.size==reservoir.size==replay.capacity,'full overwrite/append cost accounting differs')
    return {'new_profile_records':records,'total_seconds':elapsed,'actor_seconds':actor_seconds[0],
        'resolver_storage_bookkeeping_seconds':max(0.,elapsed-actor_seconds[0]),'completed_games':collector.completed,
        'root_rounds_played':sorted({s.rounds_played for s in states}),'snapshot_seconds_included':False,
        'buffer_record_offset':sink.base,'new_stream_offset':sink.stream_base,'collector_local_records':collector.records,
        'scope':'real legal dense-root collection and full-buffer overwrite/Algorithm-R work; explicit local record namespace, no optimizer clock set'}


def sample_states(replay,count,rng):
    indices=rng.integers(0,replay.size,count);return tuple(replay.spec.codec.unpack(replay.arrays['spatial'][int(i)])[0] for i in indices)


def forecast(config,rates,*,margin,evaluation_games,checkpoint_every=102400):
    require(math.isfinite(margin) and margin>=1 and checkpoint_every>0,'declared runtime margin and checkpoint interval required')
    names=('collection_per_record','actor_per_record','snapshot','q_update','supervised_update','target_copy','checkpoint_write','checkpoint_restore','baseline_eval_round','neural_eval_round','public_archive','public_model_load','policy_integrity','setup')
    require(set(rates)==set(names) and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in rates.values()),'complete finite cost component inventory required')
    require(set(evaluation_games)=={'pilot','screening_cumulative','confirmation'} and all(type(v) is int and v>=0 for v in evaluation_games.values()),'declared per-seed evaluation budgets required')
    require(evaluation_games['screening_cumulative']>=evaluation_games['pilot'],'screening evaluation cost must include its pilot')
    result={};cuda=torch.device(config.device).type=='cuda'
    for name,n,milestones in (('pilot',8000000,7),('screening_cumulative',20000000,8),('confirmation',80000000,9)):
        q_updates=max(0,n//config.batch-config.learning_start//config.batch+1)*config.q_reuse
        sl_updates=max(0,n//config.batch-5000000//config.batch)
        checkpoints=math.ceil(n/checkpoint_every)+milestones
        parts={'collection':n*rates['collection_per_record'],'private_actors':n*rates['actor_per_record'],
               'frozen_snapshots':(math.ceil(n/config.batch)+1)*rates['snapshot'],'Q':q_updates*rates['q_update'],
               'supervised_average':sl_updates*rates['supervised_update'],'target_refresh':(n//config.target_every)*rates['target_copy'],
               'checkpoints':checkpoints*rates['checkpoint_write'],'one_restore':rates['checkpoint_restore'],
               'public_archives':milestones*rates['public_archive'],
               'evaluation':evaluation_games[name]*config.rules.max_rounds*rates['baseline_eval_round'],
               'evaluation_policy_integrity':2*evaluation_games[name]*rates['policy_integrity'],'setup':rates['setup']}
        result[name]={'records':n,'q_updates':q_updates,'supervised_update_upper':sl_updates,'target_refreshes':n//config.target_every,
                      'full_checkpoint_count_upper':checkpoints,'evaluation_games':evaluation_games[name],'components_seconds':parts,
                      'projected_seconds_before_margin':sum(parts.values()),'ceiling_seconds':margin*sum(parts.values()),
                      'allocated_process_hours':margin*sum(parts.values())/3600,
                      'allocated_single_device_hours':margin*sum(parts.values())/3600 if cuda else None}
    scientific=(96000*config.rules.max_rounds*rates['neural_eval_round']+768*rates['actor_per_record']
                +(2*96000+512)*rates['policy_integrity']+12*rates['public_model_load'])
    return {'per_seed':result,'scientific_and_probe_seconds_with_margin':margin*scientific,
        'scientific_and_probe_device_hours':margin*scientific/3600 if cuda else None,'H_P':result['pilot']['allocated_single_device_hours'],
        'H_D':result['screening_cumulative']['allocated_single_device_hours'],'H_N':result['confirmation']['allocated_single_device_hours'],
        'margin':margin,'concurrency':1,'basis':'one allocated process/device; no eight-job amortization assumed',
        'scope':'observed-component forecast with declared event/game upper counts and margin, not a runtime theorem or learned strength'}


def pipeline_measurements(config,root,*,records,time_cap_seconds,check_budget=lambda:None):
    """Observe the actual bounded runtime; wrappers only synchronize/time calls."""
    import graygo.board_profile as profile_module
    import graygo.board_collection as collection_module
    original_class=profile_module.BoardNFSPRuntime;original_commit=collection_module.commit_many;active=[None];measurements=[]
    class Instrumented(original_class):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.cost_observer={'actor_seconds':0.,'snapshot_copy_seconds':0.,'snapshot_copies':0};measurements.append(self.cost_observer)
        def _snapshot(self):
            version=self.backend.counters.q_updates+self.backend.counters.supervised_updates
            changed=self.snapshot is None or self.snapshot.version!=version
            result,seconds=timed(self.backend.device,lambda:super(Instrumented,self)._snapshot())
            if changed:self.cost_observer['snapshot_copy_seconds']+=seconds;self.cost_observer['snapshot_copies']+=1
            return result
        def advance(self,*args,**kwargs):
            check_budget();active[0]=self
            try:return super().advance(*args,**kwargs)
            finally:active[0]=None
    def observed(requests):
        require(active[0] is not None,'actor timing escaped its real collector context')
        result,seconds=timed(active[0].backend.device,lambda:original_commit(requests));active[0].cost_observer['actor_seconds']+=seconds;return result
    profile_module.BoardNFSPRuntime=Instrumented;collection_module.commit_many=observed
    try:report=profile_module.profile_board(config,root,records=records,time_cap_seconds=time_cap_seconds)
    finally:profile_module.BoardNFSPRuntime=original_class;collection_module.commit_many=original_commit
    require(report['status']=='complete' and len(measurements)==2,'complete real closed/post pipeline coverage required')
    rates=[]
    for (name,result),observer in zip(report['phases'].items(),measurements):
        residual=result['timings']['collection']-observer['actor_seconds']-observer['snapshot_copy_seconds']
        require(residual>=-.001,'nested timing scopes exceed real collection interval')
        observer.update(phase=name,records=result['records'],collection_seconds=result['timings']['collection'],
                        resolver_storage_bookkeeping_seconds=max(0.,residual))
        rates.append(max(0.,residual)/result['records'])
    return report,measurements,max(rates)


def io_measurements(config,backend,replay,reservoir,root,*,reserve_bytes=DEFAULT_RESERVE):
    device=backend.device;snapshots={i:FrozenPolicies(backend.q,backend.average,version=3000000+i,average_ready=True).state_dict() for i in range(config.envs)}
    event={'record':0,'step':0,'actions':[0,0],'probabilities':[.5,.5],'modes':[True,False],
           'acting_policies':['q','average'],'versions':[0,1],'epsilons':[.08875,.08875],
           'supplemental':False,'q_eligible':[True,True],'terminated':False,'reward_black':0.}
    logs=[[dict(deepcopy(event),step=i,record=2*i) for i in range(config.rules.max_rounds-1)] for _ in range(config.envs)]
    payload={'profile_only':True,'q':backend.q.state_dict(),'average':backend.average.state_dict(),'target':backend.target.state_dict(),
             'q_optimizer':backend.q_optimizer.state_dict(),'average_optimizer':backend.average_optimizer.state_dict(),
             'frozen_snapshot_stress':snapshots,'maximum_length_log_metadata_stress':logs,
             'scope':'sealed IO stress payload, not a valid learner fresh-clock/resume state or public model archive'}
    store=CheckpointStore(root,reserve_bytes=reserve_bytes)
    def writer(path):
        replay.save(path/'replay.npz');reservoir.save(path/'reservoir.npz');torch.save(payload,path/'cost-payload.pt')
    bundle,write_seconds=timed(device,lambda:store.save(replay.insertions,writer,kind='weights',
        metadata={'profile_only':True,'source_sha256':source_digest()},estimated_peak_bytes=replay.snapshot_raw_bytes+reservoir.snapshot_raw_bytes+256*1024**2))
    seal=CheckpointStore.verify(bundle);before=(weights(backend.q),weights(backend.average),weights(backend.target))
    def restore():
        data=torch.load(bundle/'cost-payload.pt',map_location='cpu',weights_only=True)
        r=ReplayBuffer.load(bundle/'replay.npz');s=ReservoirBuffer.load(bundle/'reservoir.npz')
        backend.q.load_state_dict(data['q']);backend.average.load_state_dict(data['average']);backend.target.load_state_dict(data['target'])
        backend.q_optimizer.load_state_dict(data['q_optimizer']);backend.average_optimizer.load_state_dict(data['average_optimizer'])
        return r,s
    (loaded,loaded_s),restore_seconds=timed(device,restore)
    require(before==(weights(backend.q),weights(backend.average),weights(backend.target)),'full IO changed learner parameters')
    for original,new in ((replay,loaded),(reservoir,loaded_s)):
        for name,array in original.arrays.items():require(np.array_equal(array[:original.size],new.arrays[name][:new.size]),'full buffer IO differs')
    def public_payload():
        torch.save({'profile_only':True,'q':backend.q.state_dict(),'average':backend.average.state_dict()},Path(root)/'cost-public-payload.pt')
    _,archive_seconds=timed(device,public_payload)
    def load_public():
        data=torch.load(Path(root)/'cost-public-payload.pt',map_location='cpu',weights_only=True)
        with torch.random.fork_rng(devices=[]),torch.device('cpu'):model=PolicyNet(config.model_config()).to(device)
        model.load_state_dict(data['average']);return FrozenNeuralPolicy(model)
    _,load_seconds=timed(device,load_public)
    return {'full_checkpoint_write_seconds':write_seconds,'full_checkpoint_restore_seconds':restore_seconds,
        'public_payload_seconds':archive_seconds,'public_model_load_seconds':load_seconds,'buffer_roundtrip_verified':True,'model_roundtrip_verified':True,
        'frozen_snapshots_in_payload':config.envs,'metadata_log_length_each':config.rules.max_rounds-1,
        'checkpoint':str(bundle),'checkpoint_manifest_sha256':digest(bundle/'manifest.json'),
        'payload_bytes':sum(item['bytes'] for item in seal['files'].values()),'training_resume_or_public_agent':False}


def evaluation_measurements(config,backend,states,*,check_budget=lambda:None):
    result=[];baseline_rates=[];neural_rates=[]
    average=FrozenNeuralPolicy(backend.average);q=FrozenNeuralPolicy(backend.q,representation='greedy_q')
    for representation,subject,opponent in (('q/random',q,BaselinePolicy('random')),('q/tactical',q,BaselinePolicy('tactical')),
            ('average/random',average,BaselinePolicy('random')),('average/tactical',average,BaselinePolicy('tactical')),
            ('average/average',average,average)):
        check_budget();pair,elapsed=timed(backend.device,lambda:evaluate_pair(subject,opponent,State(config.rules),unit_seed=seed_for(config.seed,'cost-evaluation',representation)))
        rounds=sum(game['rounds'] for game in pair['subject']);require(rounds>0,'evaluation produced no real rounds')
        rates=neural_rates if representation=='average/average' else baseline_rates;rates.append(elapsed/rounds)
        # Empty/dense actual legal round calls guard against short untrained
        # games hiding a slow resolver/heuristic/encoding path.
        for root in (State(config.rules),*states[:2]):
            a=subject.begin_episode(seed_for(config.seed,'cost-single-a',representation));b=opponent.begin_episode(seed_for(config.seed,'cost-single-b',representation))
            def round_call():
                black=a.action(root,1);white=b.action(root,2)
                from graygo.engine import resolve,score
                following=resolve(root,black,white).state
                if following.done:score(following.board,following.rules.size)
            _,seconds=timed(backend.device,round_call);rates.append(seconds)
        result.append({'representation':representation,'paired_games':2,'physical_rounds':rounds,'seconds':elapsed,'pair':pair})
    integrity=[]
    for policy in (q,average):
        _,seconds=timed(backend.device,policy.assert_frozen);integrity.append(seconds)
    return {'matches':result,'baseline_seconds_per_round':baseline_rates,'neural_seconds_per_round':neural_rates,'policy_integrity_seconds':integrity,
            'strength_claim':False,'measurement_weights':'profiling-only optimizer outputs'}


def profile(config,buffer_checkpoint,benchmark,root,*,records,iterations,warmup,actor_iterations,dense_records,margin,time_cap_seconds,
            evaluation_games=None,diagnostic=False,reserve_bytes=DEFAULT_RESERVE):
    import fcntl
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True);started=time.monotonic();device=torch.device(config.device)
    require(0<time_cap_seconds<=3600 and records<=32768 and config.learning_start<=records and records%config.batch==0,'bounded pipeline record/time scope required')
    require(type(iterations) is int and iterations>=1 and type(warmup) is int and warmup>=0 and type(actor_iterations) is int and actor_iterations>=1,
            'positive kernel/actor counts and nonnegative warmup required before profiling')
    require(type(dense_records) is int and dense_records>=2 and dense_records%2==0 and math.isfinite(margin) and margin>=1,
            'paired dense records and stated forecast margin required before profiling')
    games={'pilot':36000,'screening_cumulative':40000,'confirmation':44000} if evaluation_games is None else evaluation_games
    require(set(games)=={'pilot','screening_cumulative','confirmation'} and all(type(v) is int and v>=0 for v in games.values())
            and games['screening_cumulative']>=games['pilot'],'complete declared per-seed evaluation scope required before profiling')
    require(config.rules.size==9 or diagnostic,'production cost profile uses official 9x9 rules')
    require(not (root/'plan.json').exists(),'cost profile exists; inspect its original process/result')
    def check_budget():
        if time.monotonic()-started>=time_cap_seconds:raise TimeoutError('original cost profile time ceiling exhausted; no complete forecast')
    with (root/'.cost-profile.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        configure_runtime(device,True);torch.set_num_threads(1)
        seal=CheckpointStore.verify(buffer_checkpoint);source=ReplayBuffer.load(Path(buffer_checkpoint)/'replay.npz')
        require(source.spec.rules==config.rules,'full cost corpus rules differ')
        replay,reservoir,materialized=materialize(source);del source
        require(replay.spec.native_library==config.native_library,'cost corpus decoder differs from selected runtime')
        require(benchmark.family_coverage_complete,'complete candidate-independent dense benchmark required')
        dense_states=tuple(root_.state for family in benchmark.families for root_ in (family.before,family.after))
        plan={'version':VERSION,'source_sha256':source_digest(),'tool_sha256':digest(Path(__file__)),'runtime':config.plan(),
            'input_checkpoint_manifest_sha256':digest(Path(buffer_checkpoint)/'manifest.json'),'native_schema':replay.spec.to_dict(),'dense_benchmark_sha256':benchmark.sha256,
            'records_per_pipeline_phase':records,'kernel_iterations_each':iterations,'kernel_warmup_each':warmup,
            'actor_iterations_each':actor_iterations,'dense_collection_records':dense_records,'time_ceiling_seconds':time_cap_seconds,
            'target':'one-step','evaluation_game_budgets_per_seed':games,'checkpoint_every_records':102400,
            'margin':margin,'diagnostic':diagnostic,'reserve_bytes':reserve_bytes,'scope':'hardware-only, source-bound offline materialization and actual source kernels; no A2/learning-scope grant'}
        write_report(root/'plan.json',plan,reserve_bytes=reserve_bytes)
        write_report(root/'run.json',make_manifest(seed=config.seed,command=plan,rules=config.plan()['rules']),reserve_bytes=reserve_bytes)
        check_budget();pipeline,observed,base_rate=pipeline_measurements(config,root/'pipeline',records=records,time_cap_seconds=max(.001,time_cap_seconds-(time.monotonic()-started)),check_budget=check_budget)
        check_budget();backend,kernels=kernel_measurements(config,replay,reservoir,iterations=iterations,warmup=warmup,check_budget=check_budget)
        random_states=sample_states(replay,config.envs,np.random.default_rng(config.seed+901))
        actors=actor_measurements(config,backend,random_states,iterations=actor_iterations,check_budget=check_budget)
        dense=dense_collection(config,backend,replay,reservoir,dense_states,records=dense_records,check_budget=check_budget)
        check_budget();io=io_measurements(config,backend,replay,reservoir,root/'full-io',reserve_bytes=reserve_bytes)
        evaluation=evaluation_measurements(config,backend,dense_states,check_budget=check_budget)
        rates={'collection_per_record':max(base_rate,dense['resolver_storage_bookkeeping_seconds']/dense_records),
            'actor_per_record':max(max(v) for v in actors['samples_seconds'].values())/actors['requests_per_sample'],
            'snapshot':max(actors['snapshot_copy_seconds']),'q_update':max(kernels['q_seconds']),
            'supervised_update':max(kernels['supervised_seconds']),'target_copy':max(kernels['target_copy_seconds']),
            'checkpoint_write':io['full_checkpoint_write_seconds'],'checkpoint_restore':io['full_checkpoint_restore_seconds'],
            'baseline_eval_round':max(evaluation['baseline_seconds_per_round']),'neural_eval_round':max(evaluation['neural_seconds_per_round']),
            'public_archive':io['public_payload_seconds'],'public_model_load':io['public_model_load_seconds'],
            'policy_integrity':max(evaluation['policy_integrity_seconds']),'setup':pipeline['phases']['closed']['setup_seconds']}
        projected=forecast(config,rates,margin=margin,evaluation_games=games)
        qualified=bool(not diagnostic and device.type=='cuda'
            and (config.width,config.blocks,config.norm_groups)==(64,8,8) and config.capacity==500000 and config.batch==512
            and config.learning_start==4096 and config.target_every==12800 and config.rules.max_rounds==200 and config.rules.collision=='gray'
            and records>=25600 and iterations>=32 and warmup>=4 and actor_iterations>=4 and dense_records>=256
            and replay.size==reservoir.size==500000 and config.augmentation and config.deterministic and config.native_library is not None)
        check_budget();require(source_digest()==plan['source_sha256'],'cost profile source changed during measurement')
        result={'version':VERSION,'status':'complete','plan':plan,'materialized_dataset':materialized,'reference_pipeline_report_sha256':digest(root/'pipeline/report.json'),
            'reference_pipeline_timing_observers':observed,'full_buffers':{'replay_size':replay.size,'reservoir_size':reservoir.size,
                'replay_allocated_bytes':replay.allocated_bytes,'reservoir_allocated_bytes':reservoir.allocated_bytes},
            'kernels':kernels,'actors':actors,'dense_collection':dense,'io':io,'evaluation':evaluation,'rates_seconds':rates,
            'evaluation_game_budgets_per_seed':games,'forecast':projected,'full_component_coverage':True,'candidate_price_forecast_available':qualified,
            'elapsed_seconds':time.monotonic()-started,'memory':_memory(device),'fresh_profile_records':2*records+dense_records,
            'offline_rows_reused':replay.capacity,'formal_learning_run':False,'a2_complete':False,'execution_authorized':False,
            'scope':'measured complete component forecast; assumed maximum observed per-event cost plus margin, not a hardware/runtime bound, training outcome or selected scope'}
        write_report(root/'report.json',result,reserve_bytes=reserve_bytes);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--buffer-checkpoint',required=True)
    parser.add_argument('--benchmark',required=True);parser.add_argument('--native-library',required=True);parser.add_argument('--expected-source-sha',required=True)
    parser.add_argument('--reuse',type=int,choices=(1,4),required=True);parser.add_argument('--device',default='cuda');parser.add_argument('--time-cap-seconds',type=int,required=True)
    parser.add_argument('--records',type=int,default=32768);parser.add_argument('--iterations',type=int,default=64);parser.add_argument('--warmup',type=int,default=4)
    parser.add_argument('--actor-iterations',type=int,default=4);parser.add_argument('--dense-records',type=int,default=256);parser.add_argument('--margin',type=float,default=2.)
    args=parser.parse_args();require(source_digest()==args.expected_source_sha,'use exact selected learner package on PYTHONPATH')
    root=Path(args.run_dir).resolve();buffer=Path(args.buffer_checkpoint).resolve();benchmark_path=Path(args.benchmark).resolve();library=Path(args.native_library).resolve()
    import os,graygo
    os.chdir(Path(graygo.__file__).resolve().parents[2])
    from graygo.probe_benchmark import ProbeBenchmark
    benchmark=ProbeBenchmark.from_dict(json.loads(benchmark_path.read_text()))
    config=BoardRuntimeConfig(q_reuse=args.reuse,device=args.device,native_library=str(library))
    result=profile(config,buffer,benchmark,root,records=args.records,iterations=args.iterations,warmup=args.warmup,
        actor_iterations=args.actor_iterations,dense_records=args.dense_records,margin=args.margin,time_cap_seconds=args.time_cap_seconds)
    print(json.dumps({'status':result['status'],'candidate_price_forecast_available':result['candidate_price_forecast_available'],'forecast':result['forecast'],'elapsed_seconds':result['elapsed_seconds']},indent=2,sort_keys=True))


if __name__=='__main__':main()

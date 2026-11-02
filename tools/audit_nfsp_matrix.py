"""Read-only independent audit of a completed native NFSP matrix trial.

Use the trial's pinned source on PYTHONPATH. Never instantiate a trainer or
restore/truncate its audit log. Security gaps use closed-form original games.
"""
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
from pathlib import Path
import time
import numpy as np
import torch
from graygo.buffers import ReplayBuffer,ReservoirBuffer
from graygo.episode_audit import EPISODE_DTYPE
from graygo.manifest import source_digest,write_report
from graygo.nfsp_training import NFSPMatrixConfig
from graygo.storage import CheckpointStore,digest
from graygo.tabular_average import TabularAverage
from graygo.task_models import CompactQ,CompactPolicy,TabularQ


def original_game(name):
    if name=='asymmetric':return np.array([[1.,-1.],[-1.,.5]]),-1/7
    if name=='rps':return np.array([[0.,-1.,1.],[1.,0.,-1.],[-1.,1.,0.]]),0.
    if name=='matching-pennies':return np.array([[1.,-1.],[-1.,1.]]),0.
    if name=='pure82':
        a=-np.ones((82,82));a[0]=1.;return a,1.
    raise ValueError('unknown original game')


def security_gaps(a,value,policies):
    return [max(0.,float(value-(policies[0]@a).min())),max(0.,float((a@policies[1]).max()-value))]


def require(condition,message):
    if not condition:raise ValueError(message)


def published_table(state,reservoir,config,log,records,recent,last_publication):
    """Reconstruct the last published reservoir, retaining lost-label bounds."""
    result={'published_tabular_mean_verified':False,'published_tabular_mean_interval_checked':False,
            'unresolved_published_greedy_labels':0,'exact_published_mean_roles':[],'last_supervised_record':last_publication}
    if config.learner!='tabular':return result
    published=records.copy()
    if last_publication is not None:
        for record,slot,previous in reversed([change for change in recent if change[0]>=last_publication]):published[slot]=previous
    else:published[:]=-1
    counts=np.bincount(published[published>=0]%2,minlength=2);totals=np.zeros((2,reservoir.spec.actions),dtype=np.float64);unknown=np.zeros(2,dtype=np.int64)
    for slot in np.flatnonzero(published>=0):
        record=int(published[slot]);role=record%2
        if config.label_rule=='executed':totals[role,int(log[record//2]['actions'][role])]+=1
        elif published[slot]==records[slot]:
            mask=np.unpackbits(reservoir.arrays['greedy'][slot],count=reservoir.spec.actions,bitorder='little').astype(float)
            require(mask.any(),'empty published greedy target');totals[role]+=mask/mask.sum()
        else:unknown[role]+=1
    table=state['backend']['average']['probabilities'].numpy()
    require(table.shape==totals.shape and np.isfinite(table).all() and (table>=0).all() and (table<=1).all(),'published table is not a probability table')
    np.testing.assert_allclose(table.sum(-1),np.ones(2),rtol=0,atol=1e-10)
    exact=[]
    for role in range(2):
        if not unknown[role]:
            expected=totals[role]/counts[role] if counts[role] else np.ones(table.shape[1])/table.shape[1]
            np.testing.assert_allclose(table[role],expected,rtol=0,atol=1e-10,err_msg='published conditional label mean differs')
            exact.append(role+1)
        else:
            residual=table[role]*counts[role]-totals[role]
            require((residual>=-1e-8).all() and (residual<=unknown[role]+1e-8).all(),'published mean exceeds unresolved greedy-label bounds')
            np.testing.assert_allclose(residual.sum(),unknown[role],rtol=0,atol=1e-8)
    return {**result,'published_tabular_mean_verified':not unknown.any(),'published_tabular_mean_interval_checked':True,
            'unresolved_published_greedy_labels':int(unknown.sum()),'exact_published_mean_roles':exact,
            'published_label_counts_per_role':counts.tolist(),
            'published_mean_scope':'executed labels recover exactly from permanent actions; missing replaced greedy targets retain necessary simplex bounds'}


def _audit(root,*,require_formal=True):
    root=Path(root);manifest=json.loads((root/'run.json').read_text());config=NFSPMatrixConfig(**manifest['config'])
    require(manifest['source_sha256']==source_digest(),'load the original pinned source before auditing')
    require(set(manifest['dependencies'])=={'numpy','torch','scipy'},'dependency inventory differs')
    for package,version in manifest['dependencies'].items():require(version is not None and metadata.version(package)==version,'use the pinned dependency versions: '+package)
    if require_formal:require(config.formal_protocol(),'diagnostic recipe is not a formal primary trial')
    matrix,value=original_game(config.task);np.testing.assert_array_equal(matrix,manifest['payoffs'])
    n=config.records;bundle=root/'resume'/f'{n:012d}';meta=CheckpointStore.verify(bundle)
    require(meta['kind']=='resume','completed resume checkpoint required')
    require(meta['metadata']['source_sha256']==manifest['source_sha256'],'sealed source metadata differs')
    state=torch.load(bundle/'trainer.pt',weights_only=True,map_location='cpu')
    require(state['config']==asdict(config) and meta['step']==config.records,'trial is incomplete or configuration differs')
    counts=state['backend']['counters'];batch=config.batch;actions=matrix.shape[0]
    require(matrix.shape==(actions,actions),'auditor expects the declared square tasks')
    require(counts['fresh_q']==counts['fresh_ordinary']==n and state['per_role']==[n//2,n//2] and state['episodes']==n//2,'own-side/role/game accounting differs')
    corrected=state['backend'].get('clock_version',1)==2
    if corrected:require(counts['fresh_records']==n,'aggregate fresh clock differs')
    q_updates=max(0,n//batch-config.learning_start//batch+1)*config.q_reuse
    require(counts['q_updates']==q_updates and counts['q_loss_targets']==q_updates*batch and counts['target_refreshes']==n//config.target_every,'Q/reuse/refresh counters differ')
    audit_state=state['episode_audit'];path=root/'episodes.bin';size=path.stat().st_size
    require(size==audit_state['bytes']==n//2*EPISODE_DTYPE.itemsize,'completed episode audit length differs')
    h=hashlib.sha256()
    with path.open('rb') as file:
        for data in iter(lambda:file.read(1024*1024),b''):h.update(data)
    require(h.hexdigest()==audit_state['sha256'],'episode audit prefix checksum differs')
    schema=json.loads((root/'episodes.schema.json').read_text())
    require(schema['itemsize']==EPISODE_DTYPE.itemsize and schema['dtype']==json.loads(json.dumps(EPISODE_DTYPE.descr)),'episode audit schema differs')
    log=np.memmap(path,dtype=EPISODE_DTYPE,mode='r',shape=(n//2,))
    replay=ReplayBuffer.load(bundle/'replay.npz');reservoir=ReservoirBuffer.load(bundle/'reservoir.npz')
    require(replay.insertions==n and replay.capacity==reservoir.capacity==config.capacity,'buffer count/capacity differs')
    require(replay.size==min(n,config.capacity) and replay.spec==reservoir.spec and replay.spec.spatial_shape==(2,)
            and replay.spec.globals==0 and replay.spec.actions==actions and reservoir.opening_record==config.reservoir_open,'buffer observation/opening schema differs')
    if corrected:require(replay.q_insertions==n and replay.q_size==replay.size,'all-eligible replay history/live count differs')
    expected_records=np.full(config.capacity,-1,dtype=np.int64);expected_versions=np.zeros(config.capacity,dtype=np.uint64)
    mode_rng=[np.random.default_rng(config.seed+101),np.random.default_rng(config.seed+102)]
    reservoir_rng=np.random.default_rng(config.seed+105);q_rng=np.random.default_rng(config.seed+106);sl_rng=np.random.default_rng(config.seed+107)
    seen=changes=sl_updates=sl_targets=0;first_sl=None;last_sl=None;last_sl_seen=0;recent=deque(maxlen=batch)
    mode_counts=np.zeros(2,dtype=np.int64);acting_counts=np.zeros(3,dtype=np.int64)
    for start in range(0,len(log),50000):
        rows=log[start:start+50000];records=2*np.arange(start,start+len(rows),dtype=np.int64)
        np.testing.assert_array_equal(rows['episode'],records//2);np.testing.assert_array_equal(rows['record'],records)
        for role in range(2):
            draws=mode_rng[role].random((len(rows),2))
            np.testing.assert_array_equal(draws[:,0],rows['mode_draw'][:,role]);np.testing.assert_array_equal(draws[:,1],rows['ramp_draw'][:,role])
        best=rows['mode_draw']<.1;np.testing.assert_array_equal(best,rows['best_response'].astype(bool))
        eligible=best & ((records[:,None]+np.arange(2))>=config.reservoir_open)
        ordinals=(records[:,None]+np.arange(2)).reshape(-1);flags=eligible.reshape(-1)
        cumulative=seen+np.cumsum(flags,dtype=np.int64)
        boundaries=(ordinals+1)%batch==0
        boundary_records=ordinals[boundaries]+1;boundary_seen=cumulative[boundaries]
        if first_sl is None and (boundary_seen>=batch).any():first_sl=int(boundary_records[np.flatnonzero(boundary_seen>=batch)[0]])
        supervised=boundary_seen>=batch
        if supervised.any():last_sl=int(boundary_records[supervised][-1]);last_sl_seen=int(boundary_seen[supervised][-1])
        sl_updates+=int(supervised.sum())
        sl_targets+=int(np.minimum(boundary_seen[supervised],config.capacity).sum()) if config.learner=='tabular' else int(supervised.sum())*batch
        ready=np.zeros(len(rows),dtype=bool) if first_sl is None else records>=first_sl
        np.testing.assert_array_equal(ready,rows['average_ready'])
        ramp=np.clip((records-config.reservoir_open)/(config.handoff_end-config.reservoir_open),0,1)
        eps=.01+.09*np.maximum(0,1-records/config.epsilon_decay)
        np.testing.assert_allclose(rows['ramp_probability'],np.repeat(ramp[:,None],2,axis=1),atol=1e-15,rtol=0)
        np.testing.assert_allclose(rows['epsilon'],np.repeat(eps[:,None],2,axis=1),atol=1e-15,rtol=0)
        acting=np.where(~best & ready[:,None] & (rows['ramp_draw']<ramp[:,None]),1,np.where(~best & config.uniform_comparison,2,0))
        np.testing.assert_array_equal(acting,rows['acting'])
        qgen=np.maximum(0,records//batch-config.learning_start//batch+1)*config.q_reuse
        slgen=np.zeros(len(rows),dtype=np.int64) if first_sl is None else np.maximum(0,records//batch-first_sl//batch+1)
        np.testing.assert_array_equal(qgen+slgen,rows['generation'])
        require(np.all(rows['actions']<actions) and np.isfinite(rows['behavior_prob']).all() and np.all((rows['behavior_prob']>0)&(rows['behavior_prob']<=1)),'invalid action/probability in audit')
        np.testing.assert_array_equal(rows['reward_black'],matrix[rows['actions'][:,0],rows['actions'][:,1]])
        mode_counts+=best.sum(0);acting_counts+=np.bincount(acting.reshape(-1),minlength=3)
        for boundary,cumulative_seen in zip(boundary_records,boundary_seen):
            if boundary>=config.learning_start:
                for _ in range(config.q_reuse):q_rng.integers(0,min(int(boundary),config.capacity),batch)
            if cumulative_seen>=batch and config.learner=='neural':sl_rng.integers(0,min(int(cumulative_seen),config.capacity),batch)
        for local in np.flatnonzero(flags):
            record=int(ordinals[local]);row=rows[local//2]
            slot=seen if seen<config.capacity else int(reservoir_rng.integers(0,seen+1));seen+=1
            if slot<config.capacity:
                recent.append((record,slot,int(expected_records[slot])))
                expected_records[slot]=record;expected_versions[slot]=int(row['generation'])*2;changes+=1
    require(seen==reservoir.seen and changes==reservoir.revisions==state['inserted_labels'],'reservoir eligibility/replacement counts differ')
    np.testing.assert_array_equal(expected_records[:reservoir.size],reservoir.arrays['record'][:reservoir.size])
    np.testing.assert_array_equal(expected_versions[:reservoir.size],reservoir.arrays['policy_version'][:reservoir.size])
    require(mode_counts.tolist()==state['best_response_visits'],'logical-mode counts differ')
    require(acting_counts.tolist()==[state['acting_counts'][k] for k in ('q','average','uniform')],'actual-policy counts differ')
    require(counts['supervised_updates']==sl_updates and counts['supervised_targets']==sl_targets,'supervised publication/loss counters differ')
    for key,rng in [('mode_black',mode_rng[0]),('mode_white',mode_rng[1]),('reservoir',reservoir_rng),('q_replay',q_rng),('supervised',sl_rng)]:
        require(rng.bit_generator.state==state['rng'][key],key+' RNG continuation differs')
    for role,name in enumerate(('action_black','action_white')):
        rng=np.random.default_rng(config.seed+103+role);rng.bit_generator.advance(n//2)
        require(rng.bit_generator.state==state['rng'][name],name+' private draw count differs')
    for optimizer,updates in [('q_optimizer',q_updates),('average_optimizer',sl_updates)]:
        data=state['backend'][optimizer]
        if data is not None:
            require(all(int(s['step'].item())==updates for s in data['state'].values()),'Adam step counts differ')
    # Independently check all retained executed records against the permanent log.
    a=replay.arrays;valid=replay.size;ordinals=a['ordinal'][:valid].astype(np.int64);roles=ordinals%2;rows=log[ordinals//2]
    np.testing.assert_array_equal(np.sort(ordinals),np.arange(n-valid,n))
    np.testing.assert_array_equal(a['action'][:valid],rows['actions'][np.arange(valid),roles])
    np.testing.assert_array_equal(a['behavior_prob'][:valid],rows['behavior_prob'][np.arange(valid),roles].astype(np.float32))
    np.testing.assert_array_equal(a['mode'][:valid],rows['best_response'][np.arange(valid),roles])
    versions=2*rows['generation'][:,None]+(rows['acting']==1)
    np.testing.assert_array_equal(a['policy_version'][:valid],versions[np.arange(valid),roles])
    np.testing.assert_array_equal(a['opponent_version'][:valid],versions[np.arange(valid),1-roles])
    np.testing.assert_array_equal(a['reward'][:valid],(rows['reward_black']*np.where(roles==0,1,-1)).astype(np.float32))
    require(a['terminated'][:valid].all() and not a['supplemental'][:valid].any() and a['actor_mask'][:valid].all() and a['value_mask'][:valid].all(),'matrix record eligibility/termination differs')
    np.testing.assert_array_equal(a['stream'][:valid],ordinals);require(not a['step'][:valid].any(),'one-round stream has nonzero step')
    spatial,_,legal=replay.spec.unpack(a['spatial'][:valid],a['globals'][:valid],a['legal'][:valid])
    np.testing.assert_array_equal(spatial,np.eye(2,dtype=np.uint8)[roles]);require(legal.all() and not a['next_legal'][:valid].any() and not a['next_spatial'][:valid].any(),'role/terminal observations differ')
    np.testing.assert_array_equal(a['spatial'][:valid],np.packbits(np.eye(2,dtype=np.uint8)[roles],axis=-1,bitorder='little'))
    np.testing.assert_array_equal(a['legal'][:valid],np.packbits(np.ones((valid,actions),dtype=np.uint8),axis=-1,bitorder='little'))
    labels=reservoir.arrays;label_records=labels['record'][:reservoir.size].astype(np.int64);label_roles=label_records%2
    label_x,label_g,label_legal=reservoir.spec.unpack(labels['spatial'][:reservoir.size],labels['globals'][:reservoir.size],labels['legal'][:reservoir.size])
    np.testing.assert_array_equal(label_x,np.eye(2,dtype=np.uint8)[label_roles]);require(label_legal.all() and label_g.shape==(reservoir.size,0),'reservoir observation differs')
    np.testing.assert_array_equal(labels['spatial'][:reservoir.size],np.packbits(label_x,axis=-1,bitorder='little'))
    np.testing.assert_array_equal(labels['legal'][:reservoir.size],np.packbits(label_legal,axis=-1,bitorder='little'))
    masks=np.unpackbits(labels['greedy'][:reservoir.size],axis=-1,count=actions,bitorder='little').astype(bool)
    require(masks.any(-1).all(),'empty frozen reservoir target')
    np.testing.assert_array_equal(labels['greedy'][:reservoir.size],np.packbits(masks,axis=-1,bitorder='little'))
    executed_labels=0
    if config.label_rule=='executed':
        expected=np.eye(actions,dtype=bool)[log[label_records//2]['actions'][np.arange(reservoir.size),label_roles]]
        np.testing.assert_array_equal(masks,expected);executed_labels=reservoir.size
    # Reconstruct every scheduled policy and its security gaps from archived weights.
    torch.set_num_threads(1);evaluations=state['evaluations'];expected_steps=list(range(config.eval_every,n+1,config.eval_every))
    if not expected_steps or expected_steps[-1]!=n:expected_steps.append(n)
    require([e['records'] for e in evaluations]==expected_steps,'scheduled evaluations missing or duplicated')
    require(sorted(int(path.name) for path in (root/'weights').glob('[0-9]*'))==expected_steps,'permanent archive inventory differs')
    checks=[];scheduled=[];archive_hashes={};boundary_actions=boundary_targets=0
    for e in evaluations:
        step=e['records'];archive=root/'weights'/f'{step:012d}';sealed=CheckpointStore.verify(archive)
        require(sealed['kind']=='weights' and sealed['step']==step and sealed['metadata']['source_sha256']==manifest['source_sha256'],'archive step/source differs')
        archive_hashes[str(step)]=digest(archive/'manifest.json')
        weights=torch.load(archive/'models.pt',weights_only=True,map_location='cpu')
        require(weights['config']==asdict(config),'archived model configuration differs')
        if config.learner=='tabular':q,policy=TabularQ(2,actions),TabularAverage(2,actions)
        else:q,policy=CompactQ(2,0,actions,config.hidden),CompactPolicy(2,0,actions,config.hidden)
        q.load_state_dict(weights['q']);policy.load_state_dict(weights['average'])
        with torch.no_grad():
            features=torch.eye(2);globals=torch.empty(2,0);qvalues=q(features,globals)
            probabilities=policy(features,globals).log_softmax(-1).exp().double().numpy()
        require(bool(torch.isfinite(qvalues).all()) and bool((qvalues.abs()<=1+1e-6).all()) and np.isfinite(probabilities).all(),'invalid archived Q/average outputs')
        probabilities/=probabilities.sum(-1,keepdims=True)
        scores=qvalues.detach().numpy();greedy=(scores==scores.max(-1,keepdims=True)).astype(float);greedy/=greedy.sum(-1,keepdims=True)
        gaps=security_gaps(matrix,value,probabilities);qgaps=security_gaps(matrix,value,greedy)
        np.testing.assert_allclose(e['average_policies'],probabilities,rtol=0,atol=1e-12)
        np.testing.assert_allclose(e['q_values'],scores,rtol=0,atol=0)
        np.testing.assert_allclose(e['average_gaps'],gaps,rtol=0,atol=1e-12)
        np.testing.assert_allclose(e['greedy_q_gaps'],qgaps,rtol=0,atol=1e-12)
        require(e['counters']['fresh_q']==e['counters']['fresh_ordinary']==step,'scheduled own-side clock differs')
        require(e['counters']['q_updates']==max(0,step//batch-config.learning_start//batch+1)*config.q_reuse
                and e['counters']['q_loss_targets']==e['counters']['q_updates']*batch
                and e['counters']['target_refreshes']==step//config.target_every,'scheduled Q/reuse/refresh clock differs')
        if corrected:require(e['counters']['fresh_records']==step,'scheduled aggregate clock differs')
        eps=.01+.09*max(0.,1-step/config.epsilon_decay);ramp=min(1.,max(0.,(step-config.reservoir_open)/(config.handoff_end-config.reservoir_open)))
        np.testing.assert_allclose([e['epsilon'],e['handoff_probability']],[eps,ramp],rtol=0,atol=1e-15)
        ready=first_sl is not None and step>=first_sl
        require(ready==e['average_ready'],'evaluation readiness differs from label clock')
        checks.append(ready and max(gaps)<=.05)
        expected_sl=0 if first_sl is None else max(0,step//batch-first_sl//batch+1)
        require(e['counters']['supervised_updates']==expected_sl,'scheduled supervised clock differs')
        scheduled.append({'records':step,'average_ready':ready,'average_gaps':gaps,'greedy_q_gaps':qgaps})
        if step==n:
            for head in ('q','average'):
                require(weights[head].keys()==state['backend'][head].keys()
                        and all(torch.equal(v,state['backend'][head][k]) for k,v in weights[head].items()),'final public parameters differ from final learner')
        # At the next episode, public archive parameters still define the actor.
        if step<n:
            end=min(n,(step//batch+1)*batch);rows=log[step//2:end//2]
            for role in range(2):
                scores_role=qvalues[role];ties=(scores_role==scores_role.max()).float()
                eps=torch.tensor(rows['epsilon'][:,role].copy(),dtype=scores_role.dtype)[:,None]
                p=((1-eps)*(ties/ties.sum())[None]+eps*(torch.ones_like(ties)/actions)[None]).double().numpy()
                p[rows['acting'][:,role]==1]=probabilities[role]
                p[rows['acting'][:,role]==2]=1/actions
                p/=p.sum(-1,keepdims=True);selected=rows['actions'][:,role].astype(np.int64)
                np.testing.assert_allclose(rows['behavior_prob'][:,role],p[np.arange(len(rows)),selected],rtol=0,atol=1e-12)
                rng=np.random.default_rng(config.seed+103+role);rng.bit_generator.advance(step//2)
                cdf=p.cumsum(-1);cdf[:,-1]=1.;drawn=(rng.random(len(rows))[:,None]>=cdf).sum(-1)
                np.testing.assert_array_equal(drawn,selected);boundary_actions+=len(rows)
                if config.label_rule=='greedy':
                    slots=np.flatnonzero((label_records>=step)&(label_records<end)&(label_roles==role))
                    if len(slots):np.testing.assert_array_equal(masks[slots],np.repeat((greedy[role]>0)[None],len(slots),axis=0));boundary_targets+=len(slots)
    mean=published_table(state,reservoir,config,log,expected_records,recent,last_sl)
    if config.learner=='tabular':require(sum(mean['published_label_counts_per_role'])==min(last_sl_seen,config.capacity),'published label occupancy differs')
    passed=len(checks)>=5 and all(checks[-5:])
    result=json.loads((root/'result.json').read_text())
    require(result['status']=='complete' and result['records']==n and result['average_final_five_passed']==passed and result['formal_gate_passed']==(passed and config.formal_protocol()),'reported final gate differs')
    require(result['source_sha256']==manifest['source_sha256'] and result['task']==config.task and result['learner']==config.learner and result['seed']==config.seed,'reported result provenance differs')
    require(result['counters']==counts and result['per_role_records']==state['per_role'] and result['best_response_visits']==state['best_response_visits']
            and result['acting_policy_counts']==state['acting_counts'] and result['evaluations']==evaluations,'reported counters/roles/modes/evaluations differ')
    return {'format_version':2,'audit_semantics':'graygo/independent-matrix-audit/2','verified':True,'task':config.task,'learner':config.learner,'seed':config.seed,'records':n,
        'eligibility_version':2 if corrected else 1,'q_eligible_records':n,'excluded_passive_root_records':0,
        'source_sha256':manifest['source_sha256'],'manifest_sha256':hashlib.sha256((root/'run.json').read_bytes()).hexdigest(),
        'auditor_sha256':digest(Path(__file__)),'result_sha256':digest(root/'result.json'),'checkpoint_manifest_sha256':digest(bundle/'manifest.json'),
        'archive_manifests_sha256':archive_hashes,'scheduled_evaluations':scheduled,'label_rule':config.label_rule,
        'retained_replay_records_checked':valid,'retained_label_records_checked':reservoir.size,'retained_executed_label_targets_verified':executed_labels,
        'archive_boundary_actions_reconstructed':boundary_actions,'archive_boundary_greedy_targets_verified':boundary_targets,
        'episode_audit_sha256':h.hexdigest(),'physical_games':n//2,'eligible_labels':seen,'accepted_reservoir_changes':changes,
        'q_updates':q_updates,'supervised_updates':sl_updates,'first_supervised_record':first_sl,
        'formal_protocol':config.formal_protocol(),'average_final_five_passed':passed,
        'final_five_gaps':[e['average_gaps'] for e in evaluations[-5:]],'a2_complete':False,**mean,
        'method':'checksummed model reconstruction; closed-form security gaps; all episode mode/ramp/RNG accounting; exact Algorithm R and retained replay audit',
        'scope':'complete matrix episode action logs support exact executed-label publication reconstruction. Replaced unpublished greedy targets retain explicit bounds; all historical model gradients/frozen greedy sets are not reconstructed'}


def audit(root,*,require_formal=True):
    import fcntl
    root=Path(root).resolve();started=time.monotonic();threads=torch.get_num_threads()
    with (root/'.trainer.lock').open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        try:
            with torch.random.fork_rng(devices=[]),torch.device('cpu'):result=_audit(root,require_formal=require_formal)
        finally:
            torch.set_num_threads(threads);fcntl.flock(lock,fcntl.LOCK_UN)
    result['measured_audit_entry_wall_seconds']=time.monotonic()-started
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_dir');parser.add_argument('--output',required=True);parser.add_argument('--allow-diagnostic',action='store_true')
    parser.add_argument('--time-cap-seconds',type=int,default=3600);args=parser.parse_args()
    if not 0<args.time_cap_seconds<=14400:parser.error('audit watchdog must be positive and at most four hours; aggregate CPU stage budgeting remains separate')
    root=Path(args.run_dir).resolve();output=Path(args.output).resolve()
    if output.is_relative_to(root):parser.error('save audit output outside original training inputs')
    import signal
    def stop(signum,frame):raise TimeoutError('original matrix audit watchdog exhausted; no success receipt written')
    signal.signal(signal.SIGALRM,stop);signal.alarm(args.time_cap_seconds)
    result=audit(root,require_formal=not args.allow_diagnostic);signal.alarm(0);result['audit_watchdog_seconds']=args.time_cap_seconds
    write_report(output,result)
    print(json.dumps(result,indent=2))

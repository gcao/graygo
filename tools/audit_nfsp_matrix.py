"""Read-only independent audit of a completed native NFSP matrix trial.

Use the trial's pinned source on PYTHONPATH. Never instantiate a trainer or
restore/truncate its audit log. Security gaps use closed-form original games.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from graygo.buffers import ReplayBuffer,ReservoirBuffer
from graygo.episode_audit import EPISODE_DTYPE
from graygo.manifest import source_digest,write_report
from graygo.nfsp_training import NFSPMatrixConfig
from graygo.storage import CheckpointStore
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


def audit(root,*,require_formal=True):
    root=Path(root);manifest=json.loads((root/'run.json').read_text());config=NFSPMatrixConfig(**manifest['config'])
    require(manifest['source_sha256']==source_digest(),'load the original pinned source before auditing')
    if require_formal:require(config.formal_protocol(),'diagnostic recipe is not a formal primary trial')
    matrix,value=original_game(config.task);np.testing.assert_array_equal(matrix,manifest['payoffs'])
    store=CheckpointStore(root);bundle=store.latest();meta=store.verify(bundle)
    require(meta['metadata']['source_sha256']==manifest['source_sha256'],'sealed source metadata differs')
    state=torch.load(bundle/'trainer.pt',weights_only=True,map_location='cpu')
    require(state['config']==asdict(config) and meta['step']==config.records,'trial is incomplete or configuration differs')
    counts=state['backend']['counters'];n=config.records;batch=config.batch;actions=matrix.shape[0]
    require(matrix.shape==(actions,actions),'auditor expects the declared square tasks')
    require(counts['fresh_q']==counts['fresh_ordinary']==n and state['per_role']==[n//2,n//2] and state['episodes']==n//2,'own-side/role/game accounting differs')
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
    expected_records=np.full(config.capacity,-1,dtype=np.int64);expected_versions=np.zeros(config.capacity,dtype=np.uint64)
    mode_rng=[np.random.default_rng(config.seed+101),np.random.default_rng(config.seed+102)]
    reservoir_rng=np.random.default_rng(config.seed+105);q_rng=np.random.default_rng(config.seed+106);sl_rng=np.random.default_rng(config.seed+107)
    seen=changes=sl_updates=sl_targets=0;first_sl=None;mode_counts=np.zeros(2,dtype=np.int64);acting_counts=np.zeros(3,dtype=np.int64)
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
    # Reconstruct every scheduled policy and its security gaps from archived weights.
    torch.set_num_threads(1);evaluations=state['evaluations'];expected_steps=list(range(config.eval_every,n+1,config.eval_every))
    if not expected_steps or expected_steps[-1]!=n:expected_steps.append(n)
    require([e['records'] for e in evaluations]==expected_steps,'scheduled evaluations missing or duplicated')
    checks=[]
    for e in evaluations:
        step=e['records'];archive=root/'weights'/f'{step:012d}';store.verify(archive)
        weights=torch.load(archive/'models.pt',weights_only=True,map_location='cpu')
        require(weights['config']==asdict(config),'archived model configuration differs')
        if config.learner=='tabular':q,policy=TabularQ(2,actions),TabularAverage(2,actions)
        else:q,policy=CompactQ(2,0,actions,config.hidden),CompactPolicy(2,0,actions,config.hidden)
        q.load_state_dict(weights['q']);policy.load_state_dict(weights['average'])
        with torch.no_grad():
            features=torch.eye(2);globals=torch.empty(2,0);qvalues=q(features,globals)
            probabilities=policy(features,globals).log_softmax(-1).exp().double().numpy()
        probabilities/=probabilities.sum(-1,keepdims=True)
        scores=qvalues.detach().numpy();greedy=(scores==scores.max(-1,keepdims=True)).astype(float);greedy/=greedy.sum(-1,keepdims=True)
        gaps=security_gaps(matrix,value,probabilities);qgaps=security_gaps(matrix,value,greedy)
        np.testing.assert_allclose(e['average_policies'],probabilities,rtol=0,atol=1e-12)
        np.testing.assert_allclose(e['q_values'],scores,rtol=0,atol=0)
        np.testing.assert_allclose(e['average_gaps'],gaps,rtol=0,atol=1e-12)
        np.testing.assert_allclose(e['greedy_q_gaps'],qgaps,rtol=0,atol=1e-12)
        ready=first_sl is not None and step>=first_sl
        require(ready==e['average_ready'],'evaluation readiness differs from label clock')
        checks.append(ready and max(gaps)<=.05)
        # At the next episode, public archive parameters still define the actor.
        if step<n:
            row=log[step//2]
            for role in range(2):
                acting=int(row['acting'][role]);scores_role=qvalues[role]
                if acting==0:
                    ties=(scores_role==scores_role.max()).float();eps=torch.tensor(float(row['epsilon'][role]),dtype=scores_role.dtype)
                    p=((1-eps)*(ties/ties.sum())+eps*torch.ones_like(ties)/actions).double().numpy()
                elif acting==1:p=probabilities[role].copy()
                else:p=np.ones(actions)/actions
                p/=p.sum();selected=int(row['actions'][role])
                np.testing.assert_allclose(row['behavior_prob'][role],p[selected],rtol=0,atol=1e-12)
                rng=np.random.default_rng(config.seed+103+role);rng.bit_generator.advance(step//2)
                cdf=p.cumsum();cdf[-1]=1.;require(int(np.searchsorted(cdf,rng.random(),side='right'))==selected,'private action draw differs at archived boundary')
    if config.learner=='tabular':
        x,_,_=reservoir.spec.unpack(reservoir.arrays['spatial'][:reservoir.size],reservoir.arrays['globals'][:reservoir.size],reservoir.arrays['legal'][:reservoir.size])
        masks=np.unpackbits(reservoir.arrays['greedy'][:reservoir.size],axis=-1,count=actions,bitorder='little').astype(float);masks/=masks.sum(-1,keepdims=True)
        table=state['backend']['average']['probabilities'].numpy()
        for role in range(2):
            labels=masks[x[:,role]>0]
            expected=labels.mean(0) if len(labels) else np.ones(actions)/actions
            np.testing.assert_allclose(table[role],expected,rtol=0,atol=1e-10)
    passed=len(checks)>=5 and all(checks[-5:])
    result=json.loads((root/'result.json').read_text())
    require(result['status']=='complete' and result['records']==n and result['average_final_five_passed']==passed and result['formal_gate_passed']==(passed and config.formal_protocol()),'reported final gate differs')
    return {'format_version':1,'verified':True,'task':config.task,'learner':config.learner,'seed':config.seed,'records':n,
        'source_sha256':manifest['source_sha256'],'manifest_sha256':hashlib.sha256((root/'run.json').read_bytes()).hexdigest(),
        'episode_audit_sha256':h.hexdigest(),'physical_games':n//2,'eligible_labels':seen,'accepted_reservoir_changes':changes,
        'q_updates':q_updates,'supervised_updates':sl_updates,'first_supervised_record':first_sl,
        'formal_protocol':config.formal_protocol(),'average_final_five_passed':passed,
        'final_five_gaps':[e['average_gaps'] for e in evaluations[-5:]],'a2_complete':False,
        'method':'checksummed model reconstruction; closed-form security gaps; all episode mode/ramp/RNG accounting; exact Algorithm R and retained replay audit'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_dir');parser.add_argument('--output',required=True);parser.add_argument('--allow-diagnostic',action='store_true')
    args=parser.parse_args();result=audit(args.run_dir,require_formal=not args.allow_diagnostic);write_report(args.output,result)
    print(json.dumps(result,indent=2))

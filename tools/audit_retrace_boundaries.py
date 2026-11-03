"""Independent CPU Retrace boundary arithmetic, sampling and optimizer audit.

No production target, sampler, cadence or update function is called. Shared
model definitions and the separately tested observation codec define the inputs.
Float32 target arithmetic is checked with outward intervals, not a fitted
tolerance; optimizer replay uses the actual verified loss labels.
"""
from copy import deepcopy
import argparse
import json
import math
from importlib import metadata
from pathlib import Path
import numpy as np
import torch
from graygo.manifest import source_digest, write_report
from graygo.task_training import NFSPTaskConfig
from graygo.task_adapter import TaskAdapter
from tools.nfsp_resources import require
from tools.retrace_recording import file_sha

VERSION='graygo/independent-retrace-boundary-audit/1'
U=2.**-24


def equal(a,b,path='state'):
    require((isinstance(a,dict) and isinstance(b,dict)) or type(a)==type(b),path+' types differ')
    if isinstance(a,dict):
        require(set(a)==set(b),path+' fields differ')
        for k in a:equal(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        require(len(a)==len(b),path+' lengths differ')
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    elif isinstance(a,torch.Tensor):
        require(a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu()),path+' tensor differs')
    elif isinstance(a,np.ndarray):
        require(a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),path+' array differs')
    else:require(a==b,path+' value differs')


def npdata(value):return {k:v.cpu().numpy().copy() for k,v in value.items()}


def rnd(bounds):
    with np.errstate(over='ignore',invalid='ignore'):
        return (float(np.nextafter(np.float32(bounds[0]),np.float32(-np.inf))),
                float(np.nextafter(np.float32(bounds[1]),np.float32(np.inf))))


def add(a,b):return rnd((a[0]+b[0],a[1]+b[1]))
def sub(a,b):return rnd((a[0]-b[1],a[1]-b[0]))
def mul(a,b):
    terms=[x*y for x in a for y in b]
    return rnd((min(terms),max(terms)))


def reduction(terms):
    # Any float32 summation tree has at most n-1 rounding operations.
    # Products already have outward bounds. gamma_n covers reduction order.
    n=max(0,len(terms)-1);gamma=(n*U)/(1-n*U)
    magnitude=sum(max(abs(a),abs(b)) for a,b in terms)
    return rnd((sum(a for a,b in terms)-gamma*magnitude,
                sum(b for a,b in terms)+gamma*magnitude))


def contains(value,bounds,label):
    require(math.isfinite(float(value)) and bounds[0]<=float(value)<=bounds[1],
            label+' outside independently derived float32 interval')


def policies(q,legal,epsilon):
    require(q.dtype==np.float32 and np.isfinite(q[legal]).all() and (np.abs(q[legal])<=1+1e-6).all(),
            'finite bounded float32 model values required')
    result=[]
    eps=float(np.float32(epsilon));e=(eps,eps)
    for row,mask in zip(q,legal):
        require(mask.any(),'legal nonterminal source required')
        maximum=max(row[mask]);ties=mask & (row==maximum)
        greedy=rnd((1/int(ties.sum()),)*2);uniform=rnd((1/int(mask.sum()),)*2)
        values=[]
        for a in range(len(mask)):
            if not mask[a]:values.append((0.,0.));continue
            first=mul(sub((1.,1.),e),greedy) if ties[a] else (0.,0.)
            values.append(add(first,mul(e,uniform)))
        result.append(values)
    return result


def forward(model,inputs,limit):
    with torch.no_grad():
        return torch.cat([model(**{k:torch.as_tensor(v[i:i+limit],dtype=torch.float32) for k,v in inputs.items()})
                          for i in range(0,len(next(iter(inputs.values()))),limit)]).cpu().numpy()


def target_intervals(qmodel,tmodel,data,epsilon,lam,limit):
    valid=data['valid'];require(valid.dtype==bool and valid.ndim==2 and valid[:,0].all(),'complete valid prefixes required')
    require(not ((~valid[:,:-1])&valid[:,1:]).any(),'noncontiguous sequence')
    lengths=valid.sum(1);require((lengths>=1).all() and valid.shape[1]<=32,'bounded trace lengths required')
    flat={k:v[valid] for k,v in data.items()}
    legal=flat['legal'];require(legal.dtype==bool and legal.ndim==2,'legal action table required')
    require(flat['value_mask'].all() and ((flat['behavior_prob']>0)&(flat['behavior_prob']<=1)).all(),'eligible starts/context and actual behavior probabilities required')
    x={k:flat[k] for k in ('spatial','globals')}
    q=forward(qmodel,x,limit);bar=forward(tmodel,x,limit)
    pi=policies(q,legal,epsilon)
    require(np.isfinite(bar[legal]).all() and (np.abs(bar[legal])<=1+1e-6).all(),'bounded target model required')
    starts=np.r_[0,np.cumsum(lengths)[:-1]]
    last=starts+lengths-1;alive=last[~flat['terminated'][last]]
    boundaries={}
    if len(alive):
        nxt={k:flat['next_'+k][alive] for k in ('spatial','globals')}
        nl=flat['next_legal'][alive];qp=forward(qmodel,nxt,limit);bp=forward(tmodel,nxt,limit)
        pp=policies(qp,nl,epsilon)
        for j,index in enumerate(alive):
            boundaries[int(index)]=reduction([mul(pp[j][a],(float(bp[j,a]),)*2) if nl[j,a] else (0.,0.) for a in range(len(nl[j]))])
    raw=[];mass=[];start_overlap=[];continuation=[]
    lam=float(np.float32(lam))
    for row,(start,length) in enumerate(zip(starts,lengths)):
        chosen=float(bar[start,int(flat['action'][start])]);total=(chosen,chosen);weight=(1.,1.);weight_sum=(1.,1.);overlaps=[]
        for depth in range(int(length)):
            index=int(start+depth);action=int(flat['action'][index])
            require(legal[index,action],'illegal action in recorded trace')
            if depth:
                require(not flat['terminated'][index-1],'trace crosses terminal')
                for name in ('stream','opponent_version','policy_version','mode','supplemental'):
                    require(flat[name][index]==flat[name][index-1],'trace crosses frozen trajectory')
                for name in ('spatial','globals','legal'):
                    require(np.array_equal(flat['next_'+name][index-1],flat[name][index]),'adjacent trace observations differ')
            mu=float(flat['behavior_prob'][index])
            ratio=rnd((pi[index][action][0]/mu,pi[index][action][1]/mu))
            overlap=(max(0.,min(1.,ratio[0])),max(0.,min(1.,ratio[1])))
            overlaps.append(overlap)
            if depth:
                coefficient=mul((lam,lam),overlap)
                weight=mul(weight,coefficient);weight_sum=add(weight_sum,weight)
            if flat['terminated'][index]:
                require(depth==length-1 and not flat['next_legal'][index].any(),'terminal continuation differs')
                future=(0.,0.)
            elif depth+1<length:
                future=reduction([mul(pi[index+1][a],(float(bar[index+1,a]),)*2) if legal[index+1,a] else (0.,0.) for a in range(legal.shape[1])])
            else:future=boundaries[index]
            r=float(flat['reward'][index]);require(-1<=r<=1 and (flat['terminated'][index] or r==0),'terminal-outcome reward required')
            value=float(bar[index,action]);delta=sub(add((r,r),future),(value,value))
            total=add(total,mul(weight,delta))
        raw.append(total);mass.append(weight_sum);start_overlap.append(overlaps[0])
        continuation.append(reduction(overlaps[1:]) if len(overlaps)>1 else (0.,0.))
    return {'raw_targets':raw,'trace_mass':mass,'start_overlap':start_overlap,'continuation_overlap_sum':continuation,
            'sequence_records':lengths,'max_interval_width':max(b-a for a,b in raw)}


def replay_sample(store,spec,rng,count,length):
    a=npdata(store['arrays']);size=store['size']
    eligible=np.flatnonzero(a['value_mask'])
    require(len(eligible)==store['q_size'] and size==len(a['ordinal']),'replay live eligibility differs')
    draws=rng.integers(0,len(eligible),count);starts=eligible[draws]
    indices=np.full((count,length),-1,dtype=np.int64);valid=np.zeros((count,length),dtype=bool)
    for row,start in enumerate(starts):
        index=int(start)
        for col in range(length):
            require(0<=index<size,'replay index outside live store')
            indices[row,col]=index;valid[row,col]=True
            if a['terminated'][index]:break
            nxt=int(a['next_index'][index])
            if nxt<0:break
            require(nxt<size,'replay link outside live store')
            if int(a['ordinal'][nxt])!=int(a['next_ordinal'][index]) or not a['value_mask'][nxt]:break
            require(a['stream'][nxt]==a['stream'][index] and int(a['step'][nxt])==int(a['step'][index])+1
                    and a['opponent_version'][nxt]==a['opponent_version'][index],'invalid stored trajectory link')
            index=nxt
    safe=np.maximum(indices,0);data={}
    for name,value in a.items():
        if name in ('next_index','next_ordinal'):continue
        data[name]=value[safe].copy();data[name][~valid]=False if value.dtype==bool else 0
    data['terminated'][~valid]=True;data['behavior_prob'][~valid]=1
    for prefix in ('','next_'):
        values=spec.unpack(data.pop(prefix+'spatial'),data.pop(prefix+'globals'),data.pop(prefix+'legal'),valid=valid)
        data.update(zip((prefix+'spatial',prefix+'globals',prefix+'legal'),values))
    data.update(indices=indices,valid=valid)
    return data


def supervised_sample(store,spec,rng,count):
    a=npdata(store['arrays']);indices=rng.integers(0,store['size'],count)
    x,g,legal=spec.unpack(a['spatial'][indices],a['globals'][indices],a['legal'][indices])
    greedy=np.unpackbits(a['greedy'][indices],axis=-1,count=spec.actions,bitorder='little').astype(bool)
    require(greedy.any(-1).all() and not (greedy&~legal).any(),'invalid frozen averaging labels')
    targets=(greedy.astype(np.float64)/greedy.sum(-1,keepdims=True)).astype(np.float32)
    return {'spatial':x,'globals':g,'legal':legal,'targets':targets,'records':a['record'][indices].copy(),
            'policy_version':a['policy_version'][indices].copy(),'indices':indices}


def models(adapter,state,cfg):
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):q,average=adapter.models()
    q.load_state_dict(state['q']);average.load_state_dict(state['average'])
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):target,_=adapter.models()
    target.load_state_dict(state['target']);target.eval().requires_grad_(False)
    qo=torch.optim.Adam(q.parameters(),lr=cfg.learning_rate);qo.load_state_dict(state['q_optimizer'])
    ao=torch.optim.Adam(average.parameters(),lr=cfg.learning_rate);ao.load_state_dict(state['average_optimizer'])
    return q,target,average,qo,ao


def trace_accumulate(state,data,details,counters):
    counts=state['counts'].numpy();sums=state['sums'].numpy();n=len(data['valid'])
    require(state['version']=='graygo/nfsp-retrace/1' and counts.shape==(2,2,3,3) and sums.shape==(2,2,3,3,6),'trace schema differs')
    for row in range(n):
        age=counters['fresh_records']-1-int(data['ordinal'][row,0])
        generation=int(data['opponent_version'][row,0])//2
        version=counters['q_updates']+counters['supervised_updates']
        require(age>=0 and generation<=version,'future trace metadata')
        overlap=float(details['start_overlap'][row])
        index=(int(data['mode'][row,0]),int(data['supplemental'][row,0]),
               0 if age<=12800 else 1 if age<=128000 else 2,
               0 if overlap<=.1 else 1 if overlap<.9 else 2)
        counts[index]+=1
        values=[details['trace_mass'][row],details['sequence_records'][row],overlap,version-generation,age,details['continuation_overlap_sum'][row]]
        for i,value in enumerate(values):sums[index+(i,)]+=float(value)


def audit_boundary(item,adapter,cfg):
    expected=deepcopy(item['pre']);c=expected['counters'];b=cfg.batch
    require(c['q_loss_targets']==c['q_updates']*b and c['target_refreshes']==c['fresh_q']//cfg.target_every,'invalid prior Q/refresh clocks')
    require(c['q_updates']==max(0,c['fresh_q']//b-cfg.learning_start//b+1)*cfg.q_reuse,'invalid prior Q cadence')
    require(type(item['count']) is int and 0<=item['ordinary']<=item['q_eligible']<=item['count'] and item['count']>0,'invalid fresh boundary increment')
    c['fresh_records']+=item['count'];c['fresh_q']+=item['q_eligible'];old_ordinary=c['fresh_ordinary'];c['fresh_ordinary']+=item['ordinary']
    require(item['epsilon']==.01+.09*max(0.,1-c['fresh_records']/cfg.epsilon_decay),'current epsilon schedule differs')
    require(item['replay']['insertions']==c['fresh_records'] and item['replay']['q_insertions']==c['fresh_q'],'fresh/store counters differ')
    due=item['q_eligible']>0 and c['fresh_q']%b==0 and c['fresh_q']>=cfg.learning_start
    refresh=item['q_eligible']>0 and c['fresh_q']%cfg.target_every==0
    supervised=c['fresh_ordinary']//b>old_ordinary//b and item['reservoir']['size']>=b
    require(len(item['q_steps'])==(cfg.q_reuse if due else 0) and (item['supervised'] is not None)==supervised,
            'recorded optimizer inventory differs')
    q_rng=np.random.default_rng();q_rng.bit_generator.state=item['q_rng_before']
    sl_rng=q_rng if item['same_sampler'] else np.random.default_rng()
    if sl_rng is not q_rng:sl_rng.bit_generator.state=item['sl_rng_before']
    maximum_width=0.;starts=contexts=0
    for frame in item['q_steps']:
        equal(expected,frame['pre'],'Q/pre')
        data=npdata(frame['data']);sample=replay_sample(item['replay'],adapter.spec,q_rng,b,cfg.trace_length)
        equal(sample,data,'independent replay sample')
        q,target,average,qo,ao=models(adapter,expected,cfg)
        intervals=target_intervals(q,target,data,item['epsilon'],cfg.trace_lambda,cfg.target_batch)
        details=npdata(frame['details']);raw=details['raw_targets'];labels=frame['targets'].numpy()
        require(raw.shape==labels.shape==(b,) and raw.dtype==np.float64 and labels.dtype==np.float32,'loss label dimensions/dtype differ')
        require(np.array_equal(labels,np.clip(raw,-1,1).astype(np.float32)),'projection differs')
        for key in ('raw_targets','trace_mass','start_overlap','continuation_overlap_sum'):
            require(len(details[key])==b,'per-start trace inventory differs')
            for i,value in enumerate(details[key]):contains(value,intervals[key][i],key)
        require(np.array_equal(details['sequence_records'],intervals['sequence_records']),'sequence length differs')
        require(np.array_equal(raw,raw.astype(np.float32).astype(np.float64)),'raw target is not an actual float32 value')
        stats=frame['stats'];require(stats['targets']==b and stats['sequence_records']==int(data['valid'].sum())
            and stats['context_records']==int(data['valid'].sum())-b,'context counted as loss starts')
        require(stats['q_target']=='graygo/nfsp-retrace/1' and stats['sequence_length_max']==int(data['valid'].sum(1).max()),'trace implementation/length statistics differ')
        require(stats['raw_target_min']==float(raw.min()) and stats['raw_target_max']==float(raw.max())
            and stats['projected_targets']==int((raw!=labels).sum()),'raw/projection statistics differ')
        require(stats['trace_mass_min']==float(details['trace_mass'].min()) and stats['trace_mass_max']==float(details['trace_mass'].max()),'mass extrema differ')
        contains(stats['trace_mass_sum'],reduction([(float(x),float(x)) for x in details['trace_mass']]),'mass sum')
        x={key:torch.as_tensor(data[key][:,0],dtype=torch.float32) for key in ('spatial','globals')}
        prediction=q(**x).gather(1,torch.as_tensor(data['action'][:,0],dtype=torch.long)[:,None]).squeeze(1)
        loss=torch.mean(torch.square(prediction-torch.as_tensor(labels)))
        qo.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(q.parameters(),cfg.max_grad_norm,error_if_nonfinite=True);qo.step()
        require(frame['result']['loss']==float(loss.detach()) and frame['result']['gradient_norm']==float(norm),'Q loss/gradient clipping differs')
        expected['q']=q.state_dict();expected['q_optimizer']=qo.state_dict()
        trace_accumulate(expected['trace_diagnostics'],data,details,c)
        equal(expected,frame['post'],'Q/post optimizer')
        c['q_updates']+=1;c['q_loss_targets']+=b
        maximum_width=max(maximum_width,intervals['max_interval_width']);starts+=b;contexts+=stats['context_records']
    if supervised:
        frame=item['supervised'];equal(expected,frame['pre'],'supervised/pre')
        data=npdata(frame['data']);sample=supervised_sample(item['reservoir'],adapter.spec,sl_rng,b)
        equal(sample,data,'independent supervised sample')
        q,target,average,qo,ao=models(adapter,expected,cfg)
        x={key:torch.as_tensor(data[key],dtype=torch.float32) for key in ('spatial','globals')}
        legal=torch.as_tensor(data['legal']);targets=torch.as_tensor(data['targets'])
        logits=average(**x);logp=torch.log_softmax(logits.masked_fill(~legal,-torch.inf),dim=1)
        logp=torch.where(legal,logp,torch.zeros_like(logp));loss=-(targets*logp).sum(1).mean()
        ao.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(average.parameters(),cfg.max_grad_norm,error_if_nonfinite=True);ao.step()
        require(frame['result']['loss']==float(loss.detach()) and frame['result']['gradient_norm']==float(norm),'supervised loss/gradient clipping differs')
        expected['average']=average.state_dict();expected['average_optimizer']=ao.state_dict()
        equal(expected,frame['post'],'supervised/post optimizer')
        c['supervised_updates']+=1;c['supervised_targets']+=b
    if refresh:
        expected['target']=deepcopy(expected['q']);c['target_refreshes']+=1
    equal(expected,item['post'],'boundary/post')
    equal(q_rng.bit_generator.state,item['q_rng_after'],'Q sampler RNG')
    equal(sl_rng.bit_generator.state,item['sl_rng_after'],'SL sampler RNG')
    equal(item['report']['q'],[frame['result'] for frame in item['q_steps']],'reported Q update inventory')
    equal(item['report']['supervised'],None if item['supervised'] is None else item['supervised']['result'],'reported supervised update')
    require(item['report']['target_refreshed']==refresh,'target refresh order/clock differs')
    require(int(item['post']['trace_diagnostics']['counts'].sum())==c['q_loss_targets'],'trace/start total differs')
    return {'fresh_records':c['fresh_records'],'fresh_q':c['fresh_q'],'loss_starts':starts,'context_records':contexts,
            'supervised_update':supervised,'target_refreshed':refresh,'maximum_raw_target_interval_width':maximum_width,
            'all_optimizer_tensors_bit_exact':True,'all_sampler_inputs_and_rngs_exact':True}


def audit(root,*,require_complete=True):
    root=Path(root).resolve();manifest=root/'recording.json';before=file_sha(manifest);header=json.loads(manifest.read_text())
    require(header['version']=='graygo/retrace-boundary-recording/1' and header['status']=='closed'
            and header['source_sha256']==source_digest(),'closed exact-source boundary journal required')
    require(not require_complete or header['complete_predeclared_inventory'],'incomplete predeclared boundary inventory')
    require(header['device']=='cpu' and header['torch_threads']==1,'this verifier qualifies the CPU single-thread path')
    require(torch.get_default_dtype()==torch.float32 and not torch.is_autocast_enabled('cpu')
            and torch.backends.mkldnn.enabled==header['mkldnn_enabled']
            and torch.get_float32_matmul_precision()==header['matmul_precision'],'use the recorded float32 CPU runtime settings')
    require(header['dependencies']=={name:metadata.version(name) for name in ('numpy','torch','scipy')},'recorded numerical dependency versions differ')
    require(sum(r['bytes'] for r in header['boundaries'])==header['recorded_bytes']<=header['max_bytes'],'recording storage accounting differs')
    cfg=NFSPTaskConfig(**header['task_config'])
    require(cfg.q_target=='retrace' and cfg.learner=='neural' and cfg.device=='cpu','neural CPU Retrace task required')
    require(header['observer_sha256']==file_sha(Path(__file__).with_name('retrace_recording.py')),'observer implementation differs')
    from dataclasses import asdict
    equal(header['backend_config'],asdict(cfg.backend_config()),'declared backend configuration')
    require(cfg.storage=='features' and cfg.native_library is None,'this audit qualifies the CPU/features path')
    old_threads=torch.get_num_threads();old_deterministic=torch.are_deterministic_algorithms_enabled();old_warn=torch.is_deterministic_algorithms_warn_only_enabled()
    rows=[]
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(header['deterministic_algorithms'])
        with torch.random.fork_rng(devices=[]),torch.device('cpu'):
            adapter=TaskAdapter(cfg)
            for row in header['boundaries']:
                path=root/row['file'];require(path.parent==root and not path.is_symlink(),'unaliased boundary path required')
                require(file_sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],'boundary payload differs')
                item=torch.load(path,weights_only=True,map_location='cpu')
                result=audit_boundary(item,adapter,cfg);require(file_sha(path)==row['sha256'],'boundary changed during audit')
                require(result['fresh_records']==row['fresh_records'] and result['fresh_q']==row['fresh_q'],'boundary identity differs')
                rows.append(result)
    finally:
        torch.set_num_threads(old_threads);torch.use_deterministic_algorithms(old_deterministic,warn_only=old_warn)
    covered={r['fresh_q'] for r in rows if r['loss_starts']>0}
    complete=set(header['q_boundaries'])<=covered and (not header['first_supervised'] or any(r['supervised_update'] for r in rows))
    require(complete==header['complete_predeclared_inventory'],'reported selection completeness differs')
    require(len(rows)<=len(header['q_boundaries'])+int(header['first_supervised']) and all(a['fresh_records']<b['fresh_records'] for a,b in zip(rows,rows[1:])),'unbounded or duplicate boundary selection')
    require(file_sha(manifest)==before,'recording manifest changed during read')
    return {'version':VERSION,'verified':True,'source_sha256':source_digest(),'auditor_sha256':file_sha(__file__),
            'observer_sha256':header['observer_sha256'],'recording_manifest_sha256':before,'boundaries':rows,
            'input_sampling_reconstructed_for_recorded_boundaries':True,'raw_target_check':'outward float32 intervals with summation-order gamma bounds',
            'production_target_sampler_cadence_or_update_called':False,'all_optimizer_tensors_bit_exact':True,
            'unrecorded_updates_verified':False,'full_training_history_verified':False,'source_admission_qualified':False,
            'a2_complete':False,'execution_authorized':False,
            'scope':'Selected actual unaugmented CPU neural task boundaries: raw stored sampling, frozen targets, gradients, Adam state, average updates, trace counters and refresh order. Other updates and full-case game/actor history need separate evidence.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('recording');parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output).resolve()
    require(not output.exists() and not output.is_relative_to(Path(args.recording).resolve()),'new output outside original recording required')
    result=audit(args.recording);write_report(output,result)
    print(json.dumps({'verified':result['verified'],'boundaries':len(result['boundaries']),'loss_starts':sum(r['loss_starts'] for r in result['boundaries'])}))


if __name__=='__main__':main()

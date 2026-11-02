"""Read-only independent artifact/accounting audit of the bounded journal check."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from graygo.buffers import ReplayBuffer,ReservoirBuffer
from graygo.confidence import canonical_sha
from graygo.manifest import source_digest,write_report
from graygo.storage import CheckpointStore,digest


def require(condition,message):
    if not condition:raise ValueError(message)


def exact(a,b,path='state'):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,rtol=0,atol=0,msg=path);return
    if isinstance(a,dict):
        require(isinstance(b,dict) and a.keys()==b.keys(),path+' keys differ')
        for key in a:exact(a[key],b[key],path+'/'+str(key))
    elif isinstance(a,(list,tuple)):
        require(type(a)==type(b) and len(a)==len(b),path+' sequence differs')
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'/'+str(i))
    else:require(a==b,path+' differs')


def audit(root):
    root=Path(root).resolve();plan=json.loads((root/'plan.json').read_text());report=json.loads((root/'report.json').read_text())
    require(report['verified'] is True and report['purpose']=='diagnostic' and report['formal_scope_approved'] is False and report['a2_complete'] is False,'diagnostic scope changed')
    require(plan['source_sha256']==report['source_sha256']==source_digest(),'use the measured source on PYTHONPATH')
    require(report['manifest']['git_dirty'] is False and report['manifest']['git_commit'].startswith('546d2e2'),'diagnostic source was not its clean committed revision')
    require(report['elapsed_seconds']<=plan['total_time_ceiling_seconds'] and report['aggregate_records']==plan['aggregate_record_cap']==32768,'record/time cap changed')
    states=[];checks={}
    for name in ('full','resumed'):
        directory=root/name;store=CheckpointStore(directory);bundle=store.latest();seal=store.verify(bundle)
        state=torch.load(bundle/'runtime.pt',weights_only=True,map_location='cpu');control=json.loads((bundle/'controller.json').read_text())
        require(seal['metadata']['source_sha256']==source_digest() and seal['metadata']['controller_sha256']==canonical_sha(control),'sealed source/controller identity differs')
        require(control['records']==seal['step']==16384,'final endpoint differs')
        require(control['training_run_sha256']==digest(directory/'run.json'),'run/controller binding differs')
        wrapper=json.loads((directory/'opening-decision.json').read_text());decision=wrapper['body']
        require(wrapper['sha256']==canonical_sha(decision) and decision['kind']=='diagnostic' and decision['status']=='pass' and decision['records']==8192 and decision['formal_evidence'] is False,'opening is not the declared diagnostic control')
        require(control['opening_sha256']==digest(directory/'opening-decision.json'),'final checkpoint omitted its opening control')
        opening=directory/'weights'/'000000008192';store.verify(opening)
        require(decision['archive_manifest_sha256']==digest(opening/'manifest.json'),'opening decision uses another model')
        points=[int(path.name) for path in sorted((directory/'weights').glob('[0-9]*'))]
        require(points==[0,4096,8192,12288,16384] and len(list((directory/'resume').glob('[0-9]*')))==1,'retention/archive inventory differs')
        for point in points:
            path=directory/'weights'/f'{point:012d}';meta=store.verify(path)
            require(set(meta['files'])=={'models.pt'} and meta['step']==point,'public archive contains buffers or a changed clock')
            require(control['archived'][str(point)]==digest(path/'manifest.json'),'checkpoint archive prefix differs')
        counters=state['backend']['counters'];require(counters==report['counters'],'reported optimizer clocks differ')
        require(counters['fresh_q']==counters['fresh_ordinary']==16384 and counters['q_updates']==100 and counters['q_loss_targets']==51200 and counters['target_refreshes']==1,'logical-batch/reuse/refresh accounting differs')
        require(counters['supervised_updates']>0 and counters['supervised_targets']==counters['supervised_updates']*512,'averaging work not exercised')
        replay=ReplayBuffer.load(bundle/'replay.npz');reservoir=ReservoirBuffer.load(bundle/'reservoir.npz')
        require(replay.insertions==replay.size==16384 and reservoir.seen==reservoir.size==report['final_reservoir_seen'],'buffer clock differs')
        require(reservoir.opening_record==8192 and (reservoir.arrays['record'][:reservoir.size]>=8192).all(),'early labels were inserted')
        require(state['collector']['records']==16384 and state['collector']['metrics']['per_role_records']==[8192,8192],'collector role balance differs')
        require(state['config']['device']=='cuda' and (state['config']['width'],state['config']['blocks'],state['config']['batch'],state['config']['q_reuse'])==(64,8,512,4),'full-model CUDA/R4 configuration differs')
        checks[name]={'final_checkpoint_manifest_sha256':digest(bundle/'manifest.json'),'opening_sha256':control['opening_sha256'],
                      'public_archive_points':points,'labels':reservoir.seen,'buffer_array_bytes':sum(a.nbytes for buffer in (replay,reservoir) for a in buffer.arrays.values())}
        states.append((state,replay,reservoir))
    a,b=states
    for item in (a[0],b[0]):item.pop('timings',None)
    exact(a[0],b[0])
    for x,y in ((a[1],b[1]),(a[2],b[2])):
        for key,values in x.arrays.items():np.testing.assert_array_equal(values,y.arrays[key])
    return {'format_version':1,'verified':True,'purpose':'diagnostic','source_sha256':source_digest(),'auditor_sha256':digest(Path(__file__)),
            'artifact_sha256':{name:digest(root/name) for name in ('plan.json','report.json')},'checks':checks,
            'exact_runtime_tensor_scalar_dtype_and_rng_match':True,'exact_complete_buffer_match':True,'early_labels_absent':True,
            'scope':'sealed artifact, clock, full-state, label and retention audit of a bounded diagnostic; no A2, funding, pilot or campaign claim'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_dir');parser.add_argument('--output',required=True);args=parser.parse_args()
    result=audit(args.run_dir);write_report(args.output,result);print(json.dumps(result,sort_keys=True))

"""Read-only accounting audit of a completed bounded real-board profile.

Use the measured revision's src directory on PYTHONPATH. No trainer is created
and no training checkpoint, evidence log or RNG is changed.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from graygo.buffers import ReplayBuffer,ReservoirBuffer
from graygo.manifest import source_digest,write_report
from graygo.storage import CheckpointStore



def require(condition,message):
    if not condition:raise ValueError(message)


def audit(root):
    root=Path(root)
    report=json.loads((root/'report.json').read_text());run=json.loads((root/'run.json').read_text());plan=json.loads((root/'plan.json').read_text())
    require(report['plan']==plan and run['source_sha256']==plan['source_sha256']==source_digest(), 'board profile evidence check 2 failed')
    require(isinstance(run['git_commit'],str) and len(run['git_commit'])==40 and run['git_dirty'] is False, 'board profile evidence check 3 failed')
    require(report['status']=='complete' and not any(report[k] for k in ('full_campaign_price_established','a2_complete','promotion_permitted','formal_learning_run')), 'board profile evidence check 4 failed')
    require(report['elapsed_seconds']<=plan['total_time_ceiling_seconds']<=3600, 'board profile evidence check 5 failed')
    checks={}
    for name,result in report['phases'].items():
        path=Path(result['checkpoint']);seal=CheckpointStore.verify(path)
        state=torch.load(path/'runtime.pt',weights_only=True,map_location='cpu')
        replay=ReplayBuffer.load(path/'replay.npz');reservoir=ReservoirBuffer.load(path/'reservoir.npz')
        n=plan['records_per_phase'];c=result['counters'];collector=state['collector'];config=plan['phases'][name]
        require(state['config']==config and c==state['backend']['counters'], 'board profile evidence check 12 failed')
        require(seal['step']==collector['records']==c['fresh_q']==c['fresh_ordinary']==replay.insertions==replay.size==n, 'board profile evidence check 13 failed')
        require(collector['metrics']==result['metrics'] and collector['metrics']['physical_rounds']*2==n, 'board profile evidence check 14 failed')
        require(collector['metrics']['per_role_records']==[n//2,n//2], 'board profile evidence check 15 failed')
        expected_q=(n//512-4096//512+1)*config['q_reuse']
        require(c['q_updates']==expected_q and c['q_loss_targets']==expected_q*512 and c['target_refreshes']==n//12800, 'board profile evidence check 17 failed')
        require(c['supervised_targets']==c['supervised_updates']*512, 'board profile evidence check 18 failed')
        require(reservoir.seen==result['reservoir_seen'] and reservoir.size==result['reservoir_size'], 'board profile evidence check 19 failed')
        require(replay.capacity==reservoir.capacity==500000, 'board profile evidence check 20 failed')
        require(sum(a.nbytes for buffer in (replay,reservoir) for a in buffer.arrays.values())==result['buffer_array_bytes']==203000000, 'board profile evidence check 21 failed')
        require(replay.spec.to_dict()==reservoir.spec.to_dict()==result['native_schema']==seal['metadata']['native_schema'], 'board profile evidence check 22 failed')
        require(seal['metadata']['source_sha256']==source_digest(), 'board profile evidence check 23 failed')
        np.testing.assert_array_equal(np.sort(replay.arrays['ordinal'][:n]),np.arange(n,dtype=np.uint64))
        require(collector['started']==collector['completed']+sum(e is not None for e in collector['active']), 'board profile evidence check 25 failed')
        require(sum(collector['metrics']['endings'].values())==sum(collector['metrics']['outcomes'])==collector['completed'], 'board profile evidence check 26 failed')
        require(sum(collector['metrics']['episode_lengths'])==collector['completed'], 'board profile evidence check 27 failed')
        require(result['restore_record_match'] and all(result['covered'].values()) and len(result['evaluation_pairs'])==4, 'board profile evidence check 28 failed')
        require(sum(f['bytes'] for f in seal['files'].values())==result['checkpoint_payload_bytes'], 'board profile evidence check 29 failed')
        require(result['peak_cuda_allocated_bytes']<=result['peak_cuda_reserved_bytes']<12*1024**3, 'board profile evidence check 30 failed')
        if name=='closed':require(reservoir.seen==c['supervised_updates']==collector['metrics']['acting_counts']['average']==0, 'board profile evidence check 31 failed')
        else:
            require(reservoir.seen==sum(collector['metrics']['best_response_visits'])>512, 'board profile evidence check 33 failed')
            require(c['supervised_updates']>0 and collector['metrics']['acting_counts']['average']>0, 'board profile evidence check 34 failed')
        for pair in result['evaluation_pairs']:
            require(len(pair['pair']['subject'])==2, 'board profile evidence check 36 failed')
            for game in pair['pair']['subject']:
                require(game['black_score_units']+game['white_score_units']==324, 'board profile evidence check 38 failed')
                require(game['termination'] in ('mutual_pass','round_limit') and 1<=game['rounds']<=200, 'board profile evidence check 39 failed')
        checks[name]={'records':n,'q_updates':expected_q,'supervised_updates':c['supervised_updates'],'paired_evaluation_games':8,
                      'checkpoint_manifest_sha256':hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest()}
    result={'verified':True,'profile_source_sha256':source_digest(),'checks':checks,
            'artifact_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('plan.json','run.json','report.json')},
            'scope':'read-only accounting, artifact, decoder, score and resource audit; no learner-strength or full-campaign claim'}
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=audit(args.run_dir)
    write_report(args.output,result)
    print(json.dumps(result,sort_keys=True))

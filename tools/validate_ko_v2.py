"""Bounded v2 integration check using an unchanged pretrained network."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from graygo.board_runtime import BoardRuntimeConfig,BoardNFSPRuntime
from graygo.engine import Rules
from graygo.manifest import write_report,source_digest
from graygo.storage import digest
from tools.task_pipeline_parity import state_sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--records',type=int,default=400);args=parser.parse_args()
    if not 2<=args.records<512 or args.records%2:raise ValueError('validation collects 2..510 records without an optimizer update')
    root=args.run_dir.resolve();root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    cfg=BoardRuntimeConfig(rules=Rules(ko='full-history'),envs=2,initial_weights_archive=str(args.archive),reservoir_open=0,handoff_end=2,device='cpu')
    runtime=BoardNFSPRuntime(cfg);before={name:state_sha(getattr(runtime.backend,name).state_dict()) for name in ('q','average')}
    source=torch.load(args.archive/'models.pt',map_location='cpu',weights_only=True)
    assert before=={name:state_sha(source[name]) for name in before}
    result=runtime.run_profile(args.records,time_cap_seconds=600);assert result['status']=='complete'
    assert runtime.backend.counters.q_updates==runtime.backend.counters.supervised_updates==0
    assert before=={name:state_sha(getattr(runtime.backend,name).state_dict()) for name in before}
    checkpoint=runtime.save(root/'resume');restored=BoardNFSPRuntime(cfg);restored.restore(checkpoint)
    assert [None if e is None else e.state.to_dict() for e in runtime.collector.active]==[None if e is None else e.state.to_dict() for e in restored.collector.active]
    # An observation packet intentionally omits full-game ko history and is
    # suitable for NN decoding, not for reconstructing a playable game.
    for episode in restored.collector.active:
        if episode is not None:assert episode.state.rules.ko=='full-history' and episode.state.seen_boards
    report={'status':'complete','rule_version':'graygo/2','source_sha256':source_digest(),'source_archive':str(args.archive.resolve()),
        'source_archive_sha256':digest(args.archive/'manifest.json'),'weights_sha256':before,'network_weights_unchanged':True,
        'architecture':asdict(cfg.model_config()),'spatial_channels':30,'global_features':14,'records':runtime.records,
        'optimizer_updates':0,'fresh_replay':True,'replay_rule_version':runtime.spec.to_dict()['rule_version'],
        'checkpoint':str(checkpoint),'collector_ko_history_restored':True,'metrics':runtime.collector.metrics,
        'promotion_permitted':False,'production_training_reconfigured':False}
    write_report(root/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='metrics'},indent=2))


if __name__=='__main__':main()

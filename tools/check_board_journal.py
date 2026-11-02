"""Bounded full-model hardware/lifecycle diagnostic; no production approval."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from graygo.board_journal import BoardMilestoneJournal,BoardJournalPlan,_equal
from graygo.board_runtime import BoardNFSPRuntime,BoardRuntimeConfig
from graygo.evaluation import FrozenNeuralPolicy
from graygo.manifest import source_digest,write_report,make_manifest
from graygo.storage import digest


def check(root,device,seconds):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():raise FileExistsError('diagnostic exists; inspect its original process/output')
    if not 0<seconds<=600:raise ValueError('bounded total diagnostic ceiling at most ten minutes required')
    config=BoardRuntimeConfig(device=device,seed=17,q_reuse=4,reservoir_open=8192,handoff_end=12288)
    plan=BoardJournalPlan('diagnostic',16384,(0,4096,8192,12288,16384),91)
    contract={'purpose':'diagnostic','source_sha256':source_digest(),'runtime':config.plan(),'journal':plan.to_dict(),
              'aggregate_record_cap':32768,'total_time_ceiling_seconds':seconds,'scope':'two bounded full-model lifecycle trajectories; no A2/pilot/funding certificate'}
    write_report(root/'plan.json',contract);started=time.monotonic()
    def drive(runtime,journal,end):
        while runtime.records<end:
            if time.monotonic()-started>=seconds:raise TimeoutError('diagnostic resource ceiling reached; no completion assertion')
            journal.capture_if_due()
            count=min(2*runtime.config.envs,runtime.backend.counters.until_q_boundary(runtime.backend.config),end-runtime.records,journal.next_boundary()-runtime.records)
            journal.before_advance(count);runtime.advance(count,opening_enabled=journal.label_insertion_allowed())
        journal.capture_if_due()
    full=BoardNFSPRuntime(config);fj=BoardMilestoneJournal(root/'full',full,plan)
    drive(full,fj,8192);closed=full.reservoir.seen;checkpoint=fj.checkpoint();fj.diagnostic_opening('pass');fj.checkpoint()
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all() if device.startswith('cuda') else []
    FrozenNeuralPolicy.from_archive(fj._archive_path(8192),representation='greedy_q',device=device)
    torch.testing.assert_close(torch.get_rng_state(),cpu,rtol=0,atol=0)
    for a,b in zip(cuda,torch.cuda.get_rng_state_all()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    drive(full,fj,16384);fj.finalize();expected=deepcopy(full.state_dict())
    partial=BoardNFSPRuntime(config);pj=BoardMilestoneJournal(root/'resumed',partial,plan)
    drive(partial,pj,8192);path=pj.checkpoint();pj.diagnostic_opening('pass')
    saved_control=json.loads((path/'controller.json').read_text())
    if saved_control['opening_sha256'] is not None:raise ValueError('diagnostic did not exercise the post-decision/pre-checkpoint crash interval')
    restored=BoardNFSPRuntime(config);rj=BoardMilestoneJournal(root/'resumed',restored,plan,resume=True);rj.restore()
    if not rj.label_insertion_allowed():raise ValueError('fixed-opening control overlay was not recovered')
    drive(restored,rj,16384);rj.finalize();actual=restored.state_dict()
    for state in (expected,actual):state.pop('timings',None)
    if not _equal(expected,actual):raise ValueError('resumed complete model/optimizer/collector/RNG state differs')
    for a,b in ((full.replay,restored.replay),(full.reservoir,restored.reservoir)):
        for key,value in a.arrays.items():np.testing.assert_array_equal(value,b.arrays[key])
    if closed or not restored.reservoir.seen:raise ValueError('label boundary was not exercised')
    manifest=make_manifest(seed=config.seed,command={'check_board_journal':contract},rules=config.plan()['rules'])
    report={'verified':True,'purpose':'diagnostic','manifest':manifest,'source_sha256':source_digest(),'records_per_trajectory':16384,
        'aggregate_records':32768,'elapsed_seconds':time.monotonic()-started,'device':device,'logical_batch':config.batch,'q_reuse':config.q_reuse,
        'closed_reservoir_seen':closed,'final_reservoir_seen':restored.reservoir.seen,'counters':actual['backend']['counters'],
        'exact_runtime_state_match':True,'exact_full_buffer_match':True,'public_weight_load_rng_neutral':True,'pending_checkpoint_overlay_recovered':True,
        'permanent_archive_points':list(plan.milestones),'final_resume_generations_each':1,'formal_scope_approved':False,'a2_complete':False,
        'full_run_manifest_sha256':digest(root/'full/run.json'),'resumed_run_manifest_sha256':digest(root/'resumed/run.json')}
    write_report(root/'report.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--device',default='cuda')
    parser.add_argument('--time-cap-seconds',type=float,default=300);args=parser.parse_args()
    result=check(args.run_dir,args.device,args.time_cap_seconds)
    print(json.dumps({k:v for k,v in result.items() if k!='manifest'},sort_keys=True))

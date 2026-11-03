"""Run one frozen, bounded neural Retrace profile case; never resumes."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import signal
import time
from graygo.manifest import source_digest,write_report
from graygo.task_training import NFSPTaskConfig,NFSPTaskTrainer
from tools.retrace_recording import BoundaryRecorder,file_sha
from tools.task_pipeline_parity import state_sha
from tools.nfsp_resources import require


def run(specification,expected_sha,case_id):
    specification=Path(specification).resolve()
    require(file_sha(specification)==expected_sha,'exact preregistered profile specification required')
    spec=json.loads(specification.read_text())
    require(spec['version']=='graygo/retrace-boundary-prefix-allocation/1'
            and spec['source_sha256']==source_digest(),'exact neural prefix source required')
    for name in ('producer','observer','auditor'):
        path=Path(__file__).with_name(spec[name])
        require(file_sha(path)==spec[name+'_sha256'],'frozen '+name+' differs')
    cases={row['id']:row for row in spec['cases']};require(case_id in cases,'original case ID required')
    row=cases[case_id];cfg=NFSPTaskConfig(**row['config']);root=Path(row['root']).resolve()
    require(not root.exists() and cfg.purpose=='profile' and cfg.task=='delayed' and cfg.learner=='neural'
        and cfg.device=='cpu' and cfg.q_target=='retrace' and cfg.q_reuse==4 and cfg.trace_length==32
        and cfg.trace_lambda==.95 and cfg.target_batch==512 and cfg.horizons==(50,100,200)
        and cfg.equivalent_actions==40 and cfg.hidden==64 and cfg.capacity==500000,
        'fresh unchanged production-shaped CPU neural target prefix required')
    stopped=[]
    for kind in (signal.SIGTERM,signal.SIGINT):signal.signal(kind,lambda number,frame:stopped.append(number))
    started=time.monotonic();root.mkdir(parents=True)
    trainer=NFSPTaskTrainer(cfg,root/'trial')
    shutil.copytree(trainer.store.latest(),root/'initial-checkpoint')
    initial={'q':trainer.backend.q.state_dict(),'target':trainer.backend.target.state_dict(),
             'average':trainer.backend.average.state_dict(),'rng':{k:v.bit_generator.state for k,v in trainer.rng.items()}}
    write_report(root/'initialization.json',{'source_sha256':source_digest(),'seed':cfg.seed,
        'initial_models_and_private_rng_sha256':state_sha(initial),'counters':vars(trainer.backend.counters),
        'q_optimizer_empty':not trainer.backend.q_optimizer.state,'average_optimizer_empty':not trainer.backend.average_optimizer.state,
        'config':cfg.plan()})
    context=(BoundaryRecorder(trainer.backend,root/'recording',cfg,q_boundaries=tuple(spec['q_boundaries']),
        first_supervised=row['phase']=='post',max_bytes=spec['max_recording_bytes']) if row['observed'] else nullcontext())
    with context:result=trainer.run(stop_requested=lambda:bool(stopped))
    summary={'version':1,'case_id':case_id,'phase':row['phase'],'observed':row['observed'],
        'status':result['status'],'records':result['records'],'counters':result['counters'],
        'source_sha256':source_digest(),'elapsed_whole_entry_seconds':time.monotonic()-started,
        'complete':result['status']=='complete' and result['records']==cfg.records,
        'actual_average_work':result['counters']['supervised_updates']>0 and result['acting_policy_counts']['average']>0,
        'new_scientific_training_records':0,'new_accelerator_jobs':0,
        'complete_pipeline_price_qualified':False,'a2_complete':False,'execution_authorized':False}
    write_report(root/'summary.json',summary)
    print(json.dumps(summary))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--specification',required=True);parser.add_argument('--expected-specification-sha',required=True)
    parser.add_argument('--case-id',required=True);args=parser.parse_args()
    run(args.specification,args.expected_specification_sha,args.case_id)


if __name__=='__main__':main()

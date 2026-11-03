"""Run only declared bounded neural CUDA diagnostics with explicit FP32 settings."""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time
import torch
from graygo.board_runtime import configure_runtime
from graygo.manifest import source_digest,write_report
from graygo.storage import digest
from graygo.task_training import NFSPTaskConfig
from tools.nfsp_resources import require


def profile_arguments(spec,task,repo):
    require(spec['tasks']==['gray-pure','gray-mixed'] and task in spec['tasks'],'only declared bounded Gray Go families are selected')
    require(spec['device']=='cuda:0' and spec['learner']=='neural' and spec['seed']==0 and spec['q_reuse']==4,'exact original neural CUDA profile required')
    require(spec['records_per_profile']==32768 and spec['profile_case_loop_cap_seconds']==300 and spec['new_formal_a2_records']==0,'only bounded cost diagnostics are selected')
    require(spec['deterministic'] is True and spec['cudnn_tf32'] is False and spec['matmul_tf32'] is False,'declared deterministic full-FP32 execution required')
    root=Path(spec['root']);require(not root.is_absolute() and root.parts[0]=='runs' and '..' not in root.parts,'profile roots must remain in project runs')
    expected=[{'phase':phase,'config':NFSPTaskConfig(time_cap_seconds=300,task=task,learner='neural',seed=0,records=32768,eval_every=16384,q_reuse=4,device='cuda:0',purpose='profile',reservoir_open=5000000 if phase=='pre' else 0,handoff_end=7000000 if phase=='pre' else 1000).plan()} for phase in ('pre','post')]
    require(spec['profiles'][task]==expected,'declared profile recipe differs from the fixed complete learner')
    directory=Path(repo)/root/'profiles'/task
    argv=[str(Path(repo)/spec['profiler']),'--run-dir',str(directory),'--expected-source-sha',spec['learner_source_sha256'],'--records','32768','--device','cuda:0','--storage','features','--case-minutes','5','--task',task,'--learners','neural','--reuses','4']
    return directory,argv


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--repo',required=True);parser.add_argument('--allocation',required=True);parser.add_argument('--allocation-sha',required=True);parser.add_argument('--task',required=True);args=parser.parse_args(argv)
    repo=Path(args.repo).resolve();allocation=Path(args.allocation).resolve();spec=json.loads(allocation.read_text());require(digest(allocation)==args.allocation_sha,'CUDA diagnostic allocation changed')
    require(digest(Path(__file__))==spec['wrapper_sha256'] and source_digest()==spec['learner_source_sha256'],'wrong wrapper or imported learner package')
    require(digest(repo/spec['profiler'])==spec['profiler_sha256'],'pinned complete-learner profiler changed')
    directory,profile_argv=profile_arguments(spec,args.task,repo);require(not directory.exists(),'original CUDA profile root already exists; preserve it')
    require(torch.cuda.is_available(),'selected CUDA device unavailable');torch.set_num_threads(1);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    settings=configure_runtime('cuda:0',True);require(settings['deterministic_algorithms'] and settings['cudnn_deterministic'] and not settings['cudnn_benchmark'] and not settings['cudnn_tf32'] and not settings['matmul_tf32'],'selected runtime differs from full-FP32 descriptor')
    device=torch.device('cuda:0');properties=torch.cuda.get_device_properties(device);query=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader','--id=0'],text=True).strip();require(query.split(',',1)[0].strip()==spec['gpu_uuid'],'selected physical GPU differs')
    directory.mkdir(parents=True);runtime={'version':1,'allocation_sha256':args.allocation_sha,'wrapper_sha256':digest(Path(__file__)),'profiler_sha256':spec['profiler_sha256'],'source_sha256':source_digest(),'settings':settings,'gpu_query':query,'gpu_name':properties.name,'gpu_capability':list(torch.cuda.get_device_capability(device)),'gpu_memory_bytes':properties.total_memory,'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),'eager_frozen_actors':True,'native_library':None,'new_formal_a2_records':0}
    write_report(directory/'execution-runtime.json',runtime);torch.cuda.reset_peak_memory_stats(device);started=time.monotonic();previous=sys.argv
    try:sys.argv=profile_argv;runpy.run_path(str(repo/spec['profiler']),run_name='__main__')
    finally:
        sys.argv=previous;torch.cuda.synchronize(device);write_report(directory/'execution-runtime-final.json',{'initial_runtime':runtime,'measured_profile_entry_seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),'source_sha256':source_digest()})


if __name__=='__main__':main()

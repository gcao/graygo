import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/gcao/proj/graygo/runs/branch-grading-v2-validation-9f444ad')
COMMAND=['/home/gcao/proj/graygo/.venv/bin/python', '/home/gcao/proj/graygo/.worktrees/branch-grading-v2-9f444ad/tools/validate_branch_grading_v2.py', '--repo', '/home/gcao/proj/graygo', '--specification', '/home/gcao/proj/graygo/.worktrees/branch-grading-v2-9f444ad/docs/experiments/nfsp-branch-grading-v2-validation-spec.json', '--expected-specification-sha', '1ba07c682036c6ffc2c0fdbb8185d3ef52766f0c91c74c384c303db5b5a77d53', '--output', '/home/gcao/proj/graygo/runs/branch-grading-v2-validation-9f444ad/validation.json']
ENV=dict(os.environ, PYTHONPATH='/home/gcao/proj/graygo/.worktrees/branch-grading-v2-9f444ad/src:/home/gcao/proj/graygo/.worktrees/branch-grading-v2-9f444ad', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(d):
 path=ROOT/'job.json';tmp=ROOT/'.job.json.tmp';tmp.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
start=time.monotonic()
with (ROOT/'validation.stdout').open('w') as out,(ROOT/'validation.stderr').open('w') as err:
 child=subprocess.Popen(COMMAND,env=ENV,stdout=out,stderr=err,start_new_session=True)
 proc=Path('/proc',str(child.pid))
 job={'version':1,'status':'running_read_only_v2_production_validation','original_driver_pid':os.getpid(),'original_validation_pid':child.pid,'original_start_ticks':(proc/'stat').read_text().split(') ',1)[1].split()[19],'original_cmdline_sha256':sha(proc/'cmdline'),'command':COMMAND,'driver_sha256':sha(__file__),'specification_sha256':'1ba07c682036c6ffc2c0fdbb8185d3ef52766f0c91c74c384c303db5b5a77d53','new_accelerator_jobs':0,'new_training_records':0,'new_protocol_training_selected':False,'a2_complete':False}
 save(job)
 try:rc=child.wait(timeout=120)
 except subprocess.TimeoutExpired:
  child.kill();rc=child.wait();job['time_cap_reached']=True
 job.update(returncode=rc,measured_process_seconds=time.monotonic()-start,status='verified_all_ninety_archives_against_v2_production_evaluator' if rc==0 else 'read_only_v2_validation_failed')
 if (ROOT/'validation.json').exists():job['validation_sha256']=sha(ROOT/'validation.json')
 save(job)

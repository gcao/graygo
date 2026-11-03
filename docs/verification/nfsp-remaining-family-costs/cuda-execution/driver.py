from pathlib import Path
from decimal import Decimal
import fcntl,hashlib,json,os,shutil,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/nfsp-bounded-cuda-profiles-7740635';allocation=frozen/'docs/experiments/nfsp-bounded-cuda-price-profile-allocation.json';started=time.monotonic();child=None;current_row=None
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
a=load(allocation);assert sha(allocation)=='a9f20d4cb233e9d0185834260e6b09391019ec080ca9934b88b82a6d18ede916';assert sha(frozen/a['wrapper'])==a['wrapper_sha256'] and sha(repo/a['profiler'])==a['profiler_sha256'];assert sha(repo/a['cpu_comparison_profile_root']/'job.json')==a['cpu_comparison_profile_job_sha256'];assert sha(repo/a['cpu_comparison_audit_root']/'job.json')==a['cpu_comparison_audit_job_sha256'];assert load(repo/a['cpu_comparison_audit_root']/'job.json')['status']=='verified_all_twelve_original_profiles';assert shutil.disk_usage(repo).free>28*1024**3
query=lambda:subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
assert not query(),'GPU already has another compute process; preserve this unstarted allocation'
body={'version':1,'status':'profiling_declared_cuda_families','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'driver_sha256':sha(__file__),'source_sha256':a['learner_source_sha256'],'gpu_uuid':a['gpu_uuid'],'groups':{},'max_concurrent_gpu_processes':1,'new_formal_a2_records':0,'new_tabular_control_records':0,'source_admission_qualified':False,'project_scope_selected':False,'no_observation_timeout_restart':True}
def save():
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
def stop(*args):raise TimeoutError('bounded CUDA diagnostic watchdog or external stop')
def cleanup():
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

def observe(task,row):
 directory=root/'profiles'/task
 if (directory/'execution-runtime.json').exists():
  runtime=load(directory/'execution-runtime.json');settings=runtime['settings'];assert runtime['source_sha256']==a['learner_source_sha256'] and settings['device']=='cuda:0' and settings['deterministic_algorithms'] and settings['cudnn_deterministic'] and not settings['cudnn_benchmark'] and not settings['cudnn_tf32'] and not settings['matmul_tf32'];row['explicit_fp32_runtime_verified']=True;row['execution_runtime_sha256']=sha(directory/'execution-runtime.json')
 if (directory/'plan.json').exists():
  plan=load(directory/'plan.json');assert plan['source_sha256']==a['learner_source_sha256'] and plan['driver_sha256']==a['profiler_sha256'] and plan['cases']==a['profiles'][task];row['emitted_plan_verified']=True
 if (directory/'state.json').exists():row['state']=load(directory/'state.json')
 if (directory/'profile.json').exists():row['completed_profiles']=[{k:r.get(k) for k in ['phase','status','records','seconds_including_setup','counters','acting_policy_counts']} for r in load(directory/'profile.json')['results']]
signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.alarm(a['outer_watchdog_seconds']-20)
with (root/'.driver.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save()
 try:
  env=dict(os.environ,PYTHONPATH=str(repo/a['source_package_path'])+os.pathsep+str(frozen),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  for task in a['tasks']:
   assert not query(),'another compute process appeared before the next family';assert not (root/'profiles'/task).exists()
   cmd=[str(repo/'.venv/bin/python'),str(frozen/a['wrapper']),'--repo',str(repo),'--allocation',str(allocation),'--allocation-sha',sha(allocation),'--task',task];row={'command':cmd,'started_monotonic':time.monotonic(),'allocation_start_unix_seconds':str(time.time()),'physical_device_ids':[a['gpu_uuid']]};body['groups'][task]=row;current_row=row
   with (root/(task+'.stdout')).open('wb') as out,(root/(task+'.stderr')).open('wb') as err:child=subprocess.Popen(cmd,cwd=frozen,env=env,stdout=out,stderr=err,start_new_session=True)
   row['original_pid']=child.pid;save();last=0
   while child.poll() is None:
    now=time.monotonic();elapsed=now-row['started_monotonic']
    if elapsed>=a['per_family_complete_process_ceiling_seconds']-30 and not row.get('budget_stop_requested'):os.killpg(child.pid,signal.SIGTERM);row['budget_stop_requested']=True
    if elapsed>=a['per_family_complete_process_ceiling_seconds']-1:os.killpg(child.pid,signal.SIGKILL);row['hard_budget_kill_requested']=True
    if now-last>=5:observe(task,row);save();last=now
    time.sleep(.25)
   row.update(returncode=child.returncode,measured_process_upper_seconds=time.monotonic()-row['started_monotonic'],allocation_end_unix_seconds=str(time.time()));row['gpu_equipped_interval_seconds']=str(Decimal(row['allocation_end_unix_seconds'])-Decimal(row['allocation_start_unix_seconds']));observe(task,row);row['profile_complete']=(child.returncode==0 and row.get('state',{}).get('status')=='complete' and len(row.get('completed_profiles',[]))==2 and all(r['status']=='complete' and r['records']==32768 for r in row['completed_profiles']));save()
   if child.returncode!=0:body['status']='original_cuda_profile_failure_preserved';break
  else:body['status']='original_cuda_families_terminal_unreviewed'
  body['diagnostic_records']=sum(r['records'] for row in body['groups'].values() for r in row.get('completed_profiles',[]));assert body['diagnostic_records']<=a['aggregate_diagnostic_record_cap']
 except BaseException as exc:body.update(status='failed_or_stopped',error=repr(exc))
 finally:
  cleanup();body['measured_driver_wall_seconds']=time.monotonic()-started;body['gpu_equipped_intervals_seconds']=str(sum((Decimal(r.get('gpu_equipped_interval_seconds','0')) for r in body['groups'].values()),Decimal(0)));save();signal.alarm(0)

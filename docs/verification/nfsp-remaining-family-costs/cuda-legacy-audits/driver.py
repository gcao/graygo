from pathlib import Path
from contextlib import ExitStack
import fcntl,hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/nfsp-cuda-legacy-profile-audit-a003824';allocation=frozen/'docs/experiments/nfsp-cuda-and-legacy-control-profile-audit-allocation.json';started=time.monotonic();active={}
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
a=load(allocation);assert sha(allocation)=='95d37f6e01f83b4e707fbf907cbf87a2c29ec0b4c22f1b531ff7b213b4d4112c';assert sha(repo/a['cuda_profile_job'])==a['cuda_profile_job_sha256'] and sha(repo/a['cuda_profile_allocation'])==a['cuda_profile_allocation_sha256']
for c in a['cases'].values():assert sha(repo/c['auditor'])==c['auditor_sha256'] and sha(repo/c['root']/'run.json')==c['manifest_sha256'] and sha(repo/c['root']/'result.json')==c['result_sha256']
body={'version':1,'status':'auditing_original_cuda_and_legacy_profiles','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'driver_sha256':sha(__file__),'commands':[],'cases':{},'new_training_records':0,'new_scored_evaluation_games':0,'new_accelerator_jobs':0,'execution_authorized':False}
def save():
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
def snapshot():
 paths={repo/a['cuda_profile_job'],repo/a['cuda_profile_allocation']}
 for c in a['cases'].values():
  directory=repo/c['root'];paths.update(p for p in directory.rglob('*') if p.is_file() and not any(part.startswith('.') for part in p.relative_to(directory).parts));paths.update(directory.parent/name for name in ['plan.json','profile.json','state.json']);paths.update((directory.parent/'budgets').glob('*.json'))
 return {str(p):sha(p) for p in sorted(paths)}
def alarm(*args):raise TimeoutError('original CUDA/legacy audit watchdog exhausted')
def stop_children():
 for child,row in active.values():
  if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
 for child,row in active.values():
  if child.poll() is None:
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
signal.signal(signal.SIGALRM,alarm);signal.alarm(a['outer_watchdog_seconds']-20);save()
try:
 with ExitStack() as stack:
  locks={repo/c['root']/'.trainer.lock' for c in a['cases'].values()}|{(repo/c['root']).parent/'.driver.lock' for c in a['cases'].values()}
  for p in sorted(locks):fcntl.flock(stack.enter_context(p.open('rb')),fcntl.LOCK_SH|fcntl.LOCK_NB)
  original=snapshot();body['original_inputs_sha256']=original;save();queue=list(a['cases'])
  while queue or active:
   while queue and len(active)<a['max_parallel_audits']:
    label=queue.pop(0);c=a['cases'][label];directory=repo/c['root'];manifest=load(directory/'run.json');assert manifest['source_sha256']==c['source_sha256'] and manifest['config']==c['config'];output=root/(label+'.json');cmd=[str(repo/'.venv/bin/python'),str(repo/c['auditor']),str(directory),'--allow-diagnostic','--output',str(output)]
    if c['kind']=='bounded':cmd+=['--time-cap-seconds',str(a['case_inner_cap_seconds'])]
    row={'case':label,'command':cmd,'output':str(output),'started_monotonic':time.monotonic()};body['commands'].append(row);env=dict(os.environ,PYTHONPATH=str(repo/c['source_package_path']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (root/(label+'.stdout')).open('wb') as out,(root/(label+'.stderr')).open('wb') as err:child=subprocess.Popen(cmd,cwd=(repo/c['source_package_path']).parent,env=env,stdout=out,stderr=err,start_new_session=True)
    row['original_pid']=child.pid;active[label]=(child,row);save()
   for label,(child,row) in list(active.items()):
    elapsed=time.monotonic()-row['started_monotonic'];code=child.poll()
    if code is None and elapsed>=a['case_outer_cap_seconds']:os.killpg(child.pid,signal.SIGKILL);child.wait();code=child.returncode;row['outer_budget_kill_requested']=True
    if code is None:continue
    row.update(returncode=code,measured_process_upper_seconds=elapsed);c=a['cases'][label];output=Path(row['output']);verified=False
    if code==0:
     try:
      r=load(output);assert r['verified'] and r['source_sha256']==c['source_sha256'] and r['auditor_sha256']==c['auditor_sha256'] and r['training_manifest_sha256']==c['manifest_sha256'];assert r.get('result_sha256',c['result_sha256'])==c['result_sha256'];assert r['records']==c['config']['records'] and r['task']==c['config']['task'] and r['learner']==c['config']['learner'] and r['seed']==0 and not r['formal_protocol'];verified=True
     except BaseException as exc:row['verification_error']=repr(exc)
    body['cases'][label]={'verified':verified,'returncode':code,'output_sha256':sha(output) if output.exists() else None,'source_sha256':c['source_sha256']};active.pop(label);save()
   if queue or active:time.sleep(.25)
  body['original_inputs_unchanged']=original==snapshot();assert body['original_inputs_unchanged'];body['status']='verified_all_eight_profiles' if all(c['verified'] for c in body['cases'].values()) and len(body['cases'])==8 else 'original_audits_include_failures'
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
 stop_children();body['measured_child_process_upper_seconds']=sum(r.get('measured_process_upper_seconds',0) for r in body['commands']);body['measured_driver_wall_seconds']=time.monotonic()-started;save();signal.alarm(0)

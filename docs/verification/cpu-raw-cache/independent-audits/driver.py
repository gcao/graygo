from pathlib import Path
from contextlib import ExitStack
import fcntl,hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/cpu-raw-cache-prefix-audit-b2a8561';allocation=frozen/'docs/experiments/cpu-raw-cache-independent-audit-allocation.json';started=time.monotonic();active={}
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
a=load(allocation);assert sha(allocation)=='dc3a7e0bfacca865619a30db46a60b1c6c78a6efbe62ee55b831e3e255b61251';assert sha(repo/a['comparison_allocation'])==a['comparison_allocation_sha256'];assert sha(repo/a['comparison_job'])==a['comparison_job_sha256'];comparison_root=(repo/a['comparison_job']).parent;assert sha(comparison_root/'comparison.json')==a['comparison_result_sha256'] and sha(repo/a['auditor'])==a['auditor_sha256']
from graygo.manifest import source_digest
assert source_digest()==a['learner_source_sha256']
body={'version':1,'status':'auditing_original_cache_prefixes','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'driver_sha256':sha(__file__),'source_sha256':a['learner_source_sha256'],'auditor_sha256':a['auditor_sha256'],'cases':{},'commands':[],'new_training_records':0,'new_scored_evaluation_games':0,'new_accelerator_jobs':0,'execution_authorized':False}
def save():
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
def snapshot():
 paths={repo/a['comparison_job'],comparison_root/'comparison.json'}
 for c in a['cases'].values():
  directory=repo/c['root'];paths.update(p for p in directory.rglob('*') if p.is_file() and not any(part.startswith('.') for part in p.relative_to(directory).parts));paths.update(directory.parent/name for name in ['plan.json','profile.json','state.json']);paths.update((directory.parent/'budgets').glob('*.json'))
 return {str(p):sha(p) for p in sorted(paths)}
def alarm(*args):raise TimeoutError('fixed cache-prefix audit watchdog exhausted')
def cleanup():
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
    label=queue.pop(0);c=a['cases'][label];directory=repo/c['root'];manifest=load(directory/'run.json');assert sha(directory/'run.json')==c['manifest_sha256'] and sha(directory/'result.json')==c['result_sha256'] and manifest['config']==c['config'] and manifest['source_sha256']==a['learner_source_sha256'];output=root/(label.replace('/','-')+'.json');cmd=[str(repo/'.venv/bin/python'),str(repo/a['auditor']),str(directory),'--allow-diagnostic','--time-cap-seconds',str(a['case_inner_cap_seconds']),'--output',str(output)];row={'case':label,'command':cmd,'output':str(output),'started_monotonic':time.monotonic()};body['commands'].append(row);env=dict(os.environ,PYTHONPATH=str(repo/a['source_package_path']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (root/(label.replace('/','-')+'.stdout')).open('wb') as out,(root/(label.replace('/','-')+'.stderr')).open('wb') as err:child=subprocess.Popen(cmd,cwd=(repo/a['source_package_path']).parent,env=env,stdout=out,stderr=err,start_new_session=True)
    row['original_pid']=child.pid;active[label]=(child,row);save()
   for label,(child,row) in list(active.items()):
    elapsed=time.monotonic()-row['started_monotonic'];code=child.poll()
    if code is None and elapsed>=a['case_outer_cap_seconds']:os.killpg(child.pid,signal.SIGKILL);child.wait();code=child.returncode;row['outer_budget_kill_requested']=True
    if code is None:continue
    row.update(returncode=code,measured_process_upper_seconds=elapsed);output=Path(row['output']);c=a['cases'][label];verified=False
    if code==0:
     try:
      report=load(output);assert report['verified'] and report['auditor_sha256']==a['auditor_sha256'] and report['source_sha256']==a['learner_source_sha256'] and report['training_manifest_sha256']==c['manifest_sha256'] and report['result_sha256']==c['result_sha256'] and report['records']==32768 and not report['formal_protocol'];verified=True
     except BaseException as exc:row['verification_error']=repr(exc)
    body['cases'][label]={'verified':verified,'returncode':code,'output_sha256':sha(output) if output.exists() else None};active.pop(label);save()
   if queue or active:time.sleep(.25)
  body['original_inputs_unchanged']=original==snapshot();assert body['original_inputs_unchanged'];body['status']='verified_all_four_actual_cache_prefixes' if len(body['cases'])==4 and all(c['verified'] for c in body['cases'].values()) else 'original_cache_audits_include_failures'
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:cleanup();body['measured_child_process_upper_seconds']=sum(r.get('measured_process_upper_seconds',0) for r in body['commands']);body['measured_driver_wall_seconds']=time.monotonic()-started;save();signal.alarm(0)

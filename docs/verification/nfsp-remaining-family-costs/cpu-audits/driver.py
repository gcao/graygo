from pathlib import Path
from contextlib import ExitStack
import fcntl,hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/nfsp-remaining-family-profile-audit-4ec9eb4';allocation=frozen/'docs/experiments/nfsp-remaining-family-profile-audit-allocation.json';started=time.monotonic();active={};last_save=0
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
a=load(allocation);assert sha(allocation)=='f49ac3d88bfa06452d6a8ea6c485925b7b3b8780c7dceb84a7c9f75483dfed38';assert sha(repo/a['profile_allocation'])==a['profile_allocation_sha256'] and sha(repo/a['profile_root']/'job.json')==a['profile_job_sha256'];assert sha(repo/a['profile_root']/'driver.py')==a['profile_driver_sha256'];assert sha(repo/a['auditor'])==a['auditor_sha256']
from graygo.manifest import source_digest
assert source_digest()==a['learner_source_sha256']
body={'version':1,'status':'auditing_fixed_original_remaining_family_profiles','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'driver_sha256':sha(__file__),'source_sha256':a['learner_source_sha256'],'auditor_sha256':a['auditor_sha256'],'shared_helper_sha256':sha((repo/a['auditor']).with_name('audit_nfsp_delayed.py')),'cases':{},'commands':[],'new_training_records':0,'new_scored_evaluation_games':0,'new_accelerator_jobs':0,'execution_authorized':False}
def save():
 global last_save
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json');last_save=time.monotonic()
def snapshot():
 paths={repo/a['profile_root']/'job.json',repo/a['profile_root']/'driver.py'}
 for case in a['cases'].values():
  directory=repo/case['root'];paths.update(p for p in directory.rglob('*') if p.is_file() and not any(part.startswith('.') for part in p.relative_to(directory).parts));paths.update(directory.parent/name for name in ['plan.json','profile.json','state.json']);paths.update((directory.parent/'budgets').glob('*.json'))
 return {str(p):sha(p) for p in sorted(paths)}
def alarm(*args):raise TimeoutError('remaining-family independent audit watchdog exhausted')
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
  for path in sorted(locks):fcntl.flock(stack.enter_context(path.open('rb')),fcntl.LOCK_SH|fcntl.LOCK_NB)
  original=snapshot();body['original_inputs_sha256']=original;save();queue=list(a['cases'])
  while queue or active:
   while queue and len(active)<a['max_parallel_audits']:
    label=queue.pop(0);case=a['cases'][label];directory=repo/case['root'];manifest=load(directory/'run.json');result=load(directory/'result.json')
    assert sha(directory/'run.json')==case['manifest_sha256'] and sha(directory/'result.json')==case['result_sha256'];assert manifest['config']==case['config'] and manifest['source_sha256']==a['learner_source_sha256'] and manifest['pid']==case['original_profile_pid'] and result['status']=='complete' and result['records']==32768
    tag=label.replace('/','-');output=root/(tag+'.json');cmd=[str(repo/'.venv/bin/python'),str(repo/a['auditor']),str(directory),'--allow-diagnostic','--time-cap-seconds',str(a['case_inner_cap_seconds']),'--output',str(output)];row={'case':label,'command':cmd,'output':str(output),'started_monotonic':time.monotonic()};body['commands'].append(row)
    env=dict(os.environ,PYTHONPATH=str(repo/a['source_package_path']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (root/(tag+'.stdout')).open('wb') as out,(root/(tag+'.stderr')).open('wb') as err:child=subprocess.Popen(cmd,cwd=(repo/a['source_package_path']).parent,env=env,stdout=out,stderr=err,start_new_session=True)
    row['original_pid']=child.pid;active[label]=(child,row);save()
   for label,(child,row) in list(active.items()):
    elapsed=time.monotonic()-row['started_monotonic'];code=child.poll()
    if code is None and elapsed>=a['case_outer_cap_seconds']:
     os.killpg(child.pid,signal.SIGKILL);child.wait();code=child.returncode;row['outer_budget_kill_requested']=True
    if code is None:continue
    row.update(returncode=code,measured_process_upper_seconds=elapsed);case=a['cases'][label];output=Path(row['output']);verified=False
    if code==0:
     try:
      result=load(output);assert result['verified'] and result['source_sha256']==a['learner_source_sha256'] and result['auditor_sha256']==a['auditor_sha256'] and result['training_manifest_sha256']==case['manifest_sha256'] and result['result_sha256']==case['result_sha256'];assert result['records']==32768 and result['task']==case['config']['task'] and result['learner']==case['config']['learner'] and result['seed']==0 and not result['formal_protocol'];verified=True
     except BaseException as exc:row['verification_error']=repr(exc)
    body['cases'][label]={'verified':verified,'output_sha256':sha(output) if output.exists() else None,'returncode':code};active.pop(label);save()
   if time.monotonic()-last_save>=10:save()
   if queue or active:time.sleep(.25)
  body['original_inputs_unchanged']=original==snapshot();assert body['original_inputs_unchanged'];body['status']='verified_all_twelve_original_profiles' if all(c['verified'] for c in body['cases'].values()) and len(body['cases'])==12 else 'original_profile_audits_include_failures'
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
 stop_children();body['measured_child_process_upper_seconds']=sum(r.get('measured_process_upper_seconds',0) for r in body['commands']);body['measured_driver_wall_seconds']=time.monotonic()-started;save();signal.alarm(0)

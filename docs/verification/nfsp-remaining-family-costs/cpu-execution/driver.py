from pathlib import Path
import fcntl,hashlib,json,os,shutil,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/nfsp-remaining-family-cpu-profiles-14c3772';allocation=frozen/'docs/experiments/nfsp-remaining-family-cpu-profile-allocation.json';started=time.monotonic();children={};last_save=0
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
a=load(allocation);assert sha(allocation)=='696165e9c5699b7af5270d310b5819c95c27295cdc65a1c0b242da62ebab14c7';assert sha(repo/a['runner'])==a['runner_sha256'] and sha(repo/a['bounded_auditor'])==a['bounded_auditor_sha256'];assert sha(repo/a['prior_usage'])==a['prior_usage_sha256'];assert shutil.disk_usage(repo).free>28*1024**3
available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024;assert available>12*1024**3
assert not subprocess.check_output(['git','-C',str((repo/a['source_package_path']).parent),'status','--porcelain'],text=True)
assert subprocess.check_output(['git','-C',str((repo/a['source_package_path']).parent),'rev-parse','HEAD'],text=True).strip()==a['learner_source_commit']
def source_sha():
 p=repo/a['source_package_path']/'graygo';h=hashlib.sha256()
 for q in sorted([*p.glob('*.py'),*(p/'native').glob('*.cpp')]):h.update(q.relative_to(p).as_posix().encode()+b'\0'+q.read_bytes()+b'\0')
 return h.hexdigest()
assert source_sha()==a['learner_source_sha256']
body={'version':1,'status':'profiling_fixed_remaining_families','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'driver_sha256':sha(__file__),'source_sha256':a['learner_source_sha256'],'groups':{},'additional_control_reservation_seconds':a['additional_control_reservation_seconds'],'new_formal_a2_records':0,'new_accelerator_jobs':0,'no_observation_timeout_restart':True,'a2_complete':False,'project_scope_selected':False}
def save():
 global last_save
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json');last_save=time.monotonic()
def stop(*args):raise TimeoutError('remaining-family profile whole-job watchdog or external stop')
def stop_children():
 for label,child in children.items():
  if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
 for label,child in children.items():
  if child.poll() is None:
   row=body['groups'][label];remaining=max(.01,min(20,row['started_monotonic']+row['whole_process_cap_seconds']-1-time.monotonic()))
   try:child.wait(timeout=remaining)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
def observe(label,now):
 row=body['groups'][label];child=children[label];directory=root/'profiles'/label;elapsed=now-row['started_monotonic'];code=child.poll()
 if code is None:
  if elapsed>=row['whole_process_cap_seconds']-30 and not row.get('budget_stop_requested'):os.killpg(child.pid,signal.SIGTERM);row['budget_stop_requested']=True
  if elapsed>=row['whole_process_cap_seconds']-1:os.killpg(child.pid,signal.SIGKILL);row['hard_budget_kill_requested']=True
 elif 'returncode' not in row:row.update(returncode=code,measured_process_upper_seconds=elapsed)
 if (directory/'plan.json').exists():
  plan=load(directory/'plan.json');assert plan['source_sha256']==a['learner_source_sha256'] and plan['driver_sha256']==a['runner_sha256'] and plan['cases']==a['groups'][label]['profile_configs'];row['emitted_plan_verified']=True
 if (directory/'state.json').exists():row['state']=load(directory/'state.json')
 if (directory/'profile.json').exists():row['completed_profiles']=[{k:r.get(k) for k in ['phase','learner','q_reuse','status','records','seconds_including_setup','counters','acting_policy_counts']} for r in load(directory/'profile.json')['results']]
 if code is not None:
  row['profile_complete']=(code==0 and row.get('state',{}).get('status')=='complete' and len(row.get('completed_profiles',[]))==2 and all(r['status']=='complete' and r['records']==32768 for r in row['completed_profiles']))

signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.alarm(a['outer_watchdog_seconds']-20)
with (root/'.driver.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save()
 try:
  env=dict(os.environ,PYTHONPATH=str(repo/a['source_package_path']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  for label,g in a['groups'].items():
   loop=g['profile_configs'][0]['config']['time_cap_seconds'];minutes=str(loop/60);assert float(minutes)*60==loop
   directory=root/'profiles'/label;assert not directory.exists();cmd=[str(repo/'.venv/bin/python'),str(repo/a['runner']),'--run-dir',str(directory),'--expected-source-sha',a['learner_source_sha256'],'--records','32768','--device','cpu','--storage','features','--case-minutes',minutes,'--task',g['task'],'--learners',g['learner'],'--reuses','4']
   log=label.replace('/','-')
   with (root/(log+'.stdout')).open('wb') as out,(root/(log+'.stderr')).open('wb') as err:
    tick=time.monotonic();child=subprocess.Popen(cmd,cwd=(repo/a['source_package_path']).parent,env=env,stdout=out,stderr=err,start_new_session=True)
   children[label]=child;body['groups'][label]={'original_pid':child.pid,'started_monotonic':tick,'whole_process_cap_seconds':g['complete_process_ceiling_seconds'],'command':cmd};save()
  while True:
   now=time.monotonic()
   for label in children:observe(label,now)
   if now-last_save>=10:save()
   if all(child.poll() is not None for child in children.values()):break
   time.sleep(.25)
  body['status']='original_profile_groups_terminal_unreviewed';body['complete_groups']=sum(r['profile_complete'] for r in body['groups'].values());body['diagnostic_records']=sum(r['records'] for g in body['groups'].values() for r in g.get('completed_profiles',[]));assert body['diagnostic_records']<=a['aggregate_diagnostic_record_cap']
 except BaseException as exc:body.update(status='failed_or_stopped',error=repr(exc))
 finally:
  stop_children();body['source_package_unchanged']=source_sha()==a['learner_source_sha256'];body['measured_driver_wall_seconds']=time.monotonic()-started
  for label,child in children.items():body['groups'][label]['final_returncode']=child.returncode
  save();signal.alarm(0)

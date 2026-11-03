from pathlib import Path
import fcntl,hashlib,json,os,shutil,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/nfsp-delayed-neural-seeds12-11a0110';allocation=frozen/'docs/experiments/nfsp-delayed-neural-seeds12-allocation.json';started=time.monotonic();children={};last_save=0
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
a=load(allocation);assert sha(allocation)=='c5346b84cf2cd892159b6405314ddb0f5eed3812a7ee290c80b7400ff029e8de';assert sha(frozen/a['runner'])==a['runner_sha256'];assert sha(repo/a['cpu_price_receipt'])==a['cpu_price_receipt_sha256'];assert sha(repo/a['source_registry'])==a['source_registry_sha256'];assert sha(repo/a['own_source_plan'])==a['own_source_plan_sha256'];assert sha(repo/a['raw_budget'])==a['raw_budget_sha256'];assert sha(repo/a['source_registration_job'])==a['source_registration_job_sha256']
assert load(repo/a['source_registration_job'])['status']=='registered_original_selected_cpu_plan_with_unchanged_prior_cases';assert shutil.disk_usage(repo).free>28*1024**3
available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024;assert available>6*1024**3
assert not subprocess.check_output(['git','-C',str(frozen),'status','--porcelain'],text=True);assert not subprocess.check_output(['git','-C',str((repo/a['source_package_path']).parent),'status','--porcelain'],text=True)
assert subprocess.check_output(['git','-C',str((repo/a['source_package_path']).parent),'rev-parse','HEAD'],text=True).strip()==a['learner_source_commit']
def source_sha():
 p=repo/a['source_package_path']/'graygo';h=hashlib.sha256()
 for q in sorted([*p.glob('*.py'),*(p/'native').glob('*.cpp')]):h.update(q.relative_to(p).as_posix().encode()+b'\0'+q.read_bytes()+b'\0')
 return h.hexdigest()
assert source_sha()==a['learner_source_sha256'];(root/'neural-budget.json').write_bytes((repo/a['raw_budget']).read_bytes());assert sha(root/'neural-budget.json')==a['raw_budget_sha256']
body={'version':1,'status':'running_original_delayed_neural_seeds','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'driver_sha256':sha(__file__),'runner_sha256':a['runner_sha256'],'source_sha256':a['learner_source_sha256'],'per_case_complete_process_ceiling_seconds':a['per_case_process_ceiling_seconds'],'graceful_budget_stop_headroom_seconds':30,'cases':{},'additional_control_reservation_seconds':'0','new_accelerator_jobs':0,'no_implicit_resume_or_observation_timeout_restart':True,'a2_complete':False,'pilot_selected':False,'project_scope_selected':False}
def save():
 global last_save
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json');last_save=time.monotonic()
def stop(*args):raise TimeoutError('original delayed neural whole-job watchdog or external stop')
def stop_children():
 for seed,child in children.items():
  if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
 for seed,child in children.items():
  if child.poll() is None:
   remaining=max(.01,min(20,body['cases'][str(seed)]['started_monotonic']+a['per_case_process_ceiling_seconds']-1-time.monotonic()))
   try:child.wait(timeout=remaining)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
def observe(seed,now):
 row=body['cases'][str(seed)];child=children[seed];directory=repo/a['case_roots'][str(seed)];elapsed=now-row['started_monotonic'];code=child.poll()
 if code is None:
  if elapsed>=a['per_case_process_ceiling_seconds']-30 and not row.get('budget_stop_requested'):
   os.killpg(child.pid,signal.SIGUSR1);row['budget_stop_requested']=True
  if elapsed>=a['per_case_process_ceiling_seconds']-1:os.killpg(child.pid,signal.SIGKILL);row['hard_budget_kill_requested']=True
 elif 'returncode' not in row:row.update(returncode=code,measured_case_process_upper_seconds=elapsed)
 if (directory/'run.json').exists():
  manifest=load(directory/'run.json');assert manifest['source_sha256']==a['learner_source_sha256'] and manifest['config']==a['case_configs'][str(seed)] and manifest['runtime']['device']=='cpu' and manifest['runtime']['torch_threads']==manifest['runtime']['workers']==1;row['emitted_manifest_verified']=True
 if (directory/'initialization.json').exists():
  initial=load(directory/'initialization.json');assert initial['step']==0 and initial['source_sha256']==a['learner_source_sha256'] and all(v==0 for v in initial['counters'].values());assert sha(directory/'initial-checkpoint/manifest.json')==initial['retained_initial_checkpoint_manifest_sha256']==initial['original_initial_checkpoint_manifest_sha256'];row['actual_zero_record_initialization_retained']=True;row['initialization_sha256']=sha(directory/'initialization.json')
 if (directory/'case-state.json').exists():row['state']=load(directory/'case-state.json')
 if (directory/'progress.json').exists():
  progress=load(directory/'progress.json');assert progress['source_sha256']==a['learner_source_sha256'] and progress['seed']==seed and progress['records']<=15000000;row['latest_progress']={k:progress.get(k) for k in ['status','records','elapsed_seconds','counters','q_credit_passed','average_readiness']}
 if code is not None and (directory/'result.json').exists():
  result=load(directory/'result.json');row['terminal_result']={k:result.get(k) for k in ['status','records','elapsed_seconds','formal_case_passed','q_credit_passed','average_readiness']};row['result_sha256']=sha(directory/'result.json')

signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.alarm(a['outer_watchdog_seconds']-20)
with (root/'.driver.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save()
 try:
  env=dict(os.environ,PYTHONPATH=str(repo/a['source_package_path'])+os.pathsep+str(frozen),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  for seed in a['seeds']:
   assert not (repo/a['case_roots'][str(seed)]).exists();cmd=[str(repo/'.venv/bin/python'),str(frozen/a['runner']),'--repo',str(repo),'--allocation',str(allocation),'--allocation-sha',sha(allocation),'--seed',str(seed)]
   with (root/f'seed{seed}.stdout').open('wb') as out,(root/f'seed{seed}.stderr').open('wb') as err:
    tick=time.monotonic();child=subprocess.Popen(cmd,cwd=frozen,env=env,stdout=out,stderr=err,start_new_session=True)
   children[seed]=child;body['cases'][str(seed)]={'original_pid':child.pid,'started_monotonic':tick,'command':cmd};save()
  while True:
   now=time.monotonic()
   for seed in children:observe(seed,now)
   if now-last_save>=10:save()
   if all(child.poll() is not None for child in children.values()):break
   time.sleep(.25)
  body['status']='original_neural_cases_terminal_unreviewed';body['completed_formal_cases']=sum(row.get('terminal_result',{}).get('status')=='complete' and row.get('terminal_result',{}).get('records')==15000000 for row in body['cases'].values())
 except BaseException as exc:body.update(status='failed_or_stopped',error=repr(exc))
 finally:
  stop_children();body['source_package_unchanged']=source_sha()==a['learner_source_sha256'];body['measured_driver_wall_seconds']=time.monotonic()-started
  for seed,child in children.items():body['cases'][str(seed)]['final_returncode']=child.returncode
  save();signal.alarm(0)

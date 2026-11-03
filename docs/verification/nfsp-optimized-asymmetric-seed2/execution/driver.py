from pathlib import Path
import hashlib,json,os,shutil,signal,subprocess,sys,time

repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/nfsp-optimized-asymmetric-seed2-2301d7e'
allocation=frozen/'docs/experiments/nfsp-optimized-asymmetric-seed2-allocation.json';a=json.loads(allocation.read_text())
started=time.monotonic();children={};stages={};closed={};ended={};last_save=0
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def source_sha():
 p=repo/a['source_package_path']/'graygo';h=hashlib.sha256()
 for f in sorted([*p.glob('*.py'),*(p/'native').glob('*.cpp')]):h.update(f.relative_to(p).as_posix().encode()+b'\0'+f.read_bytes()+b'\0')
 return h.hexdigest()
assert sha(allocation)=='480c7b9439a41cc943cc1cbcb1b98b47b5646397d8af43afa10798be537ef716'
assert sha(frozen/a['runner'])==a['runner_sha256'];assert source_sha()==a['learner_source_sha256']
assert sha(repo/a['prior_usage'])==a['prior_usage_sha256']
assert sha(repo/a['source_registry'])==a['source_registry_sha256']
assert sha(repo/a['original_source_plan'])==a['original_source_plan_sha256']
assert shutil.disk_usage(repo).free>28*1024**3
assert not subprocess.check_output(['git','-C',str(frozen),'status','--porcelain'],text=True)
assert not subprocess.check_output(['git','-C',str((repo/a['source_package_path']).parent),'status','--porcelain'],text=True)
sys.path.insert(0,str(frozen));from tools.run_nfsp_matrix_stage import review_funding
from tools.a2_source_cohort import read_registry
read_registry(repo/a['source_registry'],repo=repo,current=True)
body={'version':1,'status':'running','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'driver_sha256':sha(__file__),
 'runner_sha256':a['runner_sha256'],'runner_commit':subprocess.check_output(['git','-C',str(frozen),'rev-parse','HEAD'],text=True).strip(),
 'learner_source_sha256':a['learner_source_sha256'],'stages':stages,'conservative_control_prefix_guard_seconds':1400,
 'control_reservation_seconds':a['additional_control_reservation_seconds'],'outer_watchdog_seconds':3660,
 'no_observation_timeout_restart':True,'a2_complete':False,'pilot_selected':False,'new_accelerator_jobs':0}
def save():
 global last_save
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json');last_save=time.monotonic()
def stop(*args):raise TimeoutError('original corrected matrix outer watchdog exhausted')
def stop_children():
 for task,child in children.items():
  if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
 for task,child in children.items():
  if child.poll() is None:
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
def observe(task,now):
 row=stages[task];child=children[task];directory=root/task;code=child.poll()
 if code is not None and task not in ended:ended[task]=now;row['returncode']=code
 if (directory/'plan.json').exists():
  plan=load(directory/'plan.json');expected=a['stages'][task]
  assert plan['source_sha256']==a['learner_source_sha256'] and plan['driver_sha256']==a['runner_sha256']
  assert plan['profiles']==expected['profiles'] and plan['conditional_trials']==expected['conditional_trials']
  assert plan['source_commit']==a['learner_source_commit'];row['emitted_plan_verified']=True
 if (directory/'state.json').exists():
  state=load(directory/'state.json');row['state']=state
  if state.get('phase')=='formal' and state.get('current_case')=='neural' and task not in closed:
   result=load(directory/'trials/tabular/result.json')
   assert result['status']=='complete' and result['records']==15000000 and result['source_sha256']==a['learner_source_sha256'] and result['formal_protocol']
   closed[task]=now;row['control_prefix_closed_at_seconds']=now-row['started_monotonic']
  phase=state.get('phase');case=state.get('current_case')
  current_label=str(phase)+'/'+str(case)
  if row.get('progress_case')!=current_label:row.pop('latest_case_progress',None);row['progress_case']=current_label
  if phase in ('profile','formal') and case:
   prefix='profiles' if phase=='profile' else 'trials';progress=directory/prefix/case/'progress.json'
   if progress.exists():
    r=load(progress);row['latest_case_progress']={key:r.get(key) for key in ('records','status','elapsed_seconds','learner','task','counters')}
 if (directory/'budget.json').exists():
  profile=load(directory/'profile.json');budget=load(directory/'budget.json')
  assert profile['plan']['source_sha256']==a['learner_source_sha256']
  expected=review_funding(profile['results'],a['stages'][task]['conditional_trials'])
  assert budget['funding']==expected and budget['source_sha256']==a['learner_source_sha256']
  row['funding']=expected;row['funding_recomputed_same_formula_in_parent']=True
 endpoint=closed.get(task,ended.get(task,now));row['conservative_control_prefix_seconds']=endpoint-row['started_monotonic']
 if code is None and now-row['started_monotonic']>3600:raise TimeoutError(task+' original stage watchdog exhausted')

signal.signal(signal.SIGALRM,stop);signal.alarm(3630);save()
try:
 env=dict(os.environ,PYTHONPATH=str(repo/a['source_package_path'])+os.pathsep+str(frozen))
 for task in ('asymmetric',):
  assert not (root/task).exists()
  cmd=[str(repo/'.venv/bin/python'),str(frozen/a['runner']),'--run-dir',str(root/task),'--task',task,
       '--expected-source-sha',a['learner_source_sha256'],'--seed','2','--profile-seconds','65','--tabular-trial-hours',str(1/3),'--neural-trial-hours','.5']
  out=(root/(task+'.stdout')).open('wb');err=(root/(task+'.stderr')).open('wb')
  tick=time.monotonic();child=subprocess.Popen(cmd,cwd=frozen,env=env,stdout=out,stderr=err,start_new_session=True);out.close();err.close()
  children[task]=child;stages[task]={'command':cmd,'original_pid':child.pid,'started_monotonic':tick};save()
 while True:
  now=time.monotonic()
  for task in children:observe(task,now)
  upper=sum(row['conservative_control_prefix_seconds'] for row in stages.values());body['conservative_control_prefix_sum_seconds']=upper
  if upper>=1400:raise TimeoutError('combined conservative control-prefix guard exhausted; no extra control budget is selected')
  if now-last_save>=10:save()
  if all(child.poll() is not None for child in children.values()):break
  time.sleep(1)
 assert all(child.returncode==0 for child in children.values())
 body['status']='original_stages_terminal_unreviewed'
 body['formal_cases_complete']=0;body['diagnostic_profile_records']=0;body['formal_records']=0
 for task in children:
  directory=root/task;state=load(directory/'state.json');row=stages[task];row['final_state']=state
  if (directory/'profile.json').exists():body['diagnostic_profile_records']+=sum(v['records'] for v in load(directory/'profile.json')['results'])
  if (directory/'summary.json').exists():
   results=load(directory/'summary.json')['results'];body['formal_records']+=sum(r['records'] for r in results)
   body['formal_cases_complete']+=sum(r['status']=='complete' and r['records']==15000000 for r in results)
  row['final_state_sha256']=sha(directory/'state.json')
 assert body['diagnostic_profile_records']<=400000 and body['formal_records']<=30000000
except BaseException as exc:body.update(status='failed_or_stopped',error=repr(exc))
finally:
 stop_children();body['source_package_unchanged']=source_sha()==a['learner_source_sha256'];body['measured_driver_seconds']=time.monotonic()-started
 for task,child in children.items():stages[task]['final_returncode']=child.returncode
 save();signal.alarm(0)

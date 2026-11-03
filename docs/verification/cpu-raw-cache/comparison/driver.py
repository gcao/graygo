from pathlib import Path
import hashlib,json,os,shutil,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/cpu-raw-cache-comparison-257276e';allocation=frozen/'docs/experiments/cpu-raw-cache-pipeline-allocation.json';started=time.monotonic();children={};comparison=None;last_save=0
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
a=load(allocation);assert sha(allocation)=='d593971e0bb310842a498a78a8b7801d3a936ef3cc1bb013bd6a38d6d481ab0e';assert sha(frozen/a['profile_runner'])==a['profile_runner_sha256'] and sha(frozen/a['comparison_tool'])==a['comparison_tool_sha256'] and sha(frozen/a['comparison_core'])==a['comparison_core_sha256'];assert sha(repo/'runs/nfsp-remaining-family-cpu-profiles-v1/job.json')==a['reference_profile_job_sha256'] and sha(repo/'runs/nfsp-remaining-family-profile-audit-4ec9eb4/job.json')==a['reference_audit_job_sha256'];assert shutil.disk_usage(repo).free>28*1024**3
assert not subprocess.check_output(['git','-C',str(frozen),'status','--porcelain'],text=True)
for c in a['cases']:assert sha(repo/c['reference_root']/'run.json')==c['reference_manifest_sha256'] and sha(repo/c['reference_root']/'result.json')==c['reference_result_sha256']
body={'version':1,'status':'running_fixed_candidate_raw_cache_prefixes','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'driver_sha256':sha(__file__),'reference_source_sha256':a['reference_source_sha256'],'candidate_source_sha256':a['candidate_source_sha256'],'groups':{},'new_formal_A2_records':0,'additional_control_reservation_seconds':'0','new_accelerator_jobs':0,'source_compatibility_qualified':False,'execution_authorized':False,'no_observation_timeout_restart':True}
def save():
 global last_save
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json');last_save=time.monotonic()
def stop(*args):raise TimeoutError('raw-cache prefix watchdog exhausted')
def cleanup():
 for task,child in children.items():
  if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
 for task,child in children.items():
  if child.poll() is None:
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 if comparison is not None and comparison.poll() is None:os.killpg(comparison.pid,signal.SIGKILL);comparison.wait()
def observe(task,now):
 row=body['groups'][task];child=children[task];directory=root/'profiles'/task;elapsed=now-row['started_monotonic'];code=child.poll()
 if code is None:
  if elapsed>=a['per_complete_candidate_process_ceiling_seconds']-30 and not row.get('budget_stop_requested'):os.killpg(child.pid,signal.SIGTERM);row['budget_stop_requested']=True
  if elapsed>=a['per_complete_candidate_process_ceiling_seconds']-1:os.killpg(child.pid,signal.SIGKILL);row['hard_budget_kill_requested']=True
 elif 'returncode' not in row:row.update(returncode=code,measured_process_upper_seconds=elapsed)
 if (directory/'plan.json').exists():
  plan=load(directory/'plan.json');expected=[{'phase':c['phase'],'config':c['config']} for c in a['cases'] if c['config']['task']==task];assert plan['source_sha256']==a['candidate_source_sha256'] and plan['driver_sha256']==a['profile_runner_sha256'] and plan['cases']==expected;row['emitted_plan_verified']=True
 if (directory/'state.json').exists():row['state']=load(directory/'state.json')
 if (directory/'profile.json').exists():row['completed_profiles']=[{k:r.get(k) for k in ['phase','status','records','seconds_including_setup','counters','acting_policy_counts']} for r in load(directory/'profile.json')['results']]
signal.signal(signal.SIGALRM,stop);signal.alarm(960);save()
try:
 env=dict(os.environ,PYTHONPATH=str(frozen/'src')+os.pathsep+str(frozen),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 for task in a['tasks']:
  cmd=[str(repo/'.venv/bin/python'),str(frozen/a['profile_runner']),'--run-dir',str(root/'profiles'/task),'--expected-source-sha',a['candidate_source_sha256'],'--records','32768','--device','cpu','--case-minutes','5','--task',task,'--learners','neural','--reuses','4']
  with (root/(task+'.stdout')).open('wb') as out,(root/(task+'.stderr')).open('wb') as err:
   tick=time.monotonic();child=subprocess.Popen(cmd,cwd=frozen,env=env,stdout=out,stderr=err,start_new_session=True)
  children[task]=child;body['groups'][task]={'command':cmd,'original_pid':child.pid,'started_monotonic':tick};save()
 while True:
  now=time.monotonic()
  for task in children:observe(task,now)
  if now-last_save>=10:save()
  if all(child.poll() is not None for child in children.values()):break
  time.sleep(.25)
 assert all(c.returncode==0 for c in children.values()) and all(r['state']['status']=='complete' and len(r['completed_profiles'])==2 and all(p['status']=='complete' and p['records']==32768 for p in r['completed_profiles']) for r in body['groups'].values())
 body['status']='comparing_actual_original_complete_pipelines';save();cmd=[str(repo/'.venv/bin/python'),'-m','tools.cpu_cache_pipeline','--reference',str(repo/a['reference_root']),'--candidate',str(root/'profiles'),'--allocation',str(allocation),'--output',str(root/'comparison.json')];body['comparison_command']=cmd
 with (root/'comparison.stdout').open('wb') as out,(root/'comparison.stderr').open('wb') as err:
  tick=time.monotonic();comparison=subprocess.Popen(cmd,cwd=frozen,env=env,stdout=out,stderr=err,start_new_session=True);body['original_comparison_pid']=comparison.pid;save();code=comparison.wait(timeout=a['comparison_outer_seconds']);body.update(comparison_returncode=code,measured_comparison_process_seconds=time.monotonic()-tick)
 assert code==0,'exact pipeline comparison failed; preserve both original sources/roots';result=load(root/'comparison.json');assert result['status']=='four_actual_raw_cache_pipelines_verified' and len(result['comparisons'])==4;body.update(status='four_actual_cache_pipelines_exactly_equal',comparison_sha256=sha(root/'comparison.json'),new_candidate_diagnostic_records=131072)
except BaseException as exc:body.update(status='failed_or_stopped',error=repr(exc))
finally:cleanup();body['measured_driver_wall_seconds']=time.monotonic()-started;save();signal.alarm(0)

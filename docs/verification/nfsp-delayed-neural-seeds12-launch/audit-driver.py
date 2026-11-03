from pathlib import Path
import fcntl,hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/nfsp-delayed-neural-seeds12-audit-a6a52bd';allocation=frozen/'docs/experiments/nfsp-delayed-neural-seeds12-audit-allocation.json';started=time.monotonic();child=None;wait_seconds=0.
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
a=load(allocation);assert sha(allocation)=='08d60b99dbeacf94c1fd05b36971bbe4e9628c98bf423cde3a5de01e1c2f6ab5';collector=repo/a['collector_path'];plan=repo/a['cohort_plan']
assert sha(collector)==a['collector_sha256'] and sha(collector.with_name('audit_nfsp_delayed.py'))==a['delayed_auditor_sha256'];assert sha(plan)==a['cohort_plan_sha256'];assert sha(repo/a['source_registry'])==a['source_registry_sha256'] and sha(frozen/a['reader'])==a['reader_sha256'];assert sha(repo/a['budget_profile'])==a['budget_profile_sha256']
assert all(sha(repo/p)==h for p,h in a['original_case_files_sha256'].items());assert all(sha(repo/p)==h for p,h in a['initialization_files_sha256'].items())
from graygo.manifest import source_digest
assert source_digest()==a['learner_source_sha256']
body={'version':1,'status':'waiting_for_fixed_original_delayed_cases','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'driver_sha256':sha(__file__),'collector_sha256':a['collector_sha256'],'delayed_auditor_sha256':a['delayed_auditor_sha256'],'reader_sha256':a['reader_sha256'],'commands':[],'cases':{},'terminal_missing_cases':[],'new_training_records':0,'new_scored_evaluation_games':0,'new_accelerator_jobs':0,'no_training_launch_or_resume':True,'a2_complete':False,'execution_authorized':False,'original_initialization_seals_verified':True}
old_cases=[repo/p for p in a['original_case_files']];new_cases=[]
def save():
 body['wait_seconds']=wait_seconds;body['driver_wall_seconds']=time.monotonic()-started;p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
def original_alive(seed):
 expected=a['original_case_identities'][str(seed)];pid=expected['pid']
 try:stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
 except FileNotFoundError:return False
 if stat[0] in ('Z','X'):return False
 assert stat[19]==expected['start_ticks'] and hashlib.sha256(cmd).hexdigest()==expected['cmdline_sha256'],'original process identity changed';return True
def unlocked(directory):
 with (directory/'.trainer.lock').open('rb') as file:
  try:fcntl.flock(file,fcntl.LOCK_SH|fcntl.LOCK_NB)
  except BlockingIOError:return False
  fcntl.flock(file,fcntl.LOCK_UN);return True
def run(label,cmd,timeout):
 global child
 row={'label':label,'command':cmd};body['commands'].append(row);body['status']='reading_fixed_original_evidence';save();env=dict(os.environ,PYTHONPATH=str(repo/a['source_package_path'])+os.pathsep+str(frozen),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with (root/(label+'.stdout')).open('wb') as out,(root/(label+'.stderr')).open('wb') as err:
  tick=time.monotonic();child=subprocess.Popen(cmd,cwd=collector.parent.parent,env=env,stdout=out,stderr=err,start_new_session=True);row['pid']=child.pid;save();code=child.wait(timeout=timeout);row.update(returncode=code,measured_process_seconds=time.monotonic()-tick)
 save();assert code==0,label

def report():
 index=len(new_cases);common=[str(repo/'.venv/bin/python'),'-m','tools.a2_source_cohort','assess','--registry',str(repo/a['source_registry']),'--repo',str(repo)]
 for p in old_cases+new_cases:common+=['--case',str(p)]
 current=root/f'assessment-{index:02d}-current.json';archived=root/f'assessment-{index:02d}-archived.json'
 run(f'assess-{index:02d}-current',common+['--output',str(current)],a['assessment_outer_seconds']);run(f'assess-{index:02d}-archived',common+['--archived','--output',str(archived)],a['assessment_outer_seconds'])
 left,right=load(current),load(archived)
 for key in ('analysis_mode','current_input_snapshots_verified','current_source_admission_inputs_verified'):left.pop(key);right.pop(key)
 assert left==right and not left['a2_complete'];body['latest_assessment']={'current':str(current),'current_sha256':sha(current),'archived':str(archived),'archived_sha256':sha(archived),'present_cases':left['present_cases'],'missing_cases':len(left['missing_cases']),'current_archived_numerics_match':True};save()
def alarm(*args):raise TimeoutError('original delayed read-only observer watchdog exhausted')
signal.signal(signal.SIGALRM,alarm);signal.alarm(a['outer_watchdog_seconds']-30);save()
try:
 while len(body['cases'])+len(body['terminal_missing_cases'])<2:
  states={seed:original_alive(seed) for seed in a['original_case_identities']};body['original_worker_handles_live']=states
  for label,relative in a['new_case_roots'].items():
   if label in body['cases'] or label in body['terminal_missing_cases']:continue
   seed=label.rsplit('/',1)[1];directory=repo/relative
   if states[seed]:continue
   if not (directory/'result.json').exists():body['terminal_missing_cases'].append(label);continue
   manifest=load(directory/'run.json');result=load(directory/'result.json');assert manifest['source_sha256']==a['learner_source_sha256'] and manifest['config']==a['new_case_configs'][label]
   assert unlocked(directory),'writer remains after original worker ended'
   assert all(sha(repo/p)==h for p,h in a['initialization_files_sha256'].items());assert sha(repo/a['budget_profile'])==a['budget_profile_sha256']
   if result['status']=='complete':assert result['records']==15000000
   output=root/('case-'+label.replace('/','-')+'.json');cmd=[str(repo/'.venv/bin/python'),str(collector),'audit-case','--plan',str(plan),'--run-dir',str(directory),'--budget-profile',str(repo/a['budget_profile']),'--time-cap-seconds',str(a['case_time_ceiling_seconds']),'--output',str(output)]
   run('audit-'+label.replace('/','-'),cmd,a['case_outer_seconds']);case=load(output)['body'];assert case['case']==['delayed','neural',int(seed)] and case['root']==str(directory.resolve());new_cases.append(output);body['cases'][label]={'case_sha256':sha(output),'status':case['status'],'records':case['records'],'fresh_audit_verified':case['fresh_audit_verified']};report()
  if len(body['cases'])+len(body['terminal_missing_cases'])==2:break
  assert wait_seconds<a['maximum_wait_seconds'],'original observation cap exhausted; no learner restart selected'
  body['status']='waiting_for_fixed_original_delayed_cases';save();tick=time.monotonic();time.sleep(5);wait_seconds+=time.monotonic()-tick
 assert all(sha(repo/p)==h for p,h in a['original_case_files_sha256'].items()) and all(sha(repo/p)==h for p,h in a['initialization_files_sha256'].items())
 body['status']='fixed_original_delayed_case_evidence_collected';body['original_case_files_unchanged']=True;body['original_initialization_files_unchanged']=True
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 body['measured_child_process_seconds']=sum(row.get('measured_process_seconds',0) for row in body['commands']);save();signal.alarm(0)

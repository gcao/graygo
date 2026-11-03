from pathlib import Path
import fcntl,hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/nfsp-optimized-matrix-audit-e818fab'
allocation=frozen/'docs/experiments/nfsp-optimized-matrix-seed1-audit-allocation.json';a=json.loads(allocation.read_text());started=time.monotonic();child=None;wait_seconds=0.
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
collector=repo/a['collector_path'];plan=repo/a['cohort_plan'];assert sha(allocation)=='54c6ee73a566a2267acc324b419835ae48b1beef91cf02d2c4d2d3f6d556aaa5'
assert sha(collector)==a['collector_sha256'] and sha(collector.with_name('audit_nfsp_matrix.py'))==a['matrix_auditor_sha256'] and sha(plan)==a['cohort_plan_sha256']
assert sha(repo/a['source_registry'])==a['source_registry_sha256'] and sha(frozen/a['reader'])==a['reader_sha256']
assert not subprocess.check_output(['git','-C',str(frozen),'status','--porcelain'],text=True)
assert not subprocess.check_output(['git','-C',str(collector.parent.parent),'status','--porcelain'],text=True)
old_case_files=[repo/p for p in a['original_case_files']];old_hashes={str(p):sha(p) for p in old_case_files}
body={'version':1,'status':'waiting_for_fixed_original_cases','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'driver_sha256':sha(__file__),'collector_sha256':a['collector_sha256'],'matrix_auditor_sha256':a['matrix_auditor_sha256'],'commands':[],'cases':{},'new_training_records':0,'new_scored_evaluation_games':0,'no_training_launch_or_resume':True,'a2_complete':False,'execution_authorized':False,'outer_watchdog_seconds':8000};case_files=[];terminal_missing=[]
def save():
 body['wait_seconds']=wait_seconds;body['driver_wall_seconds']=time.monotonic()-started
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
def original_alive(task):
 expected=a['original_stage_identities'][task];pid=expected['pid']
 try:
  stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
 except FileNotFoundError:return False
 if stat[0] in ('Z','X'):return False
 assert stat[19]==expected['start_ticks'] and hashlib.sha256(cmd).hexdigest()==expected['cmdline_sha256'],'original stage process identity changed'
 return True
def unlocked(directory):
 try:
  with (directory/'.trainer.lock').open('rb') as lock:
   fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
   fcntl.flock(lock,fcntl.LOCK_UN);return True
 except BlockingIOError:return False
def run(label,cmd,timeout):
 global child
 row={'label':label,'command':cmd};body['commands'].append(row);body['status']='reading_fixed_original_evidence';save()
 env=dict(os.environ,PYTHONPATH=str(repo/a['source_package_path'])+os.pathsep+str(frozen))
 with (root/(label+'.stdout')).open('w') as out,(root/(label+'.stderr')).open('w') as err:
  tick=time.monotonic();child=subprocess.Popen(cmd,cwd=collector.parent.parent,env=env,stdout=out,stderr=err,start_new_session=True);row['pid']=child.pid;save();code=child.wait(timeout=timeout)
 row.update(returncode=code,measured_process_seconds=time.monotonic()-tick);save()
 if code!=0:raise RuntimeError(label+' read-only audit/report failed')
def report():
 index=len(case_files);cases=old_case_files+case_files;common=[str(repo/'.venv/bin/python'),'-m','tools.a2_source_cohort','assess','--registry',str(repo/a['source_registry']),'--repo',str(repo)]
 for path in cases:common+=['--case',str(path)]
 current=root/f'assessment-{index:02d}-current.json';archived=root/f'assessment-{index:02d}-archived.json'
 run(f'assess-{index:02d}-current',common+['--output',str(current)],120)
 run(f'assess-{index:02d}-archived',common+['--archived','--output',str(archived)],120)
 left,right=load(current),load(archived)
 for key in ('analysis_mode','current_input_snapshots_verified','current_source_admission_inputs_verified'):left.pop(key);right.pop(key)
 assert left==right and left['a2_complete'] is False
 body['latest_assessment']={'current':str(current),'current_sha256':sha(current),'archived':str(archived),'archived_sha256':sha(archived),'present_cases':left['present_cases'],'missing_cases':len(left['missing_cases']),'current_archived_numerics_match':True};save()
def alarm(*args):raise TimeoutError('fixed original-case audit observer watchdog exhausted')
signal.signal(signal.SIGALRM,alarm);signal.alarm(7960);save()
try:
 while len(body['cases'])+len(terminal_missing)<4:
  states={task:original_alive(task) for task in a['original_stage_identities']};body['original_stage_handles_live']=states;changed=False
  for label,relative in a['new_case_roots'].items():
   if label in body['cases'] or label in terminal_missing:continue
   task=label.split('/')[0];directory=repo/relative;result_path=directory/'result.json'
   if not result_path.exists():
    if not states[task]:terminal_missing.append(label);changed=True
    continue
   result=load(result_path);manifest=load(directory/'run.json')
   assert manifest['source_sha256']==a['learner_source_sha256'] and manifest['config']==a['new_case_configs'][label]
   if result['status']!='complete' and states[task]:continue
   if not unlocked(directory):
    if not states[task]:raise RuntimeError('case still has a writer after original bound process ended')
    continue
   if result['status']=='complete':assert result['records']==15000000
   output=root/('case-'+label.replace('/','-')+'.json');cmd=[str(repo/'.venv/bin/python'),str(collector),'audit-case','--plan',str(plan),'--run-dir',str(directory),'--time-cap-seconds','600','--output',str(output)]
   run('audit-'+label.replace('/','-'),cmd,660);wrapper=load(output);assert wrapper['body']['root']==str(directory.resolve()) and wrapper['body']['case']==label.split('/')[:2]+[1]
   case_files.append(output);body['cases'][label]={'case_sha256':sha(output),'records':wrapper['body']['records'],'status':wrapper['body']['status'],'fresh_audit_verified':wrapper['body']['fresh_audit_verified']};changed=True;report()
  body['terminal_missing_cases']=terminal_missing
  if len(body['cases'])+len(terminal_missing)==4:break
  assert wait_seconds<a['maximum_wait_seconds'],'original-case observation time exhausted; no training restart selected'
  body['status']='waiting_for_fixed_original_cases';save();tick=time.monotonic();time.sleep(5);wait_seconds+=time.monotonic()-tick
 assert old_hashes=={str(p):sha(p) for p in old_case_files}
 body['status']='fixed_original_case_evidence_collected';body['original_case_files_unchanged']=True
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 body['measured_child_process_seconds']=sum(row.get('measured_process_seconds',0.) for row in body['commands']);save();signal.alarm(0)

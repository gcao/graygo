from pathlib import Path
from contextlib import ExitStack
import fcntl,hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/nfsp-delayed-selected-cpu-prices-1583ea7';allocation=frozen/'docs/experiments/nfsp-delayed-selected-cpu-price-allocation.json';started=time.monotonic();child=None
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
a=load(allocation);assert sha(allocation)=='55af40cc9d562f1c86e186afe7c66d38782a08da18113149e9cb8ee506bfa45e';assert sha(repo/a['runner'])==a['runner_sha256'] and sha(repo/a['auditor'])==a['auditor_sha256'];assert sha(repo/a['reference_proposal'])==a['reference_proposal_sha256']
def source_sha():
 p=repo/a['source_package_path']/'graygo';h=hashlib.sha256()
 for q in sorted([*p.glob('*.py'),*(p/'native').glob('*.cpp')]):h.update(q.relative_to(p).as_posix().encode()+b'\0'+q.read_bytes()+b'\0')
 return h.hexdigest()
assert source_sha()==a['learner_source_sha256']
body={'version':1,'status':'profiling_selected_cpu_package','original_driver_pid':os.getpid(),'allocation':str(allocation),'allocation_sha256':sha(allocation),'driver_sha256':sha(__file__),'source_sha256':a['learner_source_sha256'],'runner_sha256':a['runner_sha256'],'auditor_sha256':a['auditor_sha256'],'commands':[],'cases':{},'new_formal_a2_records':0,'new_tabular_control_records':0,'new_accelerator_jobs':0,'execution_authorized':False}
def save():
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
def run(label,cmd,timeout):
 global child
 row={'label':label,'command':cmd};body['commands'].append(row);save();env=dict(os.environ,PYTHONPATH=str(repo/a['source_package_path']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with (root/(label+'.stdout')).open('wb') as out,(root/(label+'.stderr')).open('wb') as err:
  tick=time.monotonic();child=subprocess.Popen(cmd,cwd=(repo/a['source_package_path']).parent,env=env,stdout=out,stderr=err,start_new_session=True);row['pid']=child.pid;save();code=child.wait(timeout=timeout);row.update(returncode=code,seconds=time.monotonic()-tick)
 save();assert code==0,(label,code)
def snapshot():
 paths=set()
 for directory in sorted((root/'profiles').glob('neural-R*')):paths.update(p for p in directory.rglob('*') if p.is_file() and not any(part.startswith('.') for part in p.relative_to(directory).parts))
 paths.update(p for p in (root/'profiles').rglob('*.json') if not any(x in p.parts for x in ('resume','staging')))
 return {str(p):sha(p) for p in sorted(paths)}
def alarm(*args):raise TimeoutError('selected CPU delayed profile/audit watchdog exhausted')
signal.signal(signal.SIGALRM,alarm);signal.alarm(2080);save()
try:
 cmd=[str(repo/'.venv/bin/python'),str(repo/a['runner']),'--run-dir',str(root/'profiles'),'--expected-source-sha',a['learner_source_sha256'],'--records','65536','--device','cpu','--case-minutes','5','--task','delayed','--learners','neural','--reuses','1','4']
 run('profiles',cmd,a['profile_driver_outer_seconds']);assert load(root/'profiles/state.json')['status']=='complete';profile=load(root/'profiles/profile.json');assert len(profile['results'])==4 and sum(r['records'] for r in profile['results'])==262144
 body['measured_profile_process_seconds']=body['commands'][0]['seconds'];body['diagnostic_records']=262144;body['status']='auditing_selected_cpu_profiles';save()
 with ExitStack() as stack:
  locks=[root/'profiles/.driver.lock',*sorted((root/'profiles').glob('neural-R*/.trainer.lock'))]
  assert len(locks)==5
  for p in locks:fcntl.flock(stack.enter_context(p.open('rb')),fcntl.LOCK_SH|fcntl.LOCK_NB)
  original=snapshot();body['original_inputs_sha256']=original;save()
  for row in profile['results']:
   label='C'+str(row['q_reuse'])+'/'+row['phase'];directory=root/'profiles'/row['case'];output=root/(label.replace('/','-')+'.json')
   assert row['status']=='complete' and row['records']==65536
   cmd=[str(repo/'.venv/bin/python'),str(repo/a['auditor']),str(directory),'--allow-diagnostic','--output',str(output)];run('audit-'+label.replace('/','-'),cmd,a['audit_case_outer_seconds']);audit=load(output)
   assert audit['verified'] and audit['source_sha256']==a['learner_source_sha256'] and audit['training_manifest_sha256']==sha(directory/'run.json') and audit['records']==65536 and not audit['formal_protocol']
   body['cases'][label]={'root':str(directory),'output':str(output),'output_sha256':sha(output),'verified':True};save()
  body['original_inputs_unchanged']=original==snapshot();assert body['original_inputs_unchanged'];assert source_sha()==a['learner_source_sha256'];body['status']='verified_selected_package_profiles_and_audits'
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 body['measured_driver_seconds']=time.monotonic()-started;body['measured_audit_process_seconds']=sum(r.get('seconds',0) for r in body['commands'] if r['label'].startswith('audit-'));save();signal.alarm(0)

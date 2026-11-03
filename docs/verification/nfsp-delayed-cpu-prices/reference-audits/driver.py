from pathlib import Path
from contextlib import ExitStack
import fcntl,hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/nfsp-delayed-cpu-price-audit-8650f18';allocation=frozen/'docs/experiments/nfsp-delayed-cpu-price-audit-allocation.json';started=time.monotonic();child=None
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
a=load(allocation);assert sha(allocation)=='b745f2831b125dbbcd8c7e2b80f48c4118671738d2ab7052c77a2afaca34bfd6';assert sha(repo/a['proposal'])==a['proposal_sha256'];assert sha(repo/a['auditor_path'])==a['auditor_sha256']
def source_sha():
 p=repo/a['source_package_path']/'graygo';h=hashlib.sha256()
 for q in sorted([*p.glob('*.py'),*(p/'native').glob('*.cpp')]):h.update(q.relative_to(p).as_posix().encode()+b'\0'+q.read_bytes()+b'\0')
 return h.hexdigest()
assert source_sha()==a['learner_source_sha256']
body={'version':1,'status':'reading_fixed_original_profiles','original_driver_pid':os.getpid(),'allocation':str(allocation),'allocation_sha256':sha(allocation),'driver_sha256':sha(__file__),'auditor_sha256':a['auditor_sha256'],'source_sha256':a['learner_source_sha256'],'cases':{},'commands':[],'new_training_records':0,'new_scored_evaluation_games':0,'new_accelerator_jobs':0,'execution_authorized':False}
def save():
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
def snapshot():
 paths=set()
 for c in a['cases'].values():
  directory=repo/c['root'];paths.update(p for p in directory.rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(directory).parts));paths.update(directory.parent/name for name in ['plan.json','profile.json','state.json']);paths.add(directory.parent/'budgets'/('neural-R'+str(c['config']['q_reuse'])+'.json'))
 return {str(p):sha(p) for p in sorted(paths)}
def alarm(*args):raise TimeoutError('original CPU profile read-only audit watchdog exhausted')
signal.signal(signal.SIGALRM,alarm);signal.alarm(780);save()
try:
 with ExitStack() as stack:
  locks={repo/c['root']/'.trainer.lock' for c in a['cases'].values()}|{(repo/c['root']).parent/'.driver.lock' for c in a['cases'].values()}
  for p in sorted(locks):
   lock=stack.enter_context(p.open('rb'));fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
  original=snapshot();body['original_inputs_sha256']=original;save()
  for label,c in a['cases'].items():
   directory=repo/c['root'];assert sha(directory/'run.json')==c['manifest_sha256'] and sha(directory/'result.json')==c['result_sha256'];manifest=load(directory/'run.json');assert manifest['source_sha256']==a['learner_source_sha256'] and manifest['config']==c['config']
   output=root/(label.replace('/','-')+'.json');cmd=[str(repo/'.venv/bin/python'),str(repo/a['auditor_path']),str(directory),'--allow-diagnostic','--output',str(output)];row={'case':label,'command':cmd};body['commands'].append(row);save()
   env=dict(os.environ,PYTHONPATH=str(repo/a['source_package_path']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
   with (root/(label.replace('/','-')+'.stdout')).open('wb') as out,(root/(label.replace('/','-')+'.stderr')).open('wb') as err:
    tick=time.monotonic();child=subprocess.Popen(cmd,cwd=repo/a['source_package_path'],env=env,stdout=out,stderr=err,start_new_session=True);row['pid']=child.pid;save();code=child.wait(timeout=a['case_outer_seconds']);row.update(returncode=code,seconds=time.monotonic()-tick)
   assert code==0,(label,code)
   audit=load(output);assert audit['verified'] and audit['source_sha256']==a['learner_source_sha256'] and audit['training_manifest_sha256']==c['manifest_sha256'] and audit['learner']=='neural' and audit['records']==c['config']['records'] and not audit['formal_protocol']
   body['cases'][label]={'root':str(directory),'output':str(output),'output_sha256':sha(output),'verified':True};save()
  body['original_inputs_unchanged']=original==snapshot();assert body['original_inputs_unchanged'];assert source_sha()==a['learner_source_sha256'];body['status']='verified_all_four_original_profiles'
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 body['measured_driver_seconds']=time.monotonic()-started;body['measured_child_process_seconds']=sum(c.get('seconds',0) for c in body['commands']);save();signal.alarm(0)

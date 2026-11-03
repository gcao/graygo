from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/task-source-bridge-446a058'
allocation=frozen/'docs/experiments/task-source-bridge-domain-allocation.json';a=json.loads(allocation.read_text());started=time.monotonic();child=None
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def package_sha(path):
 h=hashlib.sha256();package=path/'graygo'
 for p in sorted([*package.glob('*.py'),*(package/'native').glob('*.cpp')]):h.update(p.relative_to(package).as_posix().encode()+b'\0'+p.read_bytes()+b'\0')
 return h.hexdigest()
assert sha(allocation)=='11b97d2adeb8e68ce78142c210a83f2f8f85c50975665b7126dc05be48082b32'
assert sha(frozen/a['domain_producer'])==a['domain_producer_sha256'] and sha(frozen/a['specialization_producer'])==a['specialization_producer_sha256']
assert sha(frozen/'tools/task_source_diff.py')==a['source_inventory_helper_sha256'] and sha(frozen/'tools/task_actor_boundary.py')==a['transcript_helper_sha256']
assert not subprocess.check_output(['git','-C',str(frozen),'status','--porcelain'],text=True)
for role in ('reference','candidate'):
 p=repo/a[role+'_package_path'];assert package_sha(p)==a[role+'_source_sha256'];assert not subprocess.check_output(['git','-C',str(p.parent),'status','--porcelain'],text=True)
body={'version':1,'status':'running','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'tool_commit':subprocess.check_output(['git','-C',str(frozen),'rev-parse','HEAD'],text=True).strip(),'driver_sha256':sha(__file__),'domain_producer_sha256':a['domain_producer_sha256'],'specialization_producer_sha256':a['specialization_producer_sha256'],'commands':[],'new_training_records':0,'new_scored_evaluation_games':0,'new_accelerator_job':False,'a2_evidence_reusable':False,'execution_authorized':False,'outer_watchdog_seconds':1320}
def save():
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
def run(label,cmd,role,ceiling):
 global child
 env=dict(os.environ,PYTHONPATH=str(repo/a[role+'_package_path'])+os.pathsep+str(frozen));row={'label':label,'command':cmd};body['commands'].append(row)
 remaining=a['process_time_ceiling_seconds']-(time.monotonic()-started)-20;assert remaining>0
 with (root/(label+'.stdout')).open('w') as out,(root/(label+'.stderr')).open('w') as err:
  tick=time.monotonic();child=subprocess.Popen(cmd,cwd=frozen,env=env,stdout=out,stderr=err,start_new_session=True);row['pid']=child.pid;body['current_phase']=label;save();code=child.wait(timeout=min(ceiling,remaining))
 row.update(returncode=code,seconds=time.monotonic()-tick);save()
 if code!=0:raise RuntimeError(label+' failed: '+str(code))
def alarm(*args):raise TimeoutError('original domain/specialization watchdog exhausted')
signal.signal(signal.SIGALRM,alarm);signal.alarm(1300);save()
try:
 cmd=[str(repo/'.venv/bin/python'),'-m','tools.task_default_specialization','--reference',str(repo/a['reference_package_path']/'graygo'),'--candidate',str(repo/a['candidate_package_path']/'graygo'),'--expected-reference',a['reference_source_sha256'],'--expected-candidate',a['candidate_source_sha256'],'--output',str(root/'specialization.json')]
 run('specialization',cmd,'candidate',30)
 proof=json.loads((root/'specialization.json').read_text());assert proof['reference_collection_ast_sha256']==proof['specialized_candidate_ast_sha256']==a['expected_specialized_collection_ast_sha256']
 for role in ('reference','candidate'):
  cmd=[str(repo/'.venv/bin/python'),'-m','tools.task_domain_boundary','--expected-source-sha',a[role+'_source_sha256'],'--time-cap-seconds',str(a['per_package_domain_ceiling_seconds']),'--output',str(root/(role+'.json'))]
  run(role,cmd,role,615)
 sys.path.insert(0,str(frozen));from tools.task_domain_boundary import verify_pair
 comparison=verify_pair(json.loads((root/'reference.json').read_text()),json.loads((root/'candidate.json').read_text()),reference_source=a['reference_source_sha256'],candidate_source=a['candidate_source_sha256'])
 (root/'comparison.json').write_text(json.dumps(comparison,indent=2,sort_keys=True)+'\n');body.update(status='verified_domain_and_default_specialization',current_phase=None,comparison_sha256=sha(root/'comparison.json'),specialization_sha256=sha(root/'specialization.json'))
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 body['source_packages_unchanged']=all(package_sha(repo/a[role+'_package_path'])==a[role+'_source_sha256'] for role in ('reference','candidate'))
 if not body['source_packages_unchanged']:body.update(status='failed',error='original source changed')
 body['measured_driver_seconds']=time.monotonic()-started;save();signal.alarm(0)

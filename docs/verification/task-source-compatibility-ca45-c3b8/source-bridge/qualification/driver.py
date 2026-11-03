from pathlib import Path
import hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/task-source-qualified-cea1fa6'
allocation=frozen/'docs/experiments/task-source-bridge-qualification-allocation.json';a=json.loads(allocation.read_text());started=time.monotonic();child=None
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()
assert sha(allocation)=='64e9e0af16b78a261207b6f7de39759b90033811ca1a7e19aeaba73baf39f10c'
assert sha(frozen/a['qualifier'])==a['qualifier_sha256'] and sha(repo/a['domain_run']/'job.json')==a['domain_job_sha256']
assert not subprocess.check_output(['git','-C',str(frozen),'status','--porcelain'],text=True)
body={'version':1,'status':'running','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'qualifier_sha256':a['qualifier_sha256'],'driver_sha256':sha(__file__),'tool_commit':subprocess.check_output(['git','-C',str(frozen),'rev-parse','HEAD'],text=True).strip(),'new_training_records':0,'new_scored_evaluation_games':0,'new_accelerator_job':False,'a2_complete':False,'execution_authorized':False,'outer_watchdog_seconds':180}
def save():
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
def alarm(*args):raise TimeoutError('source qualification watchdog exhausted')
signal.signal(signal.SIGALRM,alarm);signal.alarm(160);save()
try:
 domain_allocation=repo/'.worktrees/task-source-bridge-446a058'/a['domain_allocation'];assert sha(domain_allocation)==a['domain_allocation_sha256']
 cmd=[str(repo/'.venv/bin/python'),'-m','tools.task_source_bridge','--repo',str(repo),'--domain-run',str(repo/a['domain_run']),'--allocation',str(domain_allocation),'--expected-allocation-sha',a['domain_allocation_sha256'],'--output',str(root/'bridge.json')]
 body['command']=cmd;env=dict(os.environ,PYTHONPATH=str(repo/'.worktrees/frozen-cuda-c3b8b5f/src')+os.pathsep+str(frozen))
 with (root/'stdout.json').open('w') as out,(root/'stderr.log').open('w') as err:
  child=subprocess.Popen(cmd,cwd=frozen,env=env,stdout=out,stderr=err,start_new_session=True);body['qualifier_pid']=child.pid;save();code=child.wait(timeout=120)
 body['returncode']=code
 if code==0:
  proof=json.loads((root/'bridge.json').read_text());assert proof['status']=='qualified_cpu_features_A2_source_bridge' and proof['source_compatibility_qualified'] is True
  assert proof['verifier_sha256']==a['qualifier_sha256'] and proof['reference_source_sha256']==a['reference_source_sha256'] and proof['candidate_source_sha256']==a['candidate_source_sha256']
  assert not proof['a2_complete'] and not proof['execution_authorized'] and all(sha(p)==digest for p,digest in proof['input_files_sha256'].items())
  body.update(status='qualified_cpu_features_A2_source_bridge',bridge_sha256=sha(root/'bridge.json'),original_inputs_unchanged=True)
 else:body['status']='failed'
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 body['measured_driver_seconds']=time.monotonic()-started;save();signal.alarm(0)

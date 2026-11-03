from pathlib import Path
import hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/task-pipeline-corrected-71c404e'
allocation=frozen/'docs/experiments/task-pipeline-corrected-comparison-allocation.json';a=json.loads(allocation.read_text());original=repo/a['original_run']
started=time.monotonic();child=None
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
assert sha(allocation)=='ed0765a9e875b920cdfeeae1501267c8f0c6a97f84d9d248228a898ae7fa6771'
assert sha(frozen/a['corrected_comparison_tool'])==a['corrected_comparison_tool_sha256']
assert sha(frozen/'tools/task_actor_boundary.py')==a['transcript_helper_sha256']
assert not subprocess.check_output(['git','-C',str(frozen),'status','--porcelain'],text=True)
j=json.loads((original/'job.json').read_text());assert j['status']=='failed' and j['original_driver_pid']==a['original_driver_pid']
assert j['original_learning_inputs_unchanged'] and j['original_source_packages_unchanged']
assert [(row['label'],row['returncode']) for row in j['commands']]==[('reference',0),('candidate',0),('comparison',1)]
ids=[str(j['original_driver_pid'])]+[str(row['pid']) for row in j['commands']]
p=subprocess.run(['ps','-p',','.join(ids),'-o','pid='],capture_output=True,text=True);assert p.returncode in (0,1) and not p.stdout.strip()
for role in ('reference','candidate'):
 state=json.loads((original/role/'state.json').read_text());assert state['status']=='complete' and state['completed_cases']==4
 profile=json.loads((original/role/'profile.json').read_text());assert len(profile['results'])==4 and all(v['records']==65536 and v['status']=='complete' for v in profile['results'])
before={str(p):sha(p) for p in original.rglob('*') if p.is_file()}
body={'version':1,'status':'running','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'tool_sha256':a['corrected_comparison_tool_sha256'],
 'tool_commit':subprocess.check_output(['git','-C',str(frozen),'rev-parse','HEAD'],text=True).strip(),'driver_sha256':sha(__file__),'original_failed_job_sha256':sha(original/'job.json'),
 'new_training_records':0,'new_evaluation_games':0,'new_accelerator_job':False,'source_compatibility_qualified':False,'a2_evidence_reusable':False,'execution_authorized':False,'outer_watchdog_seconds':240}
def save():
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
def alarm(*args):raise TimeoutError('corrected read-only comparison watchdog exhausted')
signal.signal(signal.SIGALRM,alarm);signal.alarm(220);save()
try:
 old_allocation=repo/'.worktrees/task-pipeline-parity-d724e06'/a['original_allocation'];assert sha(old_allocation)==a['original_allocation_sha256']
 cmd=[str(repo/'.venv/bin/python'),'-m','tools.task_pipeline_parity','--reference',str(original/'reference'),'--candidate',str(original/'candidate'),'--allocation',str(old_allocation),'--expected-allocation-sha',a['original_allocation_sha256'],'--output',str(root/'comparison.json')]
 env=dict(os.environ,PYTHONPATH=str(repo/'.worktrees/frozen-cuda-c3b8b5f/src')+os.pathsep+str(frozen));body['command']=cmd
 with (root/'stdout.json').open('w') as out,(root/'stderr.log').open('w') as err:
  child=subprocess.Popen(cmd,cwd=frozen,env=env,stdout=out,stderr=err,start_new_session=True);body['comparison_pid']=child.pid;save();code=child.wait(timeout=180)
 body['returncode']=code
 if code==0:
  result=json.loads((root/'comparison.json').read_text());assert result['status']=='four_finite_complete_pipelines_verified' and result['aggregate_new_diagnostic_records']==524288
  body.update(status='verified',comparison_sha256=sha(root/'comparison.json'))
 else:body['status']='failed'
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 body['original_inputs_unchanged']=before=={str(p):sha(p) for p in original.rglob('*') if p.is_file()}
 if not body['original_inputs_unchanged']:body.update(status='failed',error='original diagnostics changed')
 body['measured_driver_seconds']=time.monotonic()-started;save();signal.alarm(0)

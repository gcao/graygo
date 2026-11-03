from pathlib import Path
from decimal import Decimal
import hashlib,json,os,re,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/cpu-cache-suite-fixture-1f7c432';allocation=repo/'docs/experiments/cpu-raw-cache-fixed-suite-allocation.json';started=time.monotonic();child=None
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=load(allocation);assert sha(allocation)=='2c812e3bd5ddb24b3a035990de668d60d40410f3a793f4c244933a2fafa34f81';assert subprocess.check_output(['git','-C',str(frozen),'rev-parse','HEAD'],text=True).strip()==a['candidate_commit'];assert not subprocess.check_output(['git','-C',str(frozen),'status','--porcelain'],text=True)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'another GPU compute process is active'
body={'version':1,'status':'running_complete_frozen_candidate_suite','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'driver_sha256':sha(__file__),'candidate_commit':a['candidate_commit'],'source_sha256':a['candidate_source_sha256'],'gpu_uuid':a['gpu_uuid'],'new_formal_a2_records':0,'source_admission_qualified':False,'execution_authorized':False}
def save():
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
save()
try:
 env=dict(os.environ,PYTHONPATH=str(frozen/'src')+os.pathsep+str(frozen)+os.pathsep+str(frozen/'tests'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 command=[str(repo/'.venv/bin/python'),'-m','unittest','discover','-s','tests','-v'];body['command']=command;body['gpu_allocation_start_unix_seconds']=str(time.time())
 with (root/'suite.stdout').open('wb') as out,(root/'suite.stderr').open('wb') as err:
  tick=time.monotonic();child=subprocess.Popen(command,cwd=frozen,env=env,stdout=out,stderr=err,start_new_session=True);body['original_suite_pid']=child.pid;save();code=child.wait(timeout=a['suite_process_ceiling_seconds']-20);body.update(returncode=code,measured_suite_process_seconds=time.monotonic()-tick,gpu_allocation_end_unix_seconds=str(time.time()))
 body['gpu_equipped_interval_seconds']=str(Decimal(body['gpu_allocation_end_unix_seconds'])-Decimal(body['gpu_allocation_start_unix_seconds']));log=(root/'suite.stderr').read_text();matches=re.findall(r'Ran (\d+) tests? in ([0-9.]+)s',log);body['reported_tests']=int(matches[-1][0]) if matches else None;body['reported_test_seconds']=float(matches[-1][1]) if matches else None;body['status']='complete_suite_passed' if code==0 and matches and re.search(r'\nOK(?:\s|$| \()',log) else 'complete_suite_failed';body['suite_stdout_sha256']=sha(root/'suite.stdout');body['suite_stderr_sha256']=sha(root/'suite.stderr')
except BaseException as exc:body.update(status='failed_or_stopped',error=repr(exc))
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  body['gpu_allocation_end_unix_seconds']=str(time.time());body['gpu_equipped_interval_seconds']=str(Decimal(body['gpu_allocation_end_unix_seconds'])-Decimal(body['gpu_allocation_start_unix_seconds']))
 body['measured_driver_wall_seconds']=time.monotonic()-started;save()

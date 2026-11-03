from pathlib import Path
import hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/terminal-batch-comparison-7d92069';allocation=frozen/'docs/experiments/terminal-batch-comparison-allocation.json';a=json.loads(allocation.read_text());started=time.monotonic();child=None
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(allocation)=='a354fdc72fb775b6ae791dec7b16001799c71a023b75392ea217d046e72196be'
assert sha(frozen/a['producer'])==a['producer_sha256']
assert not subprocess.check_output(['git','-C',str(frozen),'status','--porcelain'],text=True)
body={'version':1,'status':'running','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'producer_sha256':a['producer_sha256'],'driver_sha256':sha(__file__),'source_sha256':a['source_sha256'],'tool_commit':subprocess.check_output(['git','-C',str(frozen),'rev-parse','HEAD'],text=True).strip(),'control_reservation_seconds':a['additional_control_reservation_seconds'],'new_formal_A2_records':0,'new_accelerator_jobs':0,'execution_authorized':False,'outer_watchdog_seconds':2460}
def save():
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
def stop(*args):raise TimeoutError('original terminal batch comparison watchdog exhausted')
signal.signal(signal.SIGALRM,stop);signal.alarm(2420);save()
try:
 env=dict(os.environ,PYTHONPATH=str(frozen/'src')+os.pathsep+str(frozen));cmd=[str(repo/'.venv/bin/python'),'-m','tools.profile_terminal_batch','--run-dir',str(root/'measurement'),'--expected-source-sha',a['source_sha256'],'--time-cap-seconds','2400'];body['command']=cmd
 with (root/'stdout.json').open('w') as out,(root/'stderr.log').open('w') as err:
  child=subprocess.Popen(cmd,cwd=frozen,env=env,stdout=out,stderr=err,start_new_session=True);body['profile_pid']=child.pid;save();code=child.wait(timeout=2400)
 body['returncode']=code
 if code==0:
  report=json.loads((root/'measurement/report.json').read_text());assert report['status']=='complete_exact_pipeline_pairs' and report['source_unchanged'] and report['diagnostic_records']==2124288 and len(report['pairs'])==12
  body.update(status='complete_exact_pipeline_pairs',report_sha256=sha(root/'measurement/report.json'),diagnostic_records=report['diagnostic_records'])
 else:body['status']='failed'
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 body['measured_driver_seconds']=time.monotonic()-started;save();signal.alarm(0)

from pathlib import Path
import hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;allocation=repo/'docs/experiments/prepared-placement-verification-allocation.json'
a=json.loads(allocation.read_text());started=time.monotonic();process=None
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
tool=repo/'tools/audit_prepared_placement.py';assert sha(tool)==a['tool_sha256']
job={'version':1,'status':'running','original_driver_pid':os.getpid(),'source_sha256':a['source_sha256'],'tool_sha256':sha(tool),
 'driver_sha256':sha(Path(__file__)),'allocation_sha256':sha(allocation),'watchdog_seconds':660,'cleanup_signal_at_seconds':630,
 'new_learning_records':0,'new_control_records':0,'kind':'CPU correctness/timing verification; no accelerator or training allocation'}
def save():
 p=root/'job.json';t=root/'job.tmp';t.write_text(json.dumps(job,indent=2,sort_keys=True)+'\n');t.replace(p)
def stop(*args):raise TimeoutError('original exhaustive verification watchdog exhausted')
signal.signal(signal.SIGALRM,stop);signal.alarm(630);save()
try:
 env=dict(os.environ,PYTHONPATH=str(repo/'src'))
 command=[str(repo/'.venv/bin/python'),'-m','tools.audit_prepared_placement','--run-dir',str(root),'--expected-source-sha',a['source_sha256'],'--time-cap-seconds','600']
 job['command']=command;save()
 with (root/'stdout.json').open('w') as out,(root/'stderr.log').open('w') as err:
  process=subprocess.Popen(command,cwd=repo,env=env,stdout=out,stderr=err,start_new_session=True);job['verification_pid']=process.pid;save();code=process.wait()
 job['returncode']=code
 if code==0:
  report=json.loads((root/'report.json').read_text());assert sha(tool)==a['tool_sha256']
  job.update(status='verified' if report['complete_domain_checked'] else 'budget_limited',report_sha256=sha(root/'report.json'),
             complete_domain_checked=report['complete_domain_checked'],original_tool_unchanged=True)
 else:job['status']='failed'
except BaseException as exc:job.update(status='failed',error=repr(exc))
finally:
 if process is not None and process.poll() is None:
  os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
 job['measured_driver_seconds']=time.monotonic()-started;save();signal.alarm(0)

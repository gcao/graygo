from pathlib import Path
import hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/terminal-batch-audit-3695aca';allocation=frozen/'docs/experiments/terminal-batch-independent-audit-allocation.json';a=json.loads(allocation.read_text());original=repo/a['original_run'];started=time.monotonic();child=None
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()
assert sha(allocation)=='37f4037fd996f50931381977839bf78ce137a073e2bc364543af15fc75d174ff'
assert sha(original/'job.json')==a['original_job_sha256'] and sha(original/'measurement/report.json')==a['original_report_sha256']
assert sha(frozen/a['auditor'])==a['auditor_sha256']
before={str(p):sha(p) for case in a['cases'] for p in (original/'measurement'/case).rglob('*') if p.is_file()}
body={'version':1,'status':'running','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'auditor_sha256':a['auditor_sha256'],'driver_sha256':sha(__file__),'commands':[],'new_training_records':0,'formal_A2_case_claimed':False,'outer_watchdog_seconds':900}
def save():
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
def alarm(*args):raise TimeoutError('original terminal-batch independent audit watchdog exhausted')
signal.signal(signal.SIGALRM,alarm);signal.alarm(860);save()
try:
 env=dict(os.environ,PYTHONPATH=str(repo/a['source_package_path'])+os.pathsep+str(frozen))
 for index,case in enumerate(a['cases']):
  output=root/(case.replace('/','-')+'.json');cmd=[str(repo/'.venv/bin/python'),str(frozen/a['auditor']),str(original/'measurement'/case),'--allow-diagnostic','--time-cap-seconds','30','--output',str(output)];row={'case':case,'command':cmd};body['commands'].append(row);body['current_case']=case
  with (root/(str(index)+'.stdout')).open('w') as out,(root/(str(index)+'.stderr')).open('w') as err:
   tick=time.monotonic();child=subprocess.Popen(cmd,cwd=frozen,env=env,stdout=out,stderr=err,start_new_session=True);row['pid']=child.pid;save();code=child.wait(timeout=min(40,800-(time.monotonic()-started)))
  row.update(returncode=code,seconds=time.monotonic()-tick);save()
  if code!=0:raise RuntimeError(case+' independent audit failed')
  report=json.loads(output.read_text());assert report['verified'] and not report['formal_protocol'] and report['source_sha256']==a['source_sha256'];row['result_sha256']=sha(output);save()
 body.update(status='verified_all_24_diagnostics',current_case=None)
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 body['original_inputs_unchanged']=before=={str(p):sha(p) for case in a['cases'] for p in (original/'measurement'/case).rglob('*') if p.is_file()}
 if not body['original_inputs_unchanged']:body.update(status='failed',error='original diagnostic inputs changed')
 body['measured_driver_seconds']=time.monotonic()-started;save();signal.alarm(0)

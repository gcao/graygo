from pathlib import Path
import hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/nfsp-corrected-matrix-audit-84d1c97'
allocation=frozen/'docs/experiments/nfsp-corrected-matrix-profile-audit-allocation.json';a=json.loads(allocation.read_text());started=time.monotonic();child=None
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()
assert sha(allocation)=='5f672ead23b9d3aa1b59050748525e8c383781b6b382113b959004d4d47d6d53'
assert sha(frozen/a['auditor'])==a['auditor_sha256']
assert not subprocess.check_output(['git','-C',str(frozen),'status','--porcelain'],text=True)
before={}
for case in a['cases']:
 directory=repo/a['original_stage_root']/case;result=json.loads((directory/'result.json').read_text());assert result['status']=='complete' and result['records']==100000 and result['source_sha256']==a['learner_source_sha256']
 for p in directory.rglob('*'):
  assert not p.is_symlink()
  if p.is_file():before[str(p)]=sha(p)
body={'version':1,'status':'running','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'auditor_sha256':a['auditor_sha256'],'driver_sha256':sha(__file__),'tool_commit':subprocess.check_output(['git','-C',str(frozen),'rev-parse','HEAD'],text=True).strip(),'commands':[],'new_training_records':0,'new_scored_evaluation_games':0,'a2_complete':False,'execution_authorized':False,'outer_watchdog_seconds':360}
def save():
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
def alarm(*args):raise TimeoutError('original profile-audit watchdog exhausted')
signal.signal(signal.SIGALRM,alarm);signal.alarm(330);save()
try:
 env=dict(os.environ,PYTHONPATH=str(repo/a['learner_package_path'])+os.pathsep+str(frozen))
 for index,case in enumerate(a['cases']):
  output=root/(case.replace('/','-')+'.json');cmd=[str(repo/'.venv/bin/python'),str(frozen/a['auditor']),str(repo/a['original_stage_root']/case),'--allow-diagnostic','--time-cap-seconds','30','--output',str(output)]
  row={'case':case,'command':cmd};body['commands'].append(row)
  with (root/(str(index)+'.stdout')).open('w') as out,(root/(str(index)+'.stderr')).open('w') as err:
   tick=time.monotonic();child=subprocess.Popen(cmd,cwd=frozen,env=env,stdout=out,stderr=err,start_new_session=True);row['pid']=child.pid;body['current_case']=case;save();code=child.wait(timeout=min(40,300-(time.monotonic()-started)))
  row.update(returncode=code,seconds=time.monotonic()-tick);save()
  if code!=0:raise RuntimeError(case+' independent audit failed')
  proof=json.loads(output.read_text());assert proof['verified'] is True and proof['records']==100000 and proof['eligibility_version']==2 and proof['formal_protocol'] is False
  row['result_sha256']=sha(output);save()
 assert len(body['commands'])==8
 body.update(status='verified_all_eight_profiles',current_case=None)
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 body['original_profile_inputs_unchanged']=before=={str(p):sha(p) for case in a['cases'] for p in (repo/a['original_stage_root']/case).rglob('*') if p.is_file()}
 if not body['original_profile_inputs_unchanged']:body.update(status='failed',error='original profile inputs changed')
 body['measured_driver_seconds']=time.monotonic()-started;save();signal.alarm(0)

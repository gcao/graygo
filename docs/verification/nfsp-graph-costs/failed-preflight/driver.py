from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/nfsp-graph-cost-53ed845'
allocation=frozen/'docs/experiments/nfsp-graph-cost-measurement-allocation.json';a=json.loads(allocation.read_text());tool=frozen/'tools/nfsp_graph_costs.py'
started=time.monotonic();epoch=time.time();process=None
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
assert not subprocess.check_output(['git','-C',str(frozen),'status','--porcelain'],text=True)
assert not subprocess.check_output(['git','-C',str((repo/a['learner_package_path']).parent),'status','--porcelain'],text=True)
assert sha(tool)==a['graph_producer_sha256'] and sha(frozen/'tools/nfsp_cost_profile.py')==a['core_producer_sha256']
assert sha(repo/a['catalog'])==a['catalog_file_sha256'] and sha(repo/a['dense_benchmark'])==a['dense_benchmark_file_sha256']
assert sha(repo/a['native_library'])==a['native_library_sha256']
seal=json.loads((repo/a['buffer_checkpoint']/'manifest.json').read_text())
inputs=[repo/a['buffer_checkpoint']/name for name in ['manifest.json','manifest.sha256',*seal['files']]]
inputs += [repo/a[name] for name in ('catalog','dense_benchmark','native_library')];before={str(p):sha(p) for p in inputs}
body={'version':1,'status':'running','original_driver_pid':os.getpid(),'tool_commit':subprocess.check_output(['git','-C',str(frozen),'rev-parse','HEAD'],text=True).strip(),
 'tool_sha256':sha(tool),'driver_sha256':sha(Path(__file__)),'allocation_sha256':sha(allocation),'learner_source_sha256':a['learner_source_sha256'],
 'physical_device_uuid':a['physical_device_uuid'],'allocation_interval_start_epoch':epoch,'outer_watchdog_seconds':1860,'cleanup_signal_at_seconds':1830,
 'production_training_selected':False,'memory_samples':0,'maximum_observed_gpu_used_mib':0}
def save():
 p=root/'job.json';tmp=root/'job.tmp';tmp.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');tmp.replace(p)
def memory():
 r=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
 assert r.returncode==0
 fields=[x.strip() for x in r.stdout.strip().split(',')];assert fields[0]==a['physical_device_uuid'] and len(fields)==3
 row={'epoch':time.time(),'uuid':fields[0],'used_mib':int(fields[1]),'free_mib':int(fields[2])}
 body['memory_samples']+=1;body['maximum_observed_gpu_used_mib']=max(body['maximum_observed_gpu_used_mib'],row['used_mib'])
 with (root/'memory.jsonl').open('a') as out:out.write(json.dumps(row,sort_keys=True)+'\n')
def stop(*args):raise TimeoutError('original batched full-cost watchdog exhausted')
signal.signal(signal.SIGALRM,stop);signal.alarm(1830);save()
try:
 memory()
 env=dict(os.environ,PYTHONPATH=str(repo/a['learner_package_path'])+os.pathsep+str(frozen))
 cmd=[str(repo/'.venv/bin/python'),'-m','tools.nfsp_graph_costs','--run-dir',str(root),'--buffer-checkpoint',str(repo/a['buffer_checkpoint']),
  '--benchmark',str(repo/a['dense_benchmark']),'--catalog',str(repo/a['catalog']),'--native-library',str(repo/a['native_library']),
  '--expected-source-sha',a['learner_source_sha256'],'--expected-components-sha',a['components_sha256'],'--reuse','4','--time-cap-seconds','1800']
 body['command']=cmd;save()
 with (root/'stdout.json').open('w') as stdout,(root/'stderr.log').open('w') as stderr:
  process=subprocess.Popen(cmd,cwd=frozen,env=env,stdout=stdout,stderr=stderr,start_new_session=True);body['profile_pid']=process.pid;save()
  while process.poll() is None:
   try:process.wait(timeout=1)
   except subprocess.TimeoutExpired:memory()
  code=process.returncode
 body.update(returncode=code,original_inputs_unchanged=before=={str(p):sha(p) for p in inputs})
 if code==0:
  r=json.loads((root/'report.json').read_text());assert r['status']=='complete' and body['original_inputs_unchanged']
  assert r['fresh_learner_profile_records']==65792 and r['evaluation_physical_games']==2480 and not r['execution_authorized']
  body.update(status='complete_measurement_unreviewed',report_sha256=sha(root/'report.json'),candidate_price_forecast_available=r['candidate_price_forecast_available'])
 else:body['status']='failed'
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
 if process is not None and process.poll() is None:
  os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
 body.update(measured_driver_seconds=time.monotonic()-started,allocation_interval_end_epoch=time.time());save();signal.alarm(0)

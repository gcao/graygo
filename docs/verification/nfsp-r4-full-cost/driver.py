from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
root=Path(__file__).resolve().parent;repo=Path('/home/gcao/proj/graygo');source=repo/'.worktrees/nfsp-cost-0c0228a';tool=source/'tools/nfsp_cost_profile.py'
allocation=source/'docs/experiments/nfsp-r4-cost-measurement-allocation.json';a=json.loads(allocation.read_text());started=time.monotonic();start_epoch=time.time();process=None
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
assert sha(tool)=='e8c9f6721242b0ebc880a6b7207f0faa4b7582c05149632e08658789c920b73a'
assert sha(repo/a['dense_benchmark'])==a['dense_benchmark_file_sha256'] and sha(repo/a['native_library'])==a['native_library_sha256']
inputs=[repo/a['buffer_checkpoint']/'manifest.json',repo/a['buffer_checkpoint']/'replay.npz',repo/a['dense_benchmark'],repo/a['native_library']];before={str(p):sha(p) for p in inputs}
body={'version':1,'status':'running','original_driver_pid':os.getpid(),'tool_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),'tool_sha256':sha(tool),'driver_sha256':sha(Path(__file__)),'allocation_sha256':sha(allocation),'learner_source_sha256':a['learner_source_sha256'],'physical_device_uuid':a['physical_device_uuid'],'allocation_interval_start_epoch':start_epoch,'outer_watchdog_seconds':3660,'production_training_selected':False}
def save():
    p=root/'job.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');tmp.replace(p)
def stop(*args):raise TimeoutError('original complete-cost profile watchdog exhausted')
signal.signal(signal.SIGALRM,stop);signal.alarm(3660);save()
try:
    env=dict(os.environ,PYTHONPATH=str(repo/a['learner_package_path']))
    cmd=[sys.executable,str(tool),'--run-dir',str(root),'--buffer-checkpoint',str(repo/a['buffer_checkpoint']),'--benchmark',str(repo/a['dense_benchmark']),'--native-library',str(repo/a['native_library']),'--expected-source-sha',a['learner_source_sha256'],'--reuse','4','--device','cuda','--time-cap-seconds','3600']
    body['command']=cmd;save()
    with (root/'stdout.json').open('w') as stdout,(root/'stderr.log').open('w') as stderr:
        process=subprocess.Popen(cmd,cwd=source,env=env,stdout=stdout,stderr=stderr,start_new_session=True);body['profile_pid']=process.pid;save()
        code=process.wait(timeout=max(1,3660-(time.monotonic()-started)))
    body.update(returncode=code,original_inputs_unchanged=before=={str(p):sha(p) for p in inputs})
    if code==0:
        r=json.loads((root/'report.json').read_text());assert r['status']=='complete' and body['original_inputs_unchanged']
        assert r['fresh_profile_records']==65792 and not r['execution_authorized'] and not r['formal_learning_run']
        body.update(status='complete_measurement_unreviewed',report_sha256=sha(root/'report.json'),candidate_price_forecast_available=r['candidate_price_forecast_available'])
    else:body['status']='failed'
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
    if process is not None and process.poll() is None:
        process.terminate()
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:process.kill();process.wait()
    body.update(measured_driver_seconds=time.monotonic()-started,allocation_interval_end_epoch=time.time());save();signal.alarm(0)

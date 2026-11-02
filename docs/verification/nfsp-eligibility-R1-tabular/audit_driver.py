from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path(__file__).resolve().parent
started=time.monotonic()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def save(body):
    p=root/'tabular-audit-job.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');tmp.replace(p)
inputs=sorted(p for p in (root/'tabular').rglob('*') if p.is_file())
before={str(p.relative_to(root)):sha(p) for p in inputs}
cmd=[sys.executable,str(root/'audit_nfsp_delayed.py'),str(root/'tabular'),'--budget-profile',str(root/'tabular-budget.json'),'--output',str(root/'tabular-audit.json')]
body={'version':1,'status':'running','original_driver_pid':os.getpid(),'command':cmd,'watchdog_seconds':600,'source_sha256':'db9db2eb532d0a2782486a486743701a0a1914277616ff5d5c0406f3b6501873','auditor_sha256':sha(root/'audit_nfsp_delayed.py'),'driver_sha256':sha(Path(__file__)),'new_learning_records':0}
save(body)
try:
    with (root/'tabular-audit.log').open('w') as log:
        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);body['audit_pid']=process.pid;save(body)
        try:code=process.wait(timeout=600)
        except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=30);body['status']='watchdog_exhausted';raise
    body['returncode']=code;body['original_inputs_unchanged']=before=={str(p.relative_to(root)):sha(p) for p in inputs}
    if code==0:
        value=json.loads((root/'tabular-audit.json').read_text());body['audit_sha256']=sha(root/'tabular-audit.json')
        body['status']='verified' if value['verified'] and body['original_inputs_unchanged'] else 'failed'
    else:body['status']='failed'
except BaseException as exc:
    body['error']=repr(exc)
    if body['status']=='running':body['status']='failed'
finally:
    body['elapsed_driver_seconds']=time.monotonic()-started;save(body)

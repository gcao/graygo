from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path(__file__).resolve().parent;started=time.monotonic();(root/'audits').mkdir()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
body={'version':1,'status':'running','original_driver_pid':os.getpid(),'auditor_sha256':sha(root/'audit_nfsp_delayed.py'),'driver_sha256':sha(Path(__file__)),'per_case_watchdog_seconds':60,'new_learning_records':0,'cases':[]}
def save():
    p=root/'audit-job.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');tmp.replace(p)
save()
try:
    for name in ('tabular-R4-pre','tabular-R4-post','neural-R4-pre','neural-R4-post'):
        directory=root/name;inputs=sorted(p for p in directory.rglob('*') if p.is_file());before={str(p.relative_to(directory)):sha(p) for p in inputs}
        cmd=[sys.executable,str(root/'audit_nfsp_delayed.py'),str(directory),'--allow-diagnostic','--output',str(root/'audits'/(name+'.json'))]
        with (root/'audits'/(name+'.log')).open('w') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);body.update(current_case=name,child_pid=p.pid);save()
            try:code=p.wait(timeout=60)
            except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30);raise
        assert code==0,name+' independent audit failed'
        a=json.loads((root/'audits'/(name+'.json')).read_text())
        assert a['verified'] and a['records']==65536 and a['q_eligible_records']==65461 and a['q_updates']==480 and a['target_refreshes']==5
        assert before=={str(p.relative_to(directory)):sha(p) for p in inputs}
        body['cases'].append({'name':name,'verified':True,'original_inputs_unchanged':True,'audit_sha256':sha(root/'audits'/(name+'.json')),'training_manifest_sha256':a['training_manifest_sha256'],'checkpoint_manifest_sha256':a['checkpoint_manifest_sha256']});save()
    body.update(status='verified',current_case=None,completed_cases=4)
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:body['measured_driver_seconds']=time.monotonic()-started;save()

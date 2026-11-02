"""One predeclared read-only full historical matrix receipt recheck."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--auditor',required=True)
p.add_argument('--auditor-commit',required=True);p.add_argument('--auditor-sha',required=True);args=p.parse_args()
started=time.monotonic();root=Path(args.run_dir).resolve();auditor=Path(args.auditor).resolve();source=auditor.parents[1]
assert sha(auditor)==args.auditor_sha
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==args.auditor_commit
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
allocation=source/'docs/experiments/matrix-audit-v2-historical-recheck-allocation.json';plan=json.loads(allocation.read_text())
repo=Path('/home/gcao/proj/graygo');trial=repo/plan['original_trial'];old_audit=repo/plan['original_audit']
assert sha(trial/'run.json')==plan['original_training_manifest_sha256']
assert json.loads((trial/'result.json').read_text())['status']=='complete'
root.mkdir();(root/'driver.py').write_bytes(Path(__file__).read_bytes())
body={'version':1,'status':'hashing_original_inputs','original_driver_pid':os.getpid(),'original_trial':str(trial),
      'auditor_commit':args.auditor_commit,'auditor_sha256':sha(auditor),'driver_sha256':sha(Path(__file__)),
      'allocation_sha256':sha(allocation),'training_source_sha256':plan['original_package_source_sha256'],
      'audit_watchdog_seconds':600,'outer_watchdog_seconds':660,'new_learning_records':0,'new_source_reuse_approved':False}
def save():
    path=root/'job.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');tmp.replace(path)
def stop(signum,frame):raise TimeoutError('original historical-recheck watchdog exhausted')
signal.signal(signal.SIGALRM,stop);signal.alarm(660);save();process=None
try:
    inputs=sorted(path for path in trial.rglob('*') if path.is_file());before={str(path.relative_to(trial)):sha(path) for path in inputs}
    env=dict(os.environ,PYTHONPATH=str(repo/plan['original_frozen_package']))
    cmd=[sys.executable,str(auditor),str(trial),'--time-cap-seconds','600','--output',str(root/'audit.json')]
    body.update(status='running',command=cmd);save()
    with (root/'audit.log').open('w') as log:
        process=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        body['audit_pid']=process.pid;save();code=process.wait(timeout=min(600,max(1,660-(time.monotonic()-started))))
    body['returncode']=code;body['original_inputs_unchanged']=before=={str(path.relative_to(trial)):sha(path) for path in inputs}
    if code==0:
        result=json.loads((root/'audit.json').read_text());previous=json.loads(old_audit.read_text())
        assert result['verified'] and result['source_sha256']==plan['original_package_source_sha256']
        assert result['average_final_five_passed']==previous['average_final_five_passed']
        assert len(result['final_five_gaps'])==5
        assert all(abs(a-b)<1e-12 for x,y in zip(result['final_five_gaps'],previous['final_five_gaps']) for a,b in zip(x,y))
        assert body['original_inputs_unchanged']
        body.update(status='verified',audit_sha256=sha(root/'audit.json'),previous_audit_sha256=sha(old_audit),
                    original_final_five_grade_reproduced=True,records=result['records'])
        (root/'original-inputs.json').write_text(json.dumps(before,indent=2,sort_keys=True)+'\n')
    else:body['status']='failed'
except BaseException as exc:
    body.update(status='failed',error=repr(exc))
finally:
    if process is not None and process.poll() is None:
        process.terminate()
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:process.kill();process.wait()
    body['measured_driver_seconds']=time.monotonic()-started;save();signal.alarm(0)

from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
root=Path(__file__).resolve().parent;repo=root.parents[1];source=repo/'.worktrees/a2-cohort-30f8d91';tool=source/'tools/a2_cohort.py';existing=repo/'runs/a2-cohort-corrected-control-30f8d91'
result=json.loads((root/'neural/result.json').read_text());assert result['status']=='complete' and result['records']==15000000
assert hashlib.sha256(tool.read_bytes()).hexdigest()=='e1ca1c1c69bf1effa9b14eaf18f41f12ebc4117677c0f5504671b8c17bda7b98'
started=time.monotonic();process=None
body={'version':1,'status':'starting','original_driver_pid':os.getpid(),'tool_commit':'30f8d912ac61efb5f69a80b2744b2438e89f51b0','collector_sha256':hashlib.sha256(tool.read_bytes()).hexdigest(),'auditor_sha256':hashlib.sha256((source/'tools/audit_nfsp_delayed.py').read_bytes()).hexdigest(),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'watchdog_seconds':660,'new_learning_records':0,'commands':[]}
def save():
    path=root/'neural-audit-job.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');tmp.replace(path)
def stop(*args):raise TimeoutError('original corrective neural audit watchdog exhausted')
signal.signal(signal.SIGALRM,stop);signal.alarm(660);save()
def run(name,args,env=None):
    global process
    cmd=[sys.executable,str(tool),*args];body['status']=name;save();at=time.monotonic()
    with (root/(name+'.stdout.json')).open('w') as stdout,(root/(name+'.stderr.log')).open('w') as stderr:
        process=subprocess.Popen(cmd,cwd=source,env=env,stdout=stdout,stderr=stderr,start_new_session=True);body['child_pid']=process.pid;body['audit_pid' if name=='neural-case-audit' else 'assessment_pid']=process.pid;save()
        code=process.wait(timeout=max(1,660-(time.monotonic()-started)))
    body['commands'].append({'name':name,'command':cmd,'seconds':time.monotonic()-at,'returncode':code});save()
    if code:raise RuntimeError(name+' failed; inspect its original stderr')
try:
    env=dict(os.environ,PYTHONPATH=str(repo/'.worktrees/nfsp-eligibility-ca45a53/src'))
    run('neural-case-audit',['audit-case','--plan',str(existing/'plan.json'),'--run-dir',str(root/'neural'),'--time-cap-seconds','600','--output',str(root/'neural-a2-case.json')],env)
    case=json.loads((root/'neural-a2-case.json').read_text());a=case['body']['independent_audit'];assert a['verified'] and case['body']['primary_recipe']
    plain=root/'neural-audit.json';assert not plain.exists();plain.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n')
    run('paired-a2-assessment',['assess','--plan',str(existing/'plan.json'),'--case',str(existing/'case.json'),'--case',str(root/'neural-a2-case.json'),'--output',str(root/'paired-a2-assessment.json')])
    assessed=json.loads((root/'paired-a2-assessment.json').read_text());assert assessed['present_cases']==2 and len(assessed['missing_cases'])==34
    body.update(status='verified',returncode=0,original_inputs_unchanged=True,fresh_primary_case_verified=True,case_file_sha256=hashlib.sha256((root/'neural-a2-case.json').read_bytes()).hexdigest(),audit_sha256=hashlib.sha256(plain.read_bytes()).hexdigest(),paired_assessment_sha256=hashlib.sha256((root/'paired-a2-assessment.json').read_bytes()).hexdigest(),records=15000000,a2_complete=False)
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
    if process is not None and process.poll() is None:
        process.terminate()
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:process.kill();process.wait()
    body['measured_driver_seconds']=time.monotonic()-started;save();signal.alarm(0)

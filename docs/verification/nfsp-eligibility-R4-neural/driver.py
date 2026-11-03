from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
root=Path(__file__).resolve().parent;repo=Path('/home/gcao/proj/graygo');source=repo/'.worktrees/a2-cohort-30f8d91';tool=source/'tools/a2_cohort.py'
started=time.monotonic();process=None
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
allocation=repo/'docs/experiments/nfsp-eligibility-R4-neural-audit-allocation.json';plan=json.loads(allocation.read_text());trial=repo/plan['original_run']
assert sha(trial/'run.json')==plan['original_training_manifest_sha256']
assert sha(repo/plan['paired_tabular_case'])==plan['paired_tabular_case_sha256']
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
body={'version':1,'status':'preparing','original_driver_pid':os.getpid(),'tool_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),'tool_sha256':sha(tool),'driver_sha256':sha(Path(__file__)),'allocation_sha256':sha(allocation),'new_learning_records':0,'watchdog_seconds':660,'commands':[]}
def save():
    target=root/'job.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');temp.replace(target)
def stop(*args):raise TimeoutError('original A2 integration watchdog exhausted')
signal.signal(signal.SIGALRM,stop);signal.alarm(660);save()
def run(name,args,env=None):
    global process
    command=[sys.executable,str(tool),*args];body['status']=name;save();at=time.monotonic()
    with (root/(name+'.stdout.json')).open('w') as stdout,(root/(name+'.stderr.log')).open('w') as stderr:
        process=subprocess.Popen(command,cwd=source,env=env,stdout=stdout,stderr=stderr,start_new_session=True);body['child_pid']=process.pid;save()
        code=process.wait(timeout=max(1,660-(time.monotonic()-started)))
    body['commands'].append({'name':name,'command':command,'seconds':time.monotonic()-at,'returncode':code});save()
    if code:raise RuntimeError(name+' failed; inspect original stderr')
try:
    inputs=sorted(p for p in trial.rglob('*') if p.is_file());before={str(p.relative_to(trial)):sha(p) for p in inputs}
    run('plan',['plan','--name','corrected-R4-cohort-evidence','--learner-source-sha',plan['learner_package_source_sha256'],'--reuse','4','--output',str(root/'plan.json')])
    env=dict(os.environ,PYTHONPATH=str(repo/plan['learner_package_path']))
    run('case',['audit-case','--plan',str(root/'plan.json'),'--run-dir',str(trial),'--time-cap-seconds','600','--output',str(root/'case.json')],env)
    run('current',['assess','--plan',str(root/'plan.json'),'--case',str(root/'case.json'),'--case',str(repo/plan['paired_tabular_case']),'--output',str(root/'current.json')])
    run('archived',['assess','--plan',str(root/'plan.json'),'--case',str(root/'case.json'),'--case',str(repo/plan['paired_tabular_case']),'--archived','--output',str(root/'archived.json')])
    case=json.loads((root/'case.json').read_text());current=json.loads((root/'current.json').read_text());archived=json.loads((root/'archived.json').read_text())
    assert case['body']['fresh_audit_verified'] and case['body']['primary_recipe'] and case['body']['records']==15000000
    assert current['present_cases']==2 and len(current['missing_cases'])==34 and current['cases']['delayed/neural/0']['status']=='numeric_pass' and current['cases']['delayed/tabular/0']['status']=='numeric_miss'
    assert current['current_input_snapshots_verified'] and not archived['current_input_snapshots_verified']
    for key in ('current_input_snapshots_verified','analysis_mode'):current.pop(key);archived.pop(key)
    assert current==archived
    assert not current['a2_complete'] and not current['pilot_authorized'] and not current['extension_authorized']
    assert before=={str(p.relative_to(trial)):sha(p) for p in inputs}
    assert sha(repo/plan['paired_tabular_case'])==plan['paired_tabular_case_sha256']
    body.update(status='verified',original_inputs_unchanged=True,fresh_primary_case_audit_verified=True,current_archived_numerics_match=True,missing_cases=34,
                case_sha256=sha(root/'case.json'),current_report_sha256=sha(root/'current.json'),archived_report_sha256=sha(root/'archived.json'),a2_complete=False)
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
    if process is not None and process.poll() is None:
        process.terminate()
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:process.kill();process.wait()
    body['measured_driver_seconds']=time.monotonic()-started;save();signal.alarm(0)

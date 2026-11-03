from pathlib import Path
import hashlib,json,os,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/a2-source-admission-584213f';a_path=frozen/'docs/experiments/terminal-batch-source-admission-allocation.json';a=json.loads(a_path.read_text());started=time.monotonic()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(a_path)=='5fbbbbbf2e1199cc7cc241af4f7e5511a34765807cc900d3d072d5a59e7b6a29'
assert sha(frozen/a['admission_tool'])==a['admission_tool_sha256'] and sha(frozen/a['cohort_reader'])==a['cohort_reader_sha256']
body={'version':1,'status':'running','original_driver_pid':os.getpid(),'allocation_sha256':sha(a_path),'driver_sha256':sha(__file__),'commands':[],'new_training_records':0,'a2_complete':False,'execution_authorized':False}
def save():
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
env=dict(os.environ,PYTHONPATH=str(frozen/'src')+os.pathsep+str(frozen))
def run(label,cmd):
 row={'label':label,'command':cmd};body['commands'].append(row)
 with (root/(label+'.stdout')).open('w') as out,(root/(label+'.stderr')).open('w') as err:
  tick=time.monotonic();child=subprocess.Popen(cmd,cwd=repo,env=env,stdout=out,stderr=err,start_new_session=True);row['pid']=child.pid;save()
  try:code=child.wait(timeout=min(90,180-(time.monotonic()-started)))
  except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=10);raise
 row.update(returncode=code,seconds=time.monotonic()-tick);save()
 assert code==0,label+' failed'
save()
try:
 py=str(repo/'.venv/bin/python')
 run('admission',[py,'-m','tools.terminal_batch_admission','--reference',str(repo/a['reference_package']),'--candidate',str(repo/a['candidate_package']),'--expected-reference',a['reference_source_sha256'],'--expected-candidate',a['candidate_source_sha256'],'--comparison',str(repo/a['comparison_run']),'--audits',str(repo/a['independent_audit_run']),'--suite',str(repo/a['full_suite_run']),'--output',str(root/'admission.json')])
 collector=repo/'.worktrees/a2-cohort-30f8d91/tools/a2_cohort.py'
 run('plan',[py,str(collector),'plan','--name','same-protocol-terminal-batch-A2','--learner-source-sha',a['candidate_source_sha256'],'--reuse','4','--output',str(root/'optimized-plan.json')])
 run('registry',[py,'-m','tools.a2_source_cohort','register','--repo',str(repo),'--name','original-and-terminal-batch-primary-A2','--canonical-plan',str(repo/a['canonical_cohort_plan']),'--plan',str(repo/a['canonical_cohort_plan']),'--plan',str(root/'optimized-plan.json'),'--bridge',str(repo/a['base_source_bridge']),'--bridge',str(root/'admission.json'),'--target-runtime-source',a['reference_source_sha256'],'--output',str(root/'registry.json')])
 cases=[repo/'runs/nfsp-eligibility-R4-neural-audit/case.json',repo/'runs/nfsp-eligibility-R4-tabular-audit/case.json']+[repo/'runs/nfsp-corrected-matrix-cohort-audit-00486de'/('case-'+task+'-'+learner+'-0.json') for task in ('asymmetric','rps') for learner in ('neural','tabular')]
 command=[py,'-m','tools.a2_source_cohort','assess','--repo',str(repo),'--registry',str(root/'registry.json')]
 for case in cases:command+=['--case',str(case)]
 run('current',command+['--output',str(root/'current.json')]);run('archived',command+['--archived','--output',str(root/'archived.json')])
 left,right=json.loads((root/'current.json').read_text()),json.loads((root/'archived.json').read_text())
 for key in ('analysis_mode','current_input_snapshots_verified','current_source_admission_inputs_verified'):left.pop(key);right.pop(key)
 assert left==right and left['present_cases']==6 and len(left['missing_cases'])==30
 legacy=json.loads((repo/'runs/nfsp-corrected-matrix-cohort-audit-00486de/assessment-04-current.json').read_text())
 for key in ('cases','neural','controls','missing_cases','marginal_extension','neural_deployment_band','worst_delayed_average_gap'):assert left[key]==legacy[key],key
 body.update(status='qualified_and_registered_same_protocol_sources',admission_sha256=sha(root/'admission.json'),registry_sha256=sha(root/'registry.json'),current_sha256=sha(root/'current.json'),archived_sha256=sha(root/'archived.json'),original_six_case_numerics_unchanged=True)
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:body['measured_driver_seconds']=time.monotonic()-started;save()

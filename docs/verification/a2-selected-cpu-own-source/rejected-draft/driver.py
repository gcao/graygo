from pathlib import Path
import hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/a2-selected-cpu-registration-537bcb9';allocation=frozen/'docs/experiments/a2-selected-cpu-plan-registration-allocation.json';started=time.monotonic();child=None
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=load(allocation);assert sha(allocation)=='5886491c34effc187c0e1aabd7baf8e965296777058b1b3205dc4642dacec3d5';assert sha(repo/a['collector'])==a['collector_sha256'] and sha(frozen/a['reader'])==a['reader_sha256'];assert sha(repo/a['original_source_registry'])==a['original_source_registry_sha256'] and sha(repo/a['cpu_price_receipt'])==a['cpu_price_receipt_sha256']
for p,h in a['original_case_files_sha256'].items():assert sha(repo/p)==h
for p,h in a['bridge_files_sha256'].items():assert sha(repo/p)==h
from graygo.manifest import source_digest
assert source_digest()==a['target_runtime_source_sha256']
body={'version':1,'status':'registering_own_source_selected_cpu_plan','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'driver_sha256':sha(__file__),'commands':[],'new_training_records':0,'new_scored_evaluation_games':0,'new_accelerator_jobs':0,'execution_authorized':False}
def save():
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
def run(label,cmd):
 global child
 row={'label':label,'command':cmd};body['commands'].append(row);save()
 with (root/(label+'.stdout')).open('wb') as out,(root/(label+'.stderr')).open('wb') as err:
  tick=time.monotonic();child=subprocess.Popen(cmd,cwd=frozen,env=os.environ.copy(),stdout=out,stderr=err,start_new_session=True);row['pid']=child.pid;save();code=child.wait(timeout=30);row.update(returncode=code,seconds=time.monotonic()-tick)
 save();assert code==0,label

def alarm(*args):raise TimeoutError('own-source CPU plan registration watchdog exhausted')
signal.signal(signal.SIGALRM,alarm);signal.alarm(175);save()
try:
 python=str(repo/'.venv/bin/python');plan=root/'selected-plan.json';registry=root/'registry.json'
 run('plan',[python,str(repo/a['collector']),'plan','--name','same-protocol-selected-CPU-A2','--learner-source-sha',a['target_runtime_source_sha256'],'--reuse','4','--output',str(plan)])
 cmd=[python,'-m','tools.a2_source_cohort','register','--name','original-optimized-and-selected-CPU-primary-A2','--canonical-plan',str(repo/a['canonical_plan']),'--repo',str(repo),'--target-runtime-source',a['target_runtime_source_sha256'],'--output',str(registry)]
 for p in [repo/a['canonical_plan'],repo/a['existing_optimized_plan'],plan]:cmd+=['--plan',str(p)]
 for p in a['bridges']:cmd+=['--bridge',str(repo/p)]
 run('register',cmd)
 common=[python,'-m','tools.a2_source_cohort','assess','--repo',str(repo)]
 for p in a['original_case_files_sha256']:common+=['--case',str(repo/p)]
 run('old-current',common+['--registry',str(repo/a['original_source_registry']),'--output',str(root/'old-current.json')])
 run('new-current',common+['--registry',str(registry),'--output',str(root/'new-current.json')])
 run('new-archived',common+['--registry',str(registry),'--archived','--output',str(root/'new-archived.json')])
 left,right=load(root/'old-current.json'),load(root/'new-current.json');left.pop('registry_sha256');right.pop('registry_sha256');assert left==right
 left,right=load(root/'new-current.json'),load(root/'new-archived.json')
 for key in ('analysis_mode','current_input_snapshots_verified','current_source_admission_inputs_verified'):left.pop(key);right.pop(key)
 assert left==right and not left['a2_complete'] and left['controls']['case_misses']==['delayed/tabular/0']
 assert all(sha(repo/p)==h for p,h in a['original_case_files_sha256'].items())
 body.update(status='registered_original_selected_cpu_plan_with_unchanged_prior_cases',source_plan_sha256=sha(plan),registry_sha256=sha(registry),baseline_present_cases=left['present_cases'],baseline_missing_cases=len(left['missing_cases']),original_case_files_unchanged=True,old_new_numerics_identical=True,current_archived_numerics_identical=True)
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 body['measured_driver_seconds']=time.monotonic()-started;save();signal.alarm(0)

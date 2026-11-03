from pathlib import Path
import hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/nfsp-delayed-cpu-price-qualification-1a98f8e';allocation=frozen/'docs/experiments/nfsp-delayed-direct-cpu-price-qualification-allocation.json';started=time.monotonic();child=None
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=load(allocation);assert sha(allocation)=='f6f0e3d7531fcd555a85d908fd32453a2ac0bfb1453d028fa91866f1d8c56541';assert sha(frozen/a['qualifier'])==a['qualifier_sha256'];assert sha(repo/a['proposal'])==a['proposal_sha256'];assert sha(repo/a['audits']/'job.json')==a['audit_job_sha256'];assert sha(repo/a['selected_profiles']/'job.json')==a['selected_profile_job_sha256']
from graygo.manifest import source_digest
proposal=load(repo/a['proposal']);assert source_digest()==proposal['target_runtime_source_sha256']
body={'version':1,'status':'qualifying_direct_cpu_prices','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'driver_sha256':sha(__file__),'qualifier_sha256':a['qualifier_sha256'],'loaded_package_sha256':source_digest(),'new_training_records':0,'new_scored_evaluation_games':0,'new_accelerator_jobs':0,'execution_authorized':False}
def save():
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
save()
try:
 cmd=[str(repo/'.venv/bin/python'),'-m','tools.nfsp_delayed_cpu_prices','--repo',str(repo),'--proposal',str(repo/a['proposal']),'--audits',str(repo/a['audits']),'--selected',str(repo/a['selected_profiles']),'--output',str(root/'prices.json')];body['command']=cmd;save()
 with (root/'qualifier.stdout').open('wb') as out,(root/'qualifier.stderr').open('wb') as err:
  tick=time.monotonic();child=subprocess.Popen(cmd,cwd=frozen,env=os.environ.copy(),stdout=out,stderr=err,start_new_session=True);body['original_qualifier_pid']=child.pid;save();code=child.wait(timeout=a['qualification_outer_seconds']);body.update(returncode=code,measured_qualifier_process_seconds=time.monotonic()-tick)
 assert code==0,'price qualifier failed; preserve original evidence'
 proof=load(root/'prices.json');assert proof['status']=='qualified_prospective_cpu_delayed_prices' and proof['direct_selected_package_profiles_verified'] and not proof['execution_authorized'];body['prices_sha256']=sha(root/'prices.json');body['status']='qualified_direct_selected_package_cpu_delayed_prices'
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 body['measured_driver_seconds']=time.monotonic()-started;save()

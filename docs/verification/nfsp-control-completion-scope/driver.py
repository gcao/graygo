from pathlib import Path
import hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');root=Path(__file__).resolve().parent;frozen=repo/'.worktrees/nfsp-control-scope-proposal-d868386';allocation=frozen/'docs/experiments/nfsp-control-scope-proposal-allocation.json';started=time.monotonic();child=None
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=load(allocation);assert sha(allocation)=='2e5016f8c90a10a13ba9fddb1c360b8e60e666f3c28b7614a5992b6a0407a7f2';assert sha(frozen/a['builder'])==a['builder_sha256'] and sha(repo/a['evidence'])==a['evidence_sha256'];assert not (repo/a['output']).exists()
body={'version':1,'status':'building_unselected_control_scope_proposal','original_driver_pid':os.getpid(),'allocation_sha256':sha(allocation),'builder_sha256':a['builder_sha256'],'driver_sha256':sha(__file__),'new_training_records':0,'new_scored_evaluation_games':0,'new_accelerator_jobs':0,'owner_approved':False,'execution_authorized':False}
def save():
 p=root/'job.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'job.json')
save()
try:
 cmd=[str(repo/'.venv/bin/python'),'-m','tools.nfsp_control_budget','--repo',str(repo),'--evidence',str(repo/a['evidence']),'--output',str(repo/a['output'])];body['command']=cmd;save()
 with (root/'builder.stdout').open('wb') as out,(root/'builder.stderr').open('wb') as err:
  tick=time.monotonic();child=subprocess.Popen(cmd,cwd=frozen,env=os.environ.copy(),stdout=out,stderr=err,start_new_session=True);body['original_builder_pid']=child.pid;save();code=child.wait(timeout=a['outer_watchdog_seconds']);body.update(returncode=code,measured_builder_process_seconds=time.monotonic()-tick)
 assert code==0,'proposal builder failed; original inputs retained';proposal=load(repo/a['output']);assert proposal['selected'] is False and proposal['owner_approved'] is False and proposal['case_count']==13 and proposal['primary_case_count']==11 and proposal['comparison_case_count']==2 and not proposal['execution_authorized'];body.update(status='built_unselected_original_control_scope',proposal_sha256=sha(repo/a['output']),proposed_control_ceiling_hours=proposal['proposed_control_ceiling_hours'],projected_control_commitment_seconds=proposal['projected_control_commitment_seconds'])
except BaseException as exc:body.update(status='failed',error=repr(exc))
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 body['measured_driver_wall_seconds']=time.monotonic()-started;save()

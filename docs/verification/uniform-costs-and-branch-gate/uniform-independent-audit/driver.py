from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
repo=Path("/home/gcao/proj/graygo");root=Path(__file__).resolve().parent;checkout=repo/".worktrees/uniform-cost-f416a9a"
allocation=checkout/"docs/experiments/nfsp-uniform-board-component-supplement-allocation.json"
audit_allocation=checkout/"docs/experiments/nfsp-uniform-board-supplement-audit-allocation.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
assert sha(allocation)=="2daee784272abcb0fd6af6c35699a8fd660f72fa384f298cb6afc2fcbf0898e7"
assert sha(audit_allocation)=="1be6edd68f374ca6ec3bab3a315db8043f9d2b03f9919c6189d9b1e23fdfe5f9"
a=load(allocation);scope=load(audit_allocation);start=time.monotonic();child=None
assert sha(checkout/a["auditor"])==scope["auditor_sha256"]==a["auditor_sha256"]
env=dict(os.environ,PYTHONPATH=str(repo/a["source_package"])+":"+str(repo/a["base_tooling_root"])+":"+str(checkout),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
command=[str(repo/".venv/bin/python"),str(checkout/a["auditor"]),"--repo",str(repo),"--run-dir",str(repo/"runs/nfsp-uniform-board-cost-f416a9a"),"--allocation",str(allocation),"--expected-allocation-sha",sha(allocation),"--producer",str(checkout/a["producer"]),"--pipeline-producer",str(checkout/a["pipeline_producer"]),"--output",str(root/"audit.json")]
job={"version":1,"status":"auditing_original_uniform_board_component","original_driver_pid":os.getpid(),"driver_sha256":sha(__file__),"audit_allocation_sha256":sha(audit_allocation),"measurement_allocation_sha256":sha(allocation),"auditor_sha256":sha(checkout/a["auditor"]),"command":command,"new_training_records":0,"new_accelerator_jobs":0,"new_scored_evaluation_games":0,"new_control_allocation_seconds":"0","a2_complete":False,"execution_authorized":False}
def save():
 job["measured_driver_wall_seconds"]=time.monotonic()-start
 p=root/"job.tmp";p.write_text(json.dumps(job,indent=2,sort_keys=True)+"\n");p.replace(root/"job.json")
try:
 with (root/"audit.stdout").open("wb") as out,(root/"audit.stderr").open("wb") as err:
  tick=time.monotonic();child=subprocess.Popen(command,cwd=checkout,env=env,stdout=out,stderr=err,start_new_session=True);job["original_audit_pid"]=child.pid;save()
  job["returncode"]=child.wait(timeout=scope["outer_watchdog_seconds"]-10);job["measured_process_seconds"]=time.monotonic()-tick
 assert job["returncode"]==0
 report=load(root/"audit.json");assert report["status"]=="verified_uniform_board_component_forecast"
 job.update(status=report["status"],audit_sha256=sha(root/"audit.json"),B_U_board_device_hours=report["independent_calculation"]["B_U_board_device_hours"])
except BaseException as error:job.update(status="original_uniform_component_audit_failed_preserved",error=repr(error));traceback.print_exc()
finally:
 if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
 save()

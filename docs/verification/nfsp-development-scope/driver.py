from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
repo=Path("/home/gcao/proj/graygo");root=Path(__file__).resolve().parent;checkout=repo/".worktrees/development-scope-212afa0"
allocation=checkout/"docs/experiments/nfsp-development-scope-preview-allocation.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(allocation)=="d2c6758836bf4cf38d94dba64fec2e43825390cab91994d2d020221291860ec4"
a=json.loads(allocation.read_text());start=time.monotonic();child=None
env=dict(os.environ,PYTHONPATH=str(repo/a["source_package"])+":"+str(repo/a["source_verifier_checkout"])+":"+str(checkout),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
command=[str(repo/".venv/bin/python"),str(checkout/a["policy"]),"--plan",str(checkout/a["resource_plan"]),"--output",str(root/"review.json")]
job={"version":1,"status":"reviewing_unselected_development_scope","original_driver_pid":os.getpid(),"driver_sha256":sha(__file__),"allocation_sha256":sha(allocation),"policy_sha256":sha(checkout/a["policy"]),"command":command,"new_training_records":0,"new_scored_evaluation_games":0,"new_accelerator_jobs":0,"new_control_allocation_seconds":"0","owner_decision_provided":False,"a2_complete":False,"execution_authorized":False}
def save():
 job["measured_driver_wall_seconds"]=time.monotonic()-start
 p=root/"job.tmp";p.write_text(json.dumps(job,indent=2,sort_keys=True)+"\n");p.replace(root/"job.json")
try:
 assert sha(checkout/a["policy"])==a["policy_sha256"] and sha(checkout/a["resource_plan"])==a["resource_plan_sha256"] and sha(checkout/a["proposal"])==a["proposal_sha256"]
 before={str(checkout/a[k]):sha(checkout/a[k]) for k in ("policy","resource_plan","proposal")}
 with (root/"preview.stdout").open("wb") as out,(root/"preview.stderr").open("wb") as err:
  tick=time.monotonic();child=subprocess.Popen(command,cwd=checkout,env=env,stdout=out,stderr=err,start_new_session=True);job["original_preview_pid"]=child.pid;save()
  job["returncode"]=child.wait(timeout=a["outer_watchdog_seconds"]-10);job["measured_process_seconds"]=time.monotonic()-tick
 assert job["returncode"]==0
 review=json.loads((root/"review.json").read_text())
 assert review["status"]=="owner_decision_missing" and review["selection"] is None and not review["selected_scope_costed_and_within_ceiling"] and not review["execution_authorized"] and not review["a2_complete"]
 assert review["scope_ceiling_device_hours"]["full-study"] is None and review["full_study_price_gaps"] and not review["runtime_prices_independently_qualified"]
 assert all(sha(p)==value for p,value in before.items())
 job.update(status="unselected_scope_preview_verified_no_launch",review_sha256=sha(root/"review.json"),original_inputs_unchanged=True,input_files_sha256=before,full_study_component_still_unpriced=True)
except BaseException as e:job.update(status="original_scope_preview_failed_preserved",error=repr(e));traceback.print_exc()
finally:
 if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
 save()

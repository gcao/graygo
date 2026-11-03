from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
repo=Path("/home/gcao/proj/graygo");root=Path(__file__).resolve().parent;checkout=repo/".worktrees/control-campaign-ec2e2b9"
allocation=checkout/"docs/experiments/nfsp-control-campaign-preview-allocation.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
assert sha(allocation)=="c3d6f7117a4505a0f022ec6df401aefa69304bb8365ee9438b2d6d6d8fe36976"
a=load(allocation);control_path=checkout/a["control_allocation"];control=load(control_path);start=time.monotonic();child=None
assert sha(control_path)==a["control_allocation_sha256"] and sha(checkout/a["runner"])==a["runner_sha256"]
for p in (repo/control["campaign_root"],repo/"runs/.control-completion-registry"):assert not p.exists()
assert all(not (repo/entry["root"]).exists() for entry in control["cases"].values())
job={"version":1,"status":"previewing_two_original_source_contexts","original_driver_pid":os.getpid(),"driver_sha256":sha(__file__),"allocation_sha256":sha(allocation),"runner_sha256":sha(checkout/a["runner"]),"commands":[],"source_contexts":{},"new_training_records":0,"new_accelerator_jobs":0,"new_control_reservation_seconds":"0","owner_decision_provided":False,"a2_complete":False,"execution_authorized":False}
def save():
 job["measured_driver_wall_seconds"]=time.monotonic()-start
 p=root/"job.tmp";p.write_text(json.dumps(job,indent=2,sort_keys=True)+"\n");p.replace(root/"job.json")
def alarm(*args):raise TimeoutError("original dual-source control preview watchdog reached")
signal.signal(signal.SIGALRM,alarm);signal.alarm(a["outer_watchdog_seconds"]-10);save()
try:
 for source,package in a["source_contexts"].items():
  output=root/(source+"-preview.json")
  command=[str(repo/".venv/bin/python"),str(checkout/a["runner"]),"preview","--repo",str(repo),"--allocation",str(control_path),"--allocation-sha",a["control_allocation_sha256"],"--output",str(output)]
  row={"source_context":source,"command":command};job["commands"].append(row);save()
  env=dict(os.environ,PYTHONPATH=str(repo/package)+":"+str(repo/a["source_verifier_checkout"])+":"+str(checkout),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
  with (root/(source+".stdout")).open("wb") as out,(root/(source+".stderr")).open("wb") as err:
   tick=time.monotonic();child=subprocess.Popen(command,cwd=checkout,env=env,stdout=out,stderr=err,start_new_session=True);row["original_pid"]=child.pid;save()
   row["returncode"]=child.wait(timeout=a["per_process_ceiling_seconds"]);row["measured_process_seconds"]=time.monotonic()-tick
  save();assert row["returncode"]==0,source
  result=load(output)
  assert result["status"]=="owner_approval_missing" and result["case_count"]==13 and not result["within_selected_ceiling"] and result["selected_control_ceiling_seconds"]=="43200" and not result["owner_approved"] and not result["execution_authorized"]
  assert result["projected_control_commitment_seconds"]=="256436.08177829580345144"
  job["source_contexts"][source]={"status":result["status"],"output_sha256":sha(output),"case_count":13};save()
 for p in (repo/control["campaign_root"],repo/"runs/.control-completion-registry"):assert not p.exists()
 assert all(not (repo/entry["root"]).exists() for entry in control["cases"].values())
 job.update(status="both_source_previews_verified_no_control_or_campaign_created",original_case_roots_absent=True,global_campaign_registry_absent=True)
except BaseException as error:job.update(status="original_dual_source_preview_failed_preserved",error=repr(error));traceback.print_exc()
finally:
 if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
 job["measured_child_process_seconds"]=sum(row.get("measured_process_seconds",0) for row in job["commands"]);save();signal.alarm(0)

from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
repo=Path("/home/gcao/proj/graygo");root=Path(__file__).resolve().parent
allocation=repo/".worktrees/cpu-cache-own-registration-0c49057/docs/experiments/cpu-cache-own-source-registration-allocation.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
assert sha(allocation)=="2d08e65fd5b038109682230869b2de6fc67d640775be94fde9b5a57d1d4d5946"
a=load(allocation);checkout=repo/a["reader_checkout"];start=time.monotonic();child=None
env=dict(os.environ,PYTHONPATH=str((repo/a["candidate_package"]).parent)+":"+str(checkout),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
job={"version":1,"status":"registering_separate_cpu_cache_source_plan","original_driver_pid":os.getpid(),"driver_sha256":sha(__file__),"allocation_sha256":sha(allocation),"reader_sha256":sha(checkout/a["reader"]),"commands":[],"new_training_records":0,"new_accelerator_jobs":0,"new_control_allocation_seconds":"0","new_scored_evaluation_games":0,"a2_complete":False,"execution_authorized":False}
def save():
 job["measured_driver_wall_seconds"]=time.monotonic()-start
 p=root/"job.tmp";p.write_text(json.dumps(job,indent=2,sort_keys=True)+"\n");p.replace(root/"job.json")
def run(label,command):
 global child
 assert time.monotonic()-start<a["outer_watchdog_seconds"]-10
 row={"label":label,"command":command};job["commands"].append(row);save()
 with (root/(label+".stdout")).open("wb") as out,(root/(label+".stderr")).open("wb") as err:
  tick=time.monotonic();child=subprocess.Popen(command,cwd=checkout,env=env,stdout=out,stderr=err,start_new_session=True)
  row["original_pid"]=child.pid;save()
  row["returncode"]=child.wait(timeout=min(a["child_ceiling_seconds"],a["outer_watchdog_seconds"]-10-(time.monotonic()-start)))
  row["measured_process_seconds"]=time.monotonic()-tick;save()
  assert row["returncode"]==0,label
try:
 assert job["reader_sha256"]==a["reader_sha256"] and sha(repo/a["collector"])==a["collector_sha256"]
 assert sha(repo/a["original_source_registry"])==a["original_source_registry_sha256"]
 for section in ("original_case_files_sha256","original_plan_files_sha256","bridge_files_sha256"):
  for p,h in a[section].items():assert sha(repo/p)==h,p
 python=str(repo/".venv/bin/python");plan=root/"candidate-plan.json";registry=root/"registry.json"
 run("plan",[python,str(repo/a["collector"]),"plan","--name","same-protocol-CPU-raw-cache-gray-A2","--learner-source-sha",a["candidate_source_sha256"],"--reuse","4","--output",str(plan)])
 command=[python,"-m","tools.a2_source_cohort","register","--name","original-and-CPU-raw-cache-primary-A2","--canonical-plan",a["canonical_plan"],"--repo",str(repo),"--target-runtime-source",a["target_runtime_source_sha256"],"--output",str(registry)]
 for p in a["prior_plan_paths"]+[str(plan)]:command+=["--plan",p]
 for p in a["bridge_paths"]:command+=["--bridge",p]
 run("register",command)
 common=[python,"-m","tools.a2_source_cohort","assess","--repo",str(repo)]
 for p in a["original_case_files_sha256"]:common+=["--case",str(repo/p)]
 run("old-current",common+["--registry",str(repo/a["original_source_registry"]),"--output",str(root/"old-current.json")])
 run("new-current",common+["--registry",str(registry),"--output",str(root/"new-current.json")])
 run("new-archived",common+["--registry",str(registry),"--archived","--output",str(root/"new-archived.json")])
 left,right=load(root/"old-current.json"),load(root/"new-current.json");left.pop("registry_sha256");right.pop("registry_sha256");assert left==right
 left,right=load(root/"new-current.json"),load(root/"new-archived.json")
 for k in ("analysis_mode","current_input_snapshots_verified","current_source_admission_inputs_verified"):left.pop(k);right.pop(k)
 assert left==right and left["present_cases"]==14 and len(left["missing_cases"])==22 and left["controls"]["case_misses"]==["delayed/tabular/0"]
 assert all(sha(repo/p)==h for p,h in a["original_case_files_sha256"].items())
 job.update(status="registered_cpu_cache_own_source_with_unchanged_prior_cases",registry_sha256=sha(registry),candidate_plan_sha256=sha(plan),baseline_present_cases=14,baseline_missing_cases=22,original_case_files_unchanged=True,old_new_numerics_identical=True,current_archived_numerics_identical=True)
except BaseException as e:
 job.update(status="registration_failed_preserved",error=repr(e));traceback.print_exc()
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=4)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 save()

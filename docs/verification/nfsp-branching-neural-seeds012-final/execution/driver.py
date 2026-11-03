from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
repo=Path("/home/gcao/proj/graygo");root=Path(__file__).resolve().parent
allocation=repo/".worktrees/nfsp-branching-allocation-cd16f60/docs/experiments/nfsp-branching-neural-seeds012-allocation.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
assert sha(allocation)=="ef8c7371709b642a4b85fede24331e61ca34db92994a3415d128f6ad03fe7caf"
a=load(allocation);checkout=repo/a["runner_checkout"];start=time.monotonic();children={};files=[]
env=dict(os.environ,PYTHONPATH=str(repo/a["source_package"])+":"+str(repo/a["source_verifier_checkout"]),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
job={"version":1,"status":"running_three_original_branching_neural_seeds","original_driver_pid":os.getpid(),"driver_sha256":sha(__file__),"allocation_sha256":sha(allocation),"runner_sha256":sha(checkout/a["runner"]),"source_sha256":a["learner_source_sha256"],"cases":{},"aggregate_neural_cpu_reservation_seconds":a["aggregate_neural_cpu_reservation_seconds"],"additional_control_reservation_seconds":"0","new_accelerator_jobs":0,"no_implicit_resume_or_observation_timeout_restart":True,"project_scope_selected":False,"pilot_selected":False,"a2_complete":False}
def save():
 job["driver_wall_seconds"]=time.monotonic()-start
 p=root/"job.tmp";p.write_text(json.dumps(job,indent=2,sort_keys=True)+"\n");p.replace(root/"job.json")
def alarm(*args):raise TimeoutError("original branching parent watchdog reached")
signal.signal(signal.SIGALRM,alarm);signal.alarm(a["parent_outer_watchdog_seconds"])
def stopped(signum,frame):raise InterruptedError("original branching parent interrupted")
signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
try:
 assert sha(checkout/a["runner"])==a["runner_sha256"]
 for key in ("cpu_price_receipt","source_registry","own_source_plan"):assert sha(repo/a[key])==a[key+"_sha256"]
 assert all(not (repo/p).exists() for p in a["case_roots"].values())
 for seed in a["seeds"]:
  command=[str(repo/".venv/bin/python"),str(checkout/a["runner"]),"run","--repo",str(repo),"--allocation",str(allocation),"--allocation-sha",sha(allocation),"--seed",str(seed)]
  out=(root/("seed"+str(seed)+".stdout")).open("xb");err=(root/("seed"+str(seed)+".stderr")).open("xb");files.extend([out,err])
  tick=time.monotonic();child=subprocess.Popen(command,cwd=checkout,env=env,stdout=out,stderr=err,start_new_session=True);children[seed]=child
  stat=Path("/proc/"+str(child.pid)+"/stat").read_text().split(") ",1)[1].split()
  job["cases"][str(seed)]={"original_pid":child.pid,"original_start_ticks":stat[19],"original_cmdline_sha256":sha(Path("/proc/"+str(child.pid)+"/cmdline")),"command":command,"started_monotonic":tick,"whole_process_cap_seconds":a["per_case_process_ceiling_seconds"]}
  save()
 last_save=0
 while any(child.poll() is None for child in children.values()):
  now=time.monotonic()
  for seed,child in children.items():
   row=job["cases"][str(seed)];elapsed=now-row["started_monotonic"]
   if child.poll() is None:
    if elapsed>=a["per_case_process_ceiling_seconds"]-a["graceful_budget_stop_headroom_seconds"] and not row.get("graceful_budget_signal"):
     os.killpg(child.pid,signal.SIGUSR1);row["graceful_budget_signal"]=True;row["graceful_signal_elapsed_seconds"]=elapsed
    if elapsed>=a["per_case_process_ceiling_seconds"]-1:
     os.killpg(child.pid,signal.SIGKILL);child.wait();row["whole_process_budget_hard_stop"]=True
   if child.poll() is not None and "returncode" not in row:
    row.update(returncode=child.returncode,measured_process_upper_seconds=time.monotonic()-row["started_monotonic"])
   case_root=repo/a["case_roots"][str(seed)]
   if (case_root/"case-state.json").exists():row["state"]=load(case_root/"case-state.json")
   if (case_root/"result.json").exists():
    result=load(case_root/"result.json");row["latest_progress"]={k:result[k] for k in ("status","records","elapsed_seconds","counters","q_credit_passed","average_readiness") if k in result}
   if not row.get("actual_zero_record_initialization_retained") and (case_root/"initialization.json").exists():
    initial=load(case_root/"initialization.json");assert initial["step"]==0 and initial["source_sha256"]==a["learner_source_sha256"] and all(v==0 for v in initial["counters"].values())
    assert sha(case_root/"initial-checkpoint/manifest.json")==initial["retained_initial_checkpoint_manifest_sha256"]
    manifest=load(case_root/"run.json");assert manifest["config"]==a["case_configs"][str(seed)] and manifest["source_sha256"]==a["learner_source_sha256"]
    row.update(actual_zero_record_initialization_retained=True,initialization_sha256=sha(case_root/"initialization.json"),emitted_manifest_verified=True)
  if now-last_save>=10:save();last_save=now
  time.sleep(.5)
 for seed,child in children.items():
  row=job["cases"][str(seed)]
  if "returncode" not in row:row.update(returncode=child.returncode,measured_process_upper_seconds=time.monotonic()-row["started_monotonic"])
 job["status"]="three_original_branching_workers_terminal"
 job["all_original_workers_complete"]=all(row.get("returncode")==0 and row.get("latest_progress",{}).get("status")=="complete" and row["latest_progress"]["records"]==15000000 for row in job["cases"].values())
except BaseException as e:
 job.update(status="original_branching_execution_stopped_preserved",error=repr(e));traceback.print_exc()
finally:
 for child in children.values():
  if child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 for f in files:f.close()
 save();signal.alarm(0)

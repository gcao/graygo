from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
repo=Path("/home/gcao/proj/graygo");root=Path(__file__).resolve().parent
allocation=repo/".worktrees/nfsp-gray-allocation-0247bc7/docs/experiments/nfsp-gray-neural-seeds012-allocation.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
assert sha(allocation)=="6368c33ff903e3d9cd8526f13c0bd2b5cd90726f02cc0fe4e916f2accf8c728c"
a=load(allocation);checkout=repo/a["runner_checkout"];start=time.monotonic();children={};files=[]
env=dict(os.environ,PYTHONPATH=str(repo/a["source_package"])+":"+str(repo/a["source_verifier_checkout"]),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
job={"version":1,"status":"running_six_original_gray_neural_cases","original_driver_pid":os.getpid(),"driver_sha256":sha(__file__),"allocation_sha256":sha(allocation),"runner_sha256":sha(checkout/a["runner"]),"source_sha256":a["learner_source_sha256"],"cases":{},"aggregate_neural_cpu_reservation_seconds":a["aggregate_neural_cpu_reservation_seconds"],"additional_control_reservation_seconds":"0","new_accelerator_jobs":0,"no_implicit_resume_or_observation_timeout_restart":True,"project_scope_selected":False,"pilot_selected":False,"a2_complete":False}
def save():
 job["driver_wall_seconds"]=time.monotonic()-start
 p=root/"job.tmp";p.write_text(json.dumps(job,indent=2,sort_keys=True)+"\n");p.replace(root/"job.json")
def refresh(key):
 row=job["cases"][key];folder=repo/a["case_roots"][key]
 if (folder/"case-state.json").exists():row["state"]=load(folder/"case-state.json")
 if (folder/"result.json").exists():
  result=load(folder/"result.json");row["latest_progress"]={k:result[k] for k in ("status","records","elapsed_seconds","counters","q_credit_passed","average_readiness") if k in result}
 if not row.get("actual_zero_record_initialization_retained") and (folder/"initialization.json").exists():
  initial=load(folder/"initialization.json")
  assert initial["step"]==0 and initial["source_sha256"]==a["learner_source_sha256"] and all(v==0 for v in initial["counters"].values())
  assert sha(folder/"initial-checkpoint/manifest.json")==initial["retained_initial_checkpoint_manifest_sha256"]
  manifest=load(folder/"run.json");assert manifest["config"]==a["case_configs"][key] and manifest["source_sha256"]==a["learner_source_sha256"]
  row.update(actual_zero_record_initialization_retained=True,initialization_sha256=sha(folder/"initialization.json"),emitted_manifest_verified=True)
def alarm(*args):raise TimeoutError("original Gray Go parent watchdog reached")
def stopped(signum,frame):raise InterruptedError("original Gray Go parent interrupted")
signal.signal(signal.SIGALRM,alarm);signal.alarm(a["parent_outer_watchdog_seconds"])
signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
try:
 assert sha(checkout/a["runner"])==a["runner_sha256"]
 for key in ("cpu_price_receipt","source_registry","own_source_plan","source_admission"):assert sha(repo/a[key])==a[key+"_sha256"]
 assert all(not (repo/p).exists() for p in a["case_roots"].values())
 for task in a["tasks"]:
  for seed in a["seeds"]:
   key=task+"/"+str(seed)
   command=[str(repo/".venv/bin/python"),str(checkout/a["runner"]),"run","--repo",str(repo),"--allocation",str(allocation),"--allocation-sha",sha(allocation),"--task",task,"--seed",str(seed)]
   label=key.replace("/","-");out=(root/(label+".stdout")).open("xb");err=(root/(label+".stderr")).open("xb");files.extend([out,err])
   tick=time.monotonic();child=subprocess.Popen(command,cwd=checkout,env=env,stdout=out,stderr=err,start_new_session=True);children[key]=child
   stat=Path("/proc/"+str(child.pid)+"/stat").read_text().rsplit(")",1)[1].split()
   job["cases"][key]={"task":task,"seed":seed,"original_pid":child.pid,"original_start_ticks":stat[19],"original_cmdline_sha256":sha(Path("/proc/"+str(child.pid)+"/cmdline")),"command":command,"started_monotonic":tick,"whole_process_cap_seconds":a["per_case_process_ceiling_seconds"][task]}
   save()
 last_save=0
 while any(child.poll() is None for child in children.values()):
  now=time.monotonic()
  for key,child in children.items():
   row=job["cases"][key];elapsed=now-row["started_monotonic"];cap=row["whole_process_cap_seconds"]
   if child.poll() is None:
    if elapsed>=cap-a["graceful_budget_stop_headroom_seconds"] and not row.get("graceful_budget_signal"):
     os.killpg(child.pid,signal.SIGUSR1);row["graceful_budget_signal"]=True;row["graceful_signal_elapsed_seconds"]=elapsed
    if elapsed>=cap-1:os.killpg(child.pid,signal.SIGKILL);child.wait();row["whole_process_budget_hard_stop"]=True
   if child.poll() is not None and "returncode" not in row:row.update(returncode=child.returncode,measured_process_upper_seconds=time.monotonic()-row["started_monotonic"])
   refresh(key)
  if now-last_save>=10:save();last_save=now
  time.sleep(.5)
 for key,child in children.items():
  row=job["cases"][key]
  if "returncode" not in row:row.update(returncode=child.returncode,measured_process_upper_seconds=time.monotonic()-row["started_monotonic"])
  refresh(key)
 job["status"]="six_original_gray_workers_terminal"
 job["all_original_workers_complete"]=all(row.get("returncode")==0 and row.get("latest_progress",{}).get("status")=="complete" and row["latest_progress"]["records"]==15000000 for row in job["cases"].values())
except BaseException as e:job.update(status="original_gray_execution_stopped_preserved",error=repr(e));traceback.print_exc()
finally:
 for child in children.values():
  if child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 for f in files:f.close()
 save();signal.alarm(0)

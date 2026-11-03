from pathlib import Path
import fcntl,hashlib,json,os,signal,subprocess,time,traceback
repo=Path("/home/gcao/proj/graygo");root=Path(__file__).resolve().parent;checkout=repo/".worktrees/delayed-r1-audit-a9408ed"
allocation=checkout/"docs/experiments/nfsp-delayed-r1-neural-seeds12-audit-allocation.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
assert sha(allocation)=="3becbf6f53cc561d866b6cc178ca46f3c5a122dfbd25139c6d0ba78bd63cef36"
a=load(allocation);start=time.monotonic();child=None;wait_seconds=0.;collector=repo/a["collector"]
training_allocation=repo/a["training_allocation"];plan=repo/a["cohort_plan"]
assert sha(training_allocation)==a["training_allocation_sha256"]
assert sha(collector)==a["collector_sha256"] and sha(collector.with_name("audit_nfsp_delayed.py"))==a["delayed_auditor_sha256"]
assert sha(plan)==a["cohort_plan_sha256"] and sha(repo/a["budget_profile"])==a["budget_profile_sha256"]
assert sha(checkout/a["comparison_reader"])==a["comparison_reader_sha256"] and sha(repo/a["comparison_admission"])==a["comparison_admission_sha256"]
assert all(sha(repo/p)==h for p,h in a["initialization_files_sha256"].items())
from graygo.manifest import source_digest
assert source_digest()==a["learner_source_sha256"]
job={"version":1,"status":"assessing_fixed_original_reuse_references","original_driver_pid":os.getpid(),"driver_sha256":sha(__file__),"allocation_sha256":sha(allocation),"comparison_reader_sha256":a["comparison_reader_sha256"],"collector_sha256":a["collector_sha256"],"delayed_auditor_sha256":a["delayed_auditor_sha256"],"commands":[],"cases":{},"terminal_missing_cases":[],"new_training_records":0,"new_scored_evaluation_games":0,"new_accelerator_jobs":0,"no_training_launch_or_resume":True,"primary_A2_slots_filled":0,"a2_complete":False,"execution_authorized":False}
env=dict(os.environ,PYTHONPATH=str(repo/a["source_package"])+":"+str(repo/a["source_verifier_checkout"])+":"+str(repo/a["admission_checkout"])+":"+str(checkout),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
new_cases={}
def save():
 job.update(wait_seconds=wait_seconds,driver_wall_seconds=time.monotonic()-start)
 p=root/"job.tmp";p.write_text(json.dumps(job,indent=2,sort_keys=True)+"\n");p.replace(root/"job.json")
def alive(seed):
 expected=a["original_case_identities"][seed];pid=expected["pid"]
 try:
  stat=Path("/proc/"+str(pid)+"/stat").read_text().rsplit(")",1)[1].split();cmd=Path("/proc/"+str(pid)+"/cmdline").read_bytes()
 except FileNotFoundError:return False
 if stat[0] in ("Z","X"):return False
 assert stat[19]==expected["start_ticks"] and hashlib.sha256(cmd).hexdigest()==expected["cmdline_sha256"],"original comparison worker identity changed"
 return True
def run(label,command,timeout):
 global child
 row={"label":label,"command":command};job["commands"].append(row);save()
 with (root/(label+".stdout")).open("wb") as out,(root/(label+".stderr")).open("wb") as err:
  tick=time.monotonic();child=subprocess.Popen(command,cwd=checkout,env=env,stdout=out,stderr=err,start_new_session=True);row["original_pid"]=child.pid;save()
  row["returncode"]=child.wait(timeout=timeout);row["measured_process_seconds"]=time.monotonic()-tick
 save();assert row["returncode"]==0,label
def report():
 index=len(new_cases);common=[str(repo/".venv/bin/python"),str(checkout/a["comparison_reader"]),"--repo",str(repo),"--allocation",str(training_allocation)]
 for seed,path in new_cases.items():common+=["--case",seed+"="+str(path)]
 current=root/("comparison-"+str(index).zfill(2)+"-current.json");archived=root/("comparison-"+str(index).zfill(2)+"-archived.json")
 run("comparison-"+str(index).zfill(2)+"-current",common+["--output",str(current)],a["assessment_outer_seconds"])
 run("comparison-"+str(index).zfill(2)+"-archived",common+["--archived","--output",str(archived)],a["assessment_outer_seconds"])
 left,right=load(current),load(archived)
 for key in ("analysis_mode","current_input_snapshots_verified","current_source_prices_verified"):left.pop(key);right.pop(key)
 assert left==right and left["primary_A2_slots_filled"]==0 and not left["reuse_selected"] and not left["Retrace_selected"] and not left["a2_complete"]
 assert left["cases"]["R1/0"]["status"]=="numeric_miss" and left["cases"]["R4/2"]["status"]=="numeric_miss"
 job["latest_comparison"]={"current":str(current),"current_sha256":sha(current),"archived":str(archived),"archived_sha256":sha(archived),"current_archived_numerics_match":True,"groups":left["groups"],"primary_A2_slots_filled":0};save()
def alarm(*args):raise TimeoutError("original R1 comparison observation cap reached")
signal.signal(signal.SIGALRM,alarm);signal.alarm(a["outer_watchdog_seconds"]-30);save()
try:
 report()
 while len(job["cases"])+len(job["terminal_missing_cases"])<2:
  states={seed:alive(seed) for seed in a["original_case_identities"]};job["original_worker_handles_live"]=states
  for seed,relative in a["new_case_roots"].items():
   if seed in job["cases"] or seed in job["terminal_missing_cases"]:continue
   directory=repo/relative
   if states[seed]:continue
   if not (directory/"result.json").exists():job["terminal_missing_cases"].append(seed);continue
   manifest=load(directory/"run.json");result=load(directory/"result.json")
   assert manifest["source_sha256"]==a["learner_source_sha256"] and manifest["config"]==a["new_case_configs"][seed]
   with (directory/".trainer.lock").open("rb") as lease:fcntl.flock(lease,fcntl.LOCK_SH|fcntl.LOCK_NB)
   assert all(sha(repo/p)==h for p,h in a["initialization_files_sha256"].items()) and sha(repo/a["budget_profile"])==a["budget_profile_sha256"]
   if result["status"]=="complete":assert result["records"]==15000000
   output=root/("case-delayed-R1-neural-"+seed+".json")
   command=[str(repo/".venv/bin/python"),str(collector),"audit-case","--plan",str(plan),"--run-dir",str(directory),"--budget-profile",str(repo/a["budget_profile"]),"--time-cap-seconds",str(a["case_time_ceiling_seconds"]),"--allow-diagnostic","--output",str(output)]
   run("audit-original-R1-seed"+seed,command,a["case_outer_seconds"])
   case=load(output)["body"]
   assert case["case"]==["delayed","neural",int(seed)] and case["root"]==str(directory.resolve()) and case["primary_recipe"] is False
   new_cases[seed]=output;job["cases"][seed]={"case_sha256":sha(output),"status":case["status"],"records":case["records"],"fresh_audit_verified":case["fresh_audit_verified"],"primary_recipe":False};report()
  if len(job["cases"])+len(job["terminal_missing_cases"])==2:break
  assert wait_seconds<a["maximum_wait_seconds"],"original comparison wait cap reached; no replacement selected"
  job["status"]="waiting_for_two_fixed_original_R1_comparisons";save();tick=time.monotonic();time.sleep(5);wait_seconds+=time.monotonic()-tick
 assert all(sha(repo/p)==h for p,h in a["initialization_files_sha256"].items())
 job.update(status="fixed_original_R1_comparison_evidence_collected",original_initialization_files_unchanged=True)
except BaseException as e:job.update(status="original_read_only_R1_comparison_audit_failed_preserved",error=repr(e));traceback.print_exc()
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 job["measured_child_process_seconds"]=sum(row.get("measured_process_seconds",0) for row in job["commands"]);save();signal.alarm(0)

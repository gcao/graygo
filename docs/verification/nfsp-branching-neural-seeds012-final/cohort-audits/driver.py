from pathlib import Path
import fcntl,hashlib,json,os,signal,subprocess,time,traceback
repo=Path("/home/gcao/proj/graygo");root=Path(__file__).resolve().parent
allocation=repo/".worktrees/nfsp-branching-audit-6e31009/docs/experiments/nfsp-branching-neural-seeds012-audit-allocation.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
assert sha(allocation)=="cd462a1d22025a233fb334a5e208118a00b5a9b8afdbe40e390eaa9f073291f6"
a=load(allocation);start=time.monotonic();child=None;wait_seconds=0.;collector=repo/a["collector_path"];frozen=repo/a["source_verifier_checkout"];plan=repo/a["cohort_plan"]
assert sha(collector)==a["collector_sha256"] and sha(collector.with_name("audit_nfsp_bounded.py"))==a["bounded_auditor_sha256"]
assert sha(plan)==a["cohort_plan_sha256"] and sha(repo/a["source_registry"])==a["source_registry_sha256"] and sha(frozen/a["reader"])==a["reader_sha256"]
assert sha(repo/a["budget_profile"])==a["budget_profile_sha256"]
assert all(sha(repo/p)==h for p,h in a["initialization_files_sha256"].items()) and all(sha(repo/p)==h for p,h in a["original_case_files_sha256"].items())
from graygo.manifest import source_digest
assert source_digest()==a["learner_source_sha256"]
job={"version":1,"status":"waiting_for_fixed_original_branching_cases","original_driver_pid":os.getpid(),"driver_sha256":sha(__file__),"allocation_sha256":sha(allocation),"collector_sha256":a["collector_sha256"],"bounded_auditor_sha256":a["bounded_auditor_sha256"],"reader_sha256":a["reader_sha256"],"commands":[],"cases":{},"terminal_missing_cases":[],"new_training_records":0,"new_scored_evaluation_games":0,"new_accelerator_jobs":0,"no_training_launch_or_resume":True,"a2_complete":False,"execution_authorized":False}
old_cases=[repo/p for p in a["original_case_files_sha256"]];new_cases=[]
env=dict(os.environ,PYTHONPATH=str(repo/a["source_package_path"])+":"+str(frozen),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
def save():
 job.update(wait_seconds=wait_seconds,driver_wall_seconds=time.monotonic()-start)
 p=root/"job.tmp";p.write_text(json.dumps(job,indent=2,sort_keys=True)+"\n");p.replace(root/"job.json")
def alive(seed):
 expected=a["original_case_identities"][seed];pid=expected["pid"]
 try:
  stat=Path("/proc/"+str(pid)+"/stat").read_text().rsplit(")",1)[1].split()
  cmd=Path("/proc/"+str(pid)+"/cmdline").read_bytes()
 except FileNotFoundError:return False
 if stat[0] in ("Z","X"):return False
 assert stat[19]==expected["start_ticks"] and hashlib.sha256(cmd).hexdigest()==expected["cmdline_sha256"],"original worker identity changed"
 return True
def run(label,command,timeout):
 global child
 row={"label":label,"command":command};job["commands"].append(row);save()
 with (root/(label+".stdout")).open("wb") as out,(root/(label+".stderr")).open("wb") as err:
  tick=time.monotonic();child=subprocess.Popen(command,cwd=frozen,env=env,stdout=out,stderr=err,start_new_session=True);row["original_pid"]=child.pid;save()
  row["returncode"]=child.wait(timeout=timeout);row["measured_process_seconds"]=time.monotonic()-tick
 save();assert row["returncode"]==0,label
def auxiliary():
 assert sha(repo/a["auxiliary_original_allocation"])==a["auxiliary_original_allocation_sha256"]
 result=[]
 for item in a["auxiliary_fixed_case_outputs"]:
  path=repo/item["path"];job_path=path.parent/"job.json"
  if not path.exists() or not job_path.exists():continue
  other=load(job_path)
  assert other["allocation_sha256"]==a["auxiliary_original_allocation_sha256"]
  label="/".join(str(v) for v in item["case"])
  if label not in other["cases"]:continue
  assert other["cases"][label]["case_sha256"]==sha(path)
  wrapper=load(path);assert wrapper["body"]["case"]==item["case"] and wrapper["body"]["root"]==item["root"]
  result.append(path)
 return result
def report():
 index=len(new_cases);extra=auxiliary();common=[str(repo/".venv/bin/python"),"-m","tools.a2_source_cohort","assess","--registry",str(repo/a["source_registry"]),"--repo",str(repo)]
 for p in old_cases+new_cases+extra:common+=["--case",str(p)]
 current=root/("assessment-"+str(index).zfill(2)+"-current.json");archived=root/("assessment-"+str(index).zfill(2)+"-archived.json")
 run("assess-"+str(index).zfill(2)+"-current",common+["--output",str(current)],a["assessment_outer_seconds"])
 run("assess-"+str(index).zfill(2)+"-archived",common+["--archived","--output",str(archived)],a["assessment_outer_seconds"])
 left,right=load(current),load(archived)
 for key in ("analysis_mode","current_input_snapshots_verified","current_source_admission_inputs_verified"):left.pop(key);right.pop(key)
 assert left==right and not left["a2_complete"]
 job["latest_assessment"]={"current":str(current),"current_sha256":sha(current),"archived":str(archived),"archived_sha256":sha(archived),"present_cases":left["present_cases"],"missing_cases":len(left["missing_cases"]),"current_archived_numerics_match":True,"included_fixed_delayed_outputs":[str(p) for p in extra]};save()
def alarm(*args):raise TimeoutError("original branching read-only observation cap reached")
signal.signal(signal.SIGALRM,alarm);signal.alarm(a["outer_watchdog_seconds"]-30);save()
try:
 while len(job["cases"])+len(job["terminal_missing_cases"])<3:
  states={seed:alive(seed) for seed in a["original_case_identities"]};job["original_worker_handles_live"]=states
  for label,relative in a["new_case_roots"].items():
   if label in job["cases"] or label in job["terminal_missing_cases"]:continue
   seed=label.rsplit("/",1)[1];directory=repo/relative
   if states[seed]:continue
   if not (directory/"result.json").exists():job["terminal_missing_cases"].append(label);continue
   manifest=load(directory/"run.json");result=load(directory/"result.json")
   assert manifest["source_sha256"]==a["learner_source_sha256"] and manifest["config"]==a["new_case_configs"][label]
   with (directory/".trainer.lock").open("rb") as lease:fcntl.flock(lease,fcntl.LOCK_SH|fcntl.LOCK_NB)
   assert all(sha(repo/p)==h for p,h in a["initialization_files_sha256"].items()) and sha(repo/a["budget_profile"])==a["budget_profile_sha256"]
   if result["status"]=="complete":assert result["records"]==15000000
   output=root/("case-"+label.replace("/","-")+".json")
   command=[str(repo/".venv/bin/python"),str(collector),"audit-case","--plan",str(plan),"--run-dir",str(directory),"--budget-profile",str(repo/a["budget_profile"]),"--time-cap-seconds",str(a["case_time_ceiling_seconds"]),"--output",str(output)]
   run("audit-"+label.replace("/","-"),command,a["case_outer_seconds"])
   case=load(output)["body"];assert case["case"]==["branching","neural",int(seed)] and case["root"]==str(directory.resolve())
   new_cases.append(output);job["cases"][label]={"case_sha256":sha(output),"status":case["status"],"records":case["records"],"fresh_audit_verified":case["fresh_audit_verified"]};report()
  if len(job["cases"])+len(job["terminal_missing_cases"])==3:break
  assert wait_seconds<a["maximum_wait_seconds"],"original wait cap reached; no replacement selected"
  job["status"]="waiting_for_fixed_original_branching_cases";save();tick=time.monotonic();time.sleep(5);wait_seconds+=time.monotonic()-tick
 assert all(sha(repo/p)==h for p,h in a["original_case_files_sha256"].items()) and all(sha(repo/p)==h for p,h in a["initialization_files_sha256"].items())
 job.update(status="fixed_original_branching_case_evidence_collected",original_case_files_unchanged=True,original_initialization_files_unchanged=True)
except BaseException as e:job.update(status="original_read_only_audit_failed_preserved",error=repr(e));traceback.print_exc()
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 job["measured_child_process_seconds"]=sum(row.get("measured_process_seconds",0) for row in job["commands"]);save();signal.alarm(0)

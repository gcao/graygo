from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
repo=Path("/home/gcao/proj/graygo")
root=repo/"runs/cpu-raw-cache-admission-1b405bd"
checkout=repo/".worktrees/cpu-cache-admission-1b405bd"
allocation=repo/"docs/experiments/cpu-raw-cache-source-admission-allocation.json"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads(allocation.read_text())
start=time.monotonic()
env=dict(os.environ,PYTHONPATH=str(repo/".worktrees/cpu-raw-cache-comparison-257276e/src")+":"+str(checkout),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
command=[str(repo/".venv/bin/python"),str(checkout/plan["verifier"]),"--repo",str(repo),"--reference",str(repo/plan["reference_package"]),"--candidate",str(repo/plan["candidate_package"]),"--comparison",str(repo/plan["comparison_root"]),"--audits",str(repo/plan["audit_root"]),"--suite",str(repo/plan["suite_root"]),"--expected-reference",plan["reference_source_sha256"],"--expected-candidate",plan["candidate_source_sha256"],"--output",str(root/"admission.json")]
job={"version":1,"status":"running_read_only_source_qualification","original_driver_pid":os.getpid(),"driver_sha256":sha(Path(__file__)),"allocation_sha256":sha(allocation),"command":command,"verifier_sha256":sha(checkout/plan["verifier"]),"source_reader_sha256":sha(checkout/plan["source_reader"]),"new_training_records":0,"new_accelerator_jobs":0,"new_control_allocation_seconds":"0","new_scored_evaluation_games":0,"execution_authorized":False,"a2_complete":False}
def save():
 job["measured_driver_wall_seconds"]=time.monotonic()-start
 tmp=root/"job.tmp";tmp.write_text(json.dumps(job,indent=2,sort_keys=True)+"\n");tmp.replace(root/"job.json")
try:
 with (root/"qualification.stdout").open("wb") as out,(root/"qualification.stderr").open("wb") as err:
  p=subprocess.Popen(command,cwd=checkout,env=env,stdout=out,stderr=err,start_new_session=True)
  job["original_qualification_pid"]=p.pid
  job["original_qualification_start_ticks"]=Path("/proc/"+str(p.pid)+"/stat").read_text().split(") ",1)[1].split()[19]
  job["original_qualification_cmdline_sha256"]=sha(Path("/proc/"+str(p.pid)+"/cmdline"))
  save()
  try:rc=p.wait(timeout=plan["outer_watchdog_seconds"]-10)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGKILL);p.wait();raise RuntimeError("original read-only qualification reached its declared time ceiling")
 job["returncode"]=rc
 if rc==0:
  receipt=json.loads((root/"admission.json").read_text())
  assert receipt["status"]=="qualified_cpu_3x3_raw_cache"
  job["status"]="qualified_cpu_3x3_raw_cache"
  job["admission_sha256"]=sha(root/"admission.json")
 else:job["status"]="source_qualification_failed_preserved"
except BaseException as e:
 job["status"]="source_qualification_failed_preserved";job["error"]=repr(e);traceback.print_exc()
finally:
 for name in ("qualification.stdout","qualification.stderr"):
  if (root/name).exists():job[name+"_sha256"]=sha(root/name)
 save()

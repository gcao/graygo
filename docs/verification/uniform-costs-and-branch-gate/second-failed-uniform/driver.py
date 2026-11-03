from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
repo=Path("/home/gcao/proj/graygo");root=Path(__file__).resolve().parent;checkout=repo/".worktrees/uniform-cost-210a960"
allocation=checkout/"docs/experiments/nfsp-uniform-board-cost-corrected-allocation.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
assert sha(allocation)=="99b6d8caf3ad4302a2182e591e7f5ec51ee8dda3b1a230ed90ca81104ec16d05"
a=load(allocation);started=time.monotonic();child=None
before=load(repo/a["base_audit"])["input_files_sha256"]
assert all(sha(p)==value for p,value in before.items())
assert sha(checkout/a["producer"])==a["producer_sha256"] and sha(checkout/a["pipeline_producer"])==a["pipeline_producer_sha256"]
gpu=subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
assert gpu==a["physical_device_uuid"]
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid","--format=csv,noheader"],text=True).strip()
env=dict(os.environ,PYTHONPATH=str(repo/a["source_package"])+":"+str(repo/a["base_tooling_root"])+":"+str(checkout),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
command=[str(repo/".venv/bin/python"),str(checkout/a["producer"]),"--repo",str(repo),"--allocation",str(allocation),"--expected-allocation-sha",sha(allocation),"--run-dir",str(root/"profile")]
job={"version":1,"status":"running_original_uniform_reference_hardware_profile","original_driver_pid":os.getpid(),"driver_sha256":sha(__file__),"allocation_sha256":sha(allocation),"producer_sha256":a["producer_sha256"],"pipeline_producer_sha256":a["pipeline_producer_sha256"],"physical_device_uuid":gpu,"command":command,"new_formal_A2_records":0,"new_control_allocation_seconds":"0","reference_training_selected":False,"pilot_selected":False,"study_selected":False,"a2_complete":False,"execution_authorized":False}
def save():
 job["measured_driver_wall_seconds"]=time.monotonic()-started
 p=root/"job.tmp";p.write_text(json.dumps(job,indent=2,sort_keys=True)+"\n");p.replace(root/"job.json")
def alarm(*args):raise TimeoutError("original uniform hardware whole-process watchdog exhausted")
signal.signal(signal.SIGALRM,alarm);signal.alarm(a["outer_watchdog_seconds"]-10);save()
try:
 with (root/"profile.stdout").open("wb") as out,(root/"profile.stderr").open("wb") as err:
  tick=time.monotonic();epoch=time.time();job["allocation_interval_start_epoch"]=str(epoch)
  child=subprocess.Popen(command,cwd=checkout,env=env,stdout=out,stderr=err,start_new_session=True)
  job["original_profile_pid"]=child.pid
  job["original_profile_start_ticks"]=Path("/proc/"+str(child.pid)+"/stat").read_text().rsplit(")",1)[1].split()[19]
  save()
  job["returncode"]=child.wait(timeout=a["outer_watchdog_seconds"]-20)
  job["allocation_interval_end_epoch"]=str(time.time());job["measured_process_seconds"]=time.monotonic()-tick
 assert job["returncode"]==0
 report=load(root/"profile/report.json")
 assert report["status"]=="complete_measurement_pending_independent_audit" and report["new_diagnostic_learner_records"]==a["new_diagnostic_record_cap"]
 assert all(sha(p)==value for p,value in before.items())
 job.update(status="complete_uniform_measurement_pending_audit",report_sha256=sha(root/"profile/report.json"),original_inputs_unchanged=True,
   new_diagnostic_records=report["new_diagnostic_learner_records"],B_U_board_device_hours=report["forecast"]["B_U_board_device_hours"])
except BaseException as error:
 job.update(status="original_uniform_measurement_failed_preserved",error=repr(error));traceback.print_exc()
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 if "allocation_interval_start_epoch" in job and "allocation_interval_end_epoch" not in job:
  job["allocation_interval_end_epoch"]=str(time.time());job["measured_process_seconds"]=time.monotonic()-tick
 save();signal.alarm(0)

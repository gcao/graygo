from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
repo=Path("/home/gcao/proj/graygo");root=Path(__file__).resolve().parent;checkout=repo/".worktrees/branch-response-5d82059"
allocation=checkout/"docs/experiments/branch-native-response-diagnostic-allocation.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
assert sha(allocation)=="d385015496df407fd3315d25397910b07d683546e771b2f53d0712d96a212431"
a=load(allocation);start=time.monotonic();child=None
assert sha(checkout/a["producer"])==a["producer_sha256"] and sha(checkout/a["specification"])==a["specification_sha256"]
env=dict(os.environ,PYTHONPATH=str(repo/a["source_package"])+":"+str(repo/a["source_verifier_checkout"])+":"+str(checkout),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
command=[str(repo/".venv/bin/python"),str(checkout/a["producer"]),"--repo",str(repo),"--specification",str(checkout/a["specification"]),"--expected-specification-sha",a["specification_sha256"],"--output",str(root/"diagnostic.json")]
job={"version":1,"status":"diagnosing_actual_original_branch_archives","original_driver_pid":os.getpid(),"driver_sha256":sha(__file__),"allocation_sha256":sha(allocation),"producer_sha256":a["producer_sha256"],"command":command,"new_training_records":0,"new_accelerator_jobs":0,"new_scored_evaluation_games":0,"new_control_allocation_seconds":"0","a2_complete":False,"execution_authorized":False}
def save():
 job["measured_driver_wall_seconds"]=time.monotonic()-start
 p=root/"job.tmp";p.write_text(json.dumps(job,indent=2,sort_keys=True)+"\n");p.replace(root/"job.json")
try:
 with (root/"diagnostic.stdout").open("wb") as out,(root/"diagnostic.stderr").open("wb") as err:
  tick=time.monotonic();child=subprocess.Popen(command,cwd=checkout,env=env,stdout=out,stderr=err,start_new_session=True);job["original_diagnostic_pid"]=child.pid;save()
  job["returncode"]=child.wait(timeout=a["outer_watchdog_seconds"]-10);job["measured_process_seconds"]=time.monotonic()-tick
 assert job["returncode"]==0
 report=load(root/"diagnostic.json")
 assert report["status"]=="all_original_branch_archives_native_response_diagnosed" and not report["original_data_grades_or_thresholds_modified"]
 job.update(status=report["status"],diagnostic_sha256=sha(root/"diagnostic.json"),original_grades_unchanged=True)
except BaseException as error:job.update(status="original_native_response_diagnostic_failed_preserved",error=repr(error));traceback.print_exc()
finally:
 if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
 save()

from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
repo=Path("/home/gcao/proj/graygo");root=Path(__file__).resolve().parent;checkout=repo/".worktrees/nfsp-gray-cpu-fa33155"
allocation=checkout/"docs/experiments/nfsp-gray-cpu-price-allocation.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(allocation)=="aaed74d244c91d8f043b1f514da84946488b63cec2a7ea750a358bf194018d86"
a=json.loads(allocation.read_text());start=time.monotonic();child=None
env=dict(os.environ,PYTHONPATH=str(repo/a["source_package"])+":"+str(repo/a["source_verifier_checkout"]),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
command=[str(repo/".venv/bin/python"),str(checkout/a["verifier"]),"qualify","--repo",str(repo),"--specification",str(checkout/a["specification"]),"--output",str(root/"prices.json")]
job={"version":1,"status":"qualifying_source_bound_cpu_gray_prices","original_driver_pid":os.getpid(),"driver_sha256":sha(__file__),"allocation_sha256":sha(allocation),"verifier_sha256":sha(checkout/a["verifier"]),"command":command,"new_training_records":0,"new_accelerator_jobs":0,"new_control_allocation_seconds":"0","a2_complete":False,"execution_authorized":False}
def save():
 job["measured_driver_wall_seconds"]=time.monotonic()-start
 p=root/"job.tmp";p.write_text(json.dumps(job,indent=2,sort_keys=True)+"\n");p.replace(root/"job.json")
try:
 assert sha(checkout/a["verifier"])==a["verifier_sha256"] and sha(checkout/a["specification"])==a["specification_sha256"]
 with (root/"qualification.stdout").open("wb") as out,(root/"qualification.stderr").open("wb") as err:
  tick=time.monotonic();child=subprocess.Popen(command,cwd=checkout,env=env,stdout=out,stderr=err,start_new_session=True);job["original_qualification_pid"]=child.pid;save()
  job["returncode"]=child.wait(timeout=a["outer_watchdog_seconds"]-10);job["measured_process_seconds"]=time.monotonic()-tick
 assert job["returncode"]==0
 proof=json.loads((root/"prices.json").read_text());assert proof["status"]=="qualified_cpu_gray_neural_prices"
 job.update(status=proof["status"],price_receipt_sha256=sha(root/"prices.json"),cpu_process_caps_seconds={t:p["cpu_process_ceiling_seconds"] for t,p in proof["prices"].items()})
except BaseException as e:job.update(status="original_price_qualification_failed_preserved",error=repr(e));traceback.print_exc()
finally:
 if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
 save()

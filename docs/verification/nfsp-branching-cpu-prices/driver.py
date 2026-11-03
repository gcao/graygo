from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
repo=Path("/home/gcao/proj/graygo");root=Path(__file__).resolve().parent;checkout=repo/".worktrees/nfsp-branching-cpu-19b2523"
allocation=checkout/"docs/experiments/nfsp-branching-cpu-price-allocation.json";sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(allocation)=="1c450af55bc17616795f5522a633eaa619a06308770ebd11b60f49ebd3b2c6a8"
a=json.loads(allocation.read_text());start=time.monotonic();child=None
env=dict(os.environ,PYTHONPATH=str(repo/a["source_package"])+":"+str(repo/a["source_verifier_checkout"]),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
command=[str(repo/".venv/bin/python"),str(checkout/a["runner"]),"qualify","--repo",str(repo),"--proposal",str(repo/a["proposal"]),"--evidence",str(repo/a["evidence"]),"--output",str(root/"prices.json")]
job={"version":1,"status":"qualifying_cpu_branching_neural_price","original_driver_pid":os.getpid(),"driver_sha256":sha(__file__),"allocation_sha256":sha(allocation),"runner_sha256":sha(checkout/a["runner"]),"command":command,"new_training_records":0,"new_accelerator_jobs":0,"new_control_allocation_seconds":"0","execution_authorized":False,"a2_complete":False}
def save():
 job["measured_driver_wall_seconds"]=time.monotonic()-start
 p=root/"job.tmp";p.write_text(json.dumps(job,indent=2,sort_keys=True)+"\n");p.replace(root/"job.json")
try:
 assert sha(checkout/a["runner"])==a["runner_sha256"] and sha(repo/a["proposal"])==a["proposal_sha256"] and sha(repo/a["evidence"])==a["evidence_sha256"]
 with (root/"qualification.stdout").open("wb") as out,(root/"qualification.stderr").open("wb") as err:
  tick=time.monotonic();child=subprocess.Popen(command,cwd=checkout,env=env,stdout=out,stderr=err,start_new_session=True);job["original_qualification_pid"]=child.pid;save()
  job["returncode"]=child.wait(timeout=a["outer_watchdog_seconds"]-10);job["measured_process_seconds"]=time.monotonic()-tick
 assert job["returncode"]==0
 proof=json.loads((root/"prices.json").read_text());assert proof["status"]=="qualified_cpu_branching_neural_price"
 job.update(status=proof["status"],price_receipt_sha256=sha(root/"prices.json"),cpu_process_ceiling_seconds=proof["price"]["cpu_process_ceiling_seconds"],control_scope_approved=False)
except BaseException as e:job.update(status="price_qualification_failed_preserved",error=repr(e));traceback.print_exc()
finally:
 if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
 save()

from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
repo=Path("/home/gcao/proj/graygo");root=Path(__file__).resolve().parent;checkout=repo/".worktrees/delayed-r1-comparison-75ab97c"
allocation=checkout/"docs/experiments/delayed-r1-comparison-source-allocation.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(allocation)=="597fccbc972dfb69cbd56cdb6336bc02372f5e5a0d24529e174665fbfe2e8298"
a=json.loads(allocation.read_text());start=time.monotonic();child=None
env=dict(os.environ,PYTHONPATH=str(repo/a["source_package"])+":"+str(repo/a["source_verifier_checkout"])+":"+str(checkout),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
command=[str(repo/".venv/bin/python"),str(checkout/a["qualifier"]),"--repo",str(repo),"--specification",str(checkout/a["specification"]),"--output",str(root/"admission.json")]
job={"version":1,"status":"qualifying_original_R1_comparison_source","original_driver_pid":os.getpid(),"driver_sha256":sha(__file__),"allocation_sha256":sha(allocation),"qualifier_sha256":sha(checkout/a["qualifier"]),"command":command,"new_training_records":0,"new_accelerator_jobs":0,"new_control_allocation_seconds":"0","primary_A2_slots_fillable":False,"a2_complete":False,"execution_authorized":False}
def save():
 job["measured_driver_wall_seconds"]=time.monotonic()-start
 p=root/"job.tmp";p.write_text(json.dumps(job,indent=2,sort_keys=True)+"\n");p.replace(root/"job.json")
try:
 assert sha(checkout/a["qualifier"])==a["qualifier_sha256"] and sha(checkout/a["specification"])==a["specification_sha256"]
 with (root/"qualification.stdout").open("wb") as out,(root/"qualification.stderr").open("wb") as err:
  tick=time.monotonic();child=subprocess.Popen(command,cwd=checkout,env=env,stdout=out,stderr=err,start_new_session=True);job["original_qualification_pid"]=child.pid;save()
  job["returncode"]=child.wait(timeout=a["outer_watchdog_seconds"]-10);job["measured_process_seconds"]=time.monotonic()-tick
 assert job["returncode"]==0
 proof=json.loads((root/"admission.json").read_text());assert proof["status"]=="qualified_cpu_delayed_R1_comparison_source"
 job.update(status=proof["status"],admission_sha256=sha(root/"admission.json"),complete_pipeline_pairs=len(proof["comparisons"]),cpu_process_cap_seconds=proof["cpu_prices"]["prices"]["C1"]["cpu_process_ceiling_seconds"])
except BaseException as e:job.update(status="original_R1_comparison_qualification_failed_preserved",error=repr(e));traceback.print_exc()
finally:
 if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
 save()

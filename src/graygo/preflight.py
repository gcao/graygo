"""Sequential, locked PPO route preflight; incomplete runs never choose a route."""
import fcntl
import json
import os
from pathlib import Path
import time
from .manifest import make_manifest, source_digest, write_report
from .matrix_training import MatrixPPOConfig, MatrixPPOTrainer, preflight_summary
from .storage import DEFAULT_RESERVE
from .reference_learning import matrix_reference_controls


def run_ppo_preflight(root, *, records=2_000_000, device="cpu", time_cap_seconds=24*3600,
                      stop_requested=lambda: False, reserve_bytes=DEFAULT_RESERVE):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/".suite.lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan={"format_version":1,"records_per_trial":records,"device":device,
              "time_cap_seconds":time_cap_seconds,"seeds":[0,1,2],
              "tasks":["asymmetric","rps"],"entropy_protocols":["taper","fixed-0.1"],
              "source_sha256":source_digest()}
        if (root/"plan.json").exists():
            if json.loads((root/"plan.json").read_text())!=plan:
                raise ValueError("preflight plan/source changed; create a new explicit run lineage")
        else:write_report(root/"plan.json",plan,reserve_bytes=reserve_bytes)
        started=time.monotonic();base_elapsed=0.;results=[]
        trials=[(task,entropy,seed) for entropy in plan["entropy_protocols"]
                for task in plan["tasks"] for seed in plan["seeds"]]
        # Already-completed cases are verified and reused, not restarted.
        for task,entropy,seed in trials:
            case=root/task/entropy/f"seed-{seed}"
            if (case/"result.json").exists():
                result=json.loads((case/"result.json").read_text())
                base_elapsed+=result.get("elapsed_seconds",0.)
        def state(status,current=None):
            report={"format_version":1,"pid":os.getpid(),"status":status,"current_trial":current,
                    "elapsed_this_process_seconds":time.monotonic()-started,
                    "route":preflight_summary(results)["route"],"completed_trials":len(results),
                    "source_sha256":plan["source_sha256"]}
            write_report(root/"state.json",report,reserve_bytes=reserve_bytes)
        state("reference_controls")
        control_path=root/"reference-controls.json"
        if control_path.exists():
            controls=json.loads(control_path.read_text())
            if controls.get("source_sha256")!=plan["source_sha256"]:
                raise ValueError("reference control source changed")
        else:
            controls={**matrix_reference_controls(),"source_sha256":plan["source_sha256"]}
            write_report(control_path,controls,reserve_bytes=reserve_bytes)
        if not controls["passed"]:
            state("reference_control_failed")
            raise RuntimeError("tabular matrix reference failed; neural preflight not started")
        state("running")
        for task,entropy,seed in trials:
            case=root/task/entropy/f"seed-{seed}"
            config=MatrixPPOConfig(task=task,entropy=entropy,seed=seed,records=records,
                                   device=device,time_cap_seconds=time_cap_seconds)
            resumed=(case/"run.json").exists()
            trainer=MatrixPPOTrainer(config,case,resume=resumed,reserve_bytes=reserve_bytes)
            if trainer.records==records:
                expected_result=trainer.result("complete")
                result_path=case/"result.json"
                if result_path.exists():
                    result=json.loads(result_path.read_text())
                    if any(result.get(key)!=value for key,value in expected_result.items() if key!="elapsed_seconds"):
                        raise ValueError("completed trial report disagrees with verified checkpoint")
                else:
                    result=expected_result
                    write_report(result_path,result,reserve_bytes=reserve_bytes)
            else:
                state("running",{"task":task,"entropy":entropy,"seed":seed,"records":trainer.records})
                def stop_reason():
                    if stop_requested():return "interrupted"
                    if base_elapsed+time.monotonic()-started>=time_cap_seconds:return "budget_limited"
                    return False
                result=trainer.run(stop_requested=stop_reason)
                # Initial stored elapsed is included in base_elapsed; don't count it twice.
                if result["status"]!="complete":
                    write_report(root/"summary.json",preflight_summary(results+[result]),reserve_bytes=reserve_bytes)
                    state(result["status"],{"task":task,"entropy":entropy,"seed":seed,"records":result["records"]})
                    return preflight_summary(results+[result])
            results.append(result)
            write_report(root/"summary.json",preflight_summary(results),reserve_bytes=reserve_bytes)
            state("running",{"finished_task":task,"entropy":entropy,"seed":seed})
        summary=preflight_summary(results)
        write_report(root/"summary.json",summary,reserve_bytes=reserve_bytes)
        state("complete")
        return summary

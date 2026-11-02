import json,os,time,subprocess
from pathlib import Path
import graygo
from graygo.nfsp_training import NFSPMatrixConfig,NFSPMatrixTrainer
from graygo.manifest import write_report,source_digest
root=Path('/home/gcao/proj/graygo/runs/nfsp-matrix-profile-6a92132')
root.mkdir(parents=True,exist_ok=True)
assert str(Path(graygo.__file__).resolve()).startswith('/home/gcao/proj/graygo/.worktrees/nfsp-matrix-6a92132/')
plan={'source_commit':subprocess.run(['git','rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip(),
      'source_sha256':source_digest(),'pid':os.getpid(),'records_per_case':100000,'scope':'terminal matrix full path; no multiround/9x9 pricing',
      'cases':[[phase,kind] for phase in ('pre','post') for kind in ('neural','tabular')]}
if (root/'plan.json').exists():raise FileExistsError('profile already exists; inspect authoritative job state instead of relaunching')
write_report(root/'plan.json',plan)
results=[];started=time.monotonic()
for phase,kind in plan['cases']:
    write_report(root/'state.json',{'status':'running','pid':os.getpid(),'current':[phase,kind],'completed':len(results)})
    config=NFSPMatrixConfig(learner=kind,records=100000,eval_every=50000,
            reservoir_open=5_000_000 if phase=='pre' else 0,handoff_end=7_000_000 if phase=='pre' else 1000,time_cap_seconds=1800)
    trainer=NFSPMatrixTrainer(config,root/phase/kind)
    result=trainer.run(verbose=False)
    if result['status']!='complete' or result['formal_protocol']:raise RuntimeError('profile did not complete as an explicitly nonformal diagnostic')
    row={'phase':phase,'learner':kind,'records':result['records'],'elapsed_seconds':result['elapsed_seconds'],
         'records_per_second':result['records']/result['elapsed_seconds'],'timings':result['timings'],
         'counters':result['counters'],'acting_policy_counts':result['acting_policy_counts'],'reservoir_size':result['reservoir_size'],
         'replay_size':trainer.replay.size,'replay_allocation_bytes':trainer.replay.allocated_bytes,
         'reservoir_allocation_bytes':trainer.reservoir.allocated_bytes,'episode_audit_bytes':result['episode_audit_bytes']}
    results.append(row);write_report(root/'profile.json',{'plan':plan,'results':results})
    print(json.dumps(row),flush=True)
write_report(root/'state.json',{'status':'complete','pid':os.getpid(),'completed':len(results),'elapsed_seconds':time.monotonic()-started})

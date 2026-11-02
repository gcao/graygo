"""Frozen-source seed-0 R4 screen after independently audited full R1 trials.

The sole learner change is one-step Q reuse 1 -> 4. This allocates only the first
paired tabular/neural screen, not all seeds, Retrace, extensions or 9x9 learning.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import graygo
from graygo.manifest import source_digest,write_report
from graygo.storage import CheckpointStore,digest
from graygo.task_training import NFSPTaskConfig,NFSPTaskTrainer

CONTROL_CEILING_SECONDS=12*3600


def require(condition,message):
    if not condition:raise ValueError(message)


def recorded_controls(project,exclude):
    rows=[]
    for path in sorted((project/'runs').glob('**/run.json')):
        if path.resolve().is_relative_to(exclude):continue
        manifest=json.loads(path.read_text());config=manifest.get('config',{})
        if config.get('learner')!='tabular':continue
        result_path=path.parent/'result.json'
        require(result_path.exists(),'unsettled tabular trial has no result: '+str(path.parent))
        result=json.loads(result_path.read_text());elapsed=result.get('elapsed_seconds')
        require(result.get('status')=='complete' and isinstance(elapsed,(float,int)) and math.isfinite(elapsed) and elapsed>=0,
                'other tabular work is incomplete/unsettled: '+str(path.parent))
        store=CheckpointStore(path.parent);bundle=store.latest();sealed=store.verify(bundle)
        require(sealed['step']==result['records'],'recorded control result is not its durable endpoint')
        require(sealed['metadata']['source_sha256']==manifest['source_sha256'],'recorded control source differs')
        rows.append({'run':str(path.parent.resolve()),'task':config.get('task'),'purpose':config.get('purpose','matrix'),
            'records':result['records'],'elapsed_seconds':float(elapsed),'manifest_sha256':digest(path),'result_sha256':digest(result_path),
            'checkpoint_manifest_sha256':digest(bundle/'manifest.json')})
    return {'unit':'recorded tabular control/profile process wall seconds','ceiling_seconds':CONTROL_CEILING_SECONDS,
            'excluded':'unit tests, independent audits and bootstrap overhead are not a complete development-cost total',
            'trials':rows,'elapsed_seconds':sum(row['elapsed_seconds'] for row in rows)}


def r1_evidence(root,audit_tool):
    state=json.loads((root/'state.json').read_text())
    require(state['status']=='complete' and state['completed_trials']==2,'both original R1 trials must be terminal')
    try:os.kill(state['pid'],0)
    except ProcessLookupError:pass
    else:raise ValueError('inspect the original R1 OS handle before starting another comparison')
    summaries={};configs={};auditor=digest(audit_tool)
    for learner in ('tabular','neural'):
        folder=root/learner;manifest=json.loads((folder/'run.json').read_text());config=NFSPTaskConfig(**manifest['config'])
        result=json.loads((folder/'result.json').read_text());audit=json.loads((folder/'audit-independent.json').read_text())
        require(config.formal_protocol() and config.q_reuse==1 and config.seed==0 and config.task=='delayed','full original seed-0 R1 protocol required')
        require(manifest['source_sha256']==source_digest() and result['source_sha256']==source_digest()==audit['source_sha256'],'R1 and R4 must use the same pinned learner source')
        require(result['status']=='complete' and result['records']==audit['records']==15000000 and audit['verified'] is True and audit['formal_protocol'] is True,'complete independently audited R1 evidence required')
        require(audit.get('auditor_sha256')==auditor and audit['training_manifest_sha256']==digest(folder/'run.json'),'audit code/run binding differs')
        bundle=CheckpointStore(folder).latest();CheckpointStore.verify(bundle)
        require(audit['checkpoint_manifest_sha256']==digest(bundle/'manifest.json'),'R1 endpoint changed after audit')
        require(audit['q_credit_passed']==result['q_credit_passed'] and audit['formal_case_passed']==result['formal_case_passed'],'R1 result and independent grade differ')
        tail=audit['scheduled_evaluations'][-5:]
        require(len(tail)==5 and [e['records'] for e in tail]==[13000000,13500000,14000000,14500000,15000000],'original final-five window required')
        max_average=max(e['max_average_gap'] for e in tail);max_error=max(e['max_early_error'] for e in tail)
        summaries[learner]={'manifest_sha256':digest(folder/'run.json'),'result_sha256':digest(folder/'result.json'),
            'audit_sha256':digest(folder/'audit-independent.json'),'checkpoint_manifest_sha256':digest(bundle/'manifest.json'),
            'q_credit_passed':audit['q_credit_passed'],'average_numerical_gap_band_passed':all(e['average_ready'] for e in tail) and max_average<=.1,
            'final_five_max_average_gap':max_average,'final_five_max_early_error':max_error,
            'final_five_max_greedy_gap':max(e['max_greedy_gap'] for e in tail),'actual_elapsed_seconds':result['elapsed_seconds']}
        configs[learner]=config
    if all(row['q_credit_passed'] and row['average_numerical_gap_band_passed'] for row in summaries.values()):
        raise ValueError('R1 is ready at this screen; R4 needs a separately justified cost comparison')
    reason='supported delayed-credit miss' if not all(row['q_credit_passed'] for row in summaries.values()) else 'declared earlier-Q/averaging-lag comparison'
    return summaries,configs,reason


def prepare_plan(args,root,project):
    require(source_digest()==args.expected_source_sha,'wrong pinned learner source')
    evidence,previous,reason=r1_evidence(Path(args.r1_run_dir).resolve(),Path(args.audit_tool).resolve())
    profiles=Path(args.profile_dir).resolve();state=json.loads((profiles/'state.json').read_text())
    require(state['status']=='complete','completed pre/post budget profiles required')
    settings=[];budgets=[]
    for learner in ('tabular','neural'):
        path=profiles/'budgets'/f'{learner}-R4.json';budget=json.loads(path.read_text())
        require(budget['source_sha256']==source_digest() and budget['task']=='delayed' and budget['learner']==learner and budget['q_reuse']==4
                and budget['target']=='one-step' and budget['complete_pipeline'] is True and set(budget['phases'])=={'pre','post'},'funding profile is not this complete R4 learner')
        cap=budget['proposed_time_ceiling_seconds']
        require(math.isfinite(cap) and cap>=2*budget['phase_weighted_15M_seconds'],'matched measured profile and twofold margin required')
        old=previous[learner];changed=old.plan()|{'q_reuse':4,'time_cap_seconds':cap,'budget_profile_sha256':digest(path)}
        config=NFSPTaskConfig(**changed)
        require(config.formal_protocol(),'complete 15M primary R4 protocol required')
        difference={key for key in old.plan() if old.plan()[key]!=config.plan()[key]}
        require(difference=={'q_reuse','time_cap_seconds','budget_profile_sha256'},'R4 changes more than reuse and its funded ceiling')
        settings.append(config);budgets.append(budget)
    controls=recorded_controls(project,root)
    tabular_cap=next(c.time_cap_seconds for c in settings if c.learner=='tabular')
    require(controls['elapsed_seconds']+tabular_cap<=CONTROL_CEILING_SECONDS,'first R4 control does not fit the recorded twelve-hour CPU-control allocation')
    allocation={'scope':'only seed-0 R4 tabular trial reserved; no other new tabular allocation until its ledger is settled',
        'recorded_spent_seconds':controls['elapsed_seconds'],'reserved_ceiling_seconds':tabular_cap,
        'projected_maximum_seconds':controls['elapsed_seconds']+tabular_cap,'ceiling_seconds':CONTROL_CEILING_SECONDS,
        'unallocated_seconds':CONTROL_CEILING_SECONDS-controls['elapsed_seconds']-tabular_cap,
        'later_cohort_rule':'all remaining seeds/tasks require a fresh ledger/cost decision; no blanket three-seed allocation'}
    plan={'format_version':1,'scope':'seed-0 paired one-step R4 delayed screen after complete independently audited R1; full objective unchanged',
        'source_sha256':source_digest(),'source_commit':subprocess.run(['git','rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip(),
        'driver_sha256':digest(Path(__file__)),'audit_tool_sha256':digest(Path(args.audit_tool).resolve()),'selection_reason':reason,
        'r1_evidence':evidence,'recorded_controls':controls,'control_allocation':allocation,'trial_settings':[c.plan() for c in settings],
        'budget_profiles':budgets,'aggregate_process_time_ceiling_seconds':sum(c.time_cap_seconds for c in settings),
        'not_selected':['other seeds','Retrace','uncorrected diagnostic','marginal extension','later opening','9x9 learning or confirmation'],
        'device_accounting':'both trials use CPU as profiled; CPU process wall times are not GPU-hours or a 9x9 campaign price'}
    return plan,settings


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--profile-dir',required=True)
    parser.add_argument('--r1-run-dir',required=True);parser.add_argument('--audit-tool',required=True)
    parser.add_argument('--project-dir',required=True);parser.add_argument('--expected-source-sha',required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=True);project=Path(args.project_dir).resolve()
    source=Path(graygo.__file__).resolve().parents[2];os.chdir(source)
    with (root/'.driver.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,settings=prepare_plan(args,root,project);path=root/'plan.json'
        if args.resume:
            require(json.loads(path.read_text())==plan,'R4 resume source/evidence/profile/resource plan differs')
        elif path.exists():raise FileExistsError('R4 screen exists; inspect its original handle before explicit resume')
        else:
            write_report(path,plan)
            for config in settings:
                raw=(Path(args.profile_dir)/'budgets'/f'{config.learner}-R4.json').read_bytes()
                (root/f'{config.learner}-budget.json').write_bytes(raw)
        if args.prepare_only:
            write_report(root/'state.json',{'status':'prepared','pid':os.getpid(),'source_sha256':source_digest(),'completed_trials':0})
            print(json.dumps({'status':'prepared','selection_reason':plan['selection_reason'],'allocation':plan['control_allocation']},sort_keys=True));return
        stopped=[]
        for name in (signal.SIGINT,signal.SIGTERM):signal.signal(name,lambda s,f:stopped.append(s))
        started=time.monotonic();results=[];status='complete'
        for config in settings:
            if stopped:status='interrupted';break
            write_report(root/'state.json',{'status':'running','pid':os.getpid(),'current_trial':config.learner,'completed_trials':len(results),'source_sha256':source_digest()})
            folder=root/config.learner;trainer=NFSPTaskTrainer(config,folder,resume=(folder/'run.json').exists())
            result=trainer.run(stop_requested=lambda:bool(stopped));results.append(result)
            write_report(root/'summary.json',{'results':results,'a2_complete':False,'selection':'R4 first screen only','source_sha256':source_digest()})
            if result['status']!='complete':status=result['status'];break
        write_report(root/'state.json',{'status':status,'pid':os.getpid(),'current_trial':None,'completed_trials':sum(r['status']=='complete' for r in results),
            'elapsed_this_process_seconds':time.monotonic()-started,'source_sha256':source_digest()})


if __name__=='__main__':main()

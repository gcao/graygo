#!/usr/bin/env python3
"""Bounded clean-source full-shape, zero-record probe CLI verification."""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
from graygo.board_journal import BoardJournalPlan,BoardMilestoneJournal
from graygo.board_runtime import BoardNFSPRuntime,BoardRuntimeConfig
from graygo.confidence import canonical_sha
from graygo.manifest import source_digest,write_report
from graygo.storage import digest,DEFAULT_RESERVE


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--expected-source-sha',required=True);args=p.parse_args()
    started=time.monotonic();source=Path(__file__).resolve().parents[1];root=Path(args.run_dir).resolve()
    if source_digest()!=args.expected_source_sha:raise ValueError('diagnostic source differs from selected clean package')
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True):raise ValueError('freeze diagnostic source before launch')
    allocation=source/'docs/experiments/probe-policy-diagnostic-allocation.json';plan=json.loads(allocation.read_text())
    if plan['inference_time_ceiling_seconds']!=600 or plan['driver_process_wall_ceiling_seconds']!=900 or plan['policies']['records']!=0:
        raise ValueError('unexpected diagnostic allocation')
    if root.exists():raise FileExistsError('diagnostic handle exists; inspect its original result')
    root.mkdir(parents=True);(root/'receipts').mkdir();(root/'policies').mkdir()
    def alarm(signum,frame):raise TimeoutError('original diagnostic watchdog reached')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(plan['setup_watchdog_seconds'])
    binding={'source_commit':commit,'source_sha256':source_digest(),'driver_sha256':digest(__file__),'allocation_sha256':digest(allocation),
             'original_pid':os.getpid(),'run_dir':str(root),'started_at_unix':time.time(),'learning_records':0,'status':'preparing'}
    write_report(root/'binding.json',binding);paths={};initial=[]
    for seed in plan['policies']['seeds']:
        runtime=BoardNFSPRuntime(BoardRuntimeConfig(seed=seed,envs=1,device='cpu'))
        journal=BoardMilestoneJournal(root/'policies'/str(seed),runtime,BoardJournalPlan('diagnostic',2,(0,2)))
        paths[seed]=journal.capture()
        if runtime.records or runtime.replay.insertions or runtime.reservoir.seen or any(asdict(runtime.backend.counters).values()):
            raise ValueError('diagnostic initialization unexpectedly learned or collected')
        initial.append({'seed':seed,'archive':str(paths[seed]),'archive_manifest_sha256':digest(paths[seed]/'manifest.json'),
                        'training_manifest_sha256':digest(journal.root/'run.json'),'records':0,'shape':[64,8,8]})
        del journal,runtime;gc.collect()
    setup=time.monotonic()-started
    if setup>plan['setup_watchdog_seconds']:raise TimeoutError('initialization exceeded its original ceiling')
    signal.alarm(max(1,int(plan['driver_process_wall_ceiling_seconds']-setup)))
    benchmark=source/'docs/verification/probe-multicell-v2-construction/benchmark.json'
    freeze=source/'docs/experiments/collision-benchmark-v2-freeze.json'
    inputs=[allocation,benchmark,freeze,*[p for run in (root/'policies').iterdir() for p in run.rglob('*') if p.is_file()]]
    before={str(path):digest(path) for path in inputs}
    base=[sys.executable,'-m','graygo','probe-evaluate','--run-dir',str(root/'evidence'),'--benchmark',str(benchmark),
          '--freeze-receipt',str(freeze),'--name','full-shape-zero-record-diagnostic','--purpose','diagnostic',
          '--time-cap-hours','0.16666666666666666','--device','cpu','--no-audit']
    for seed,path in paths.items():base+=['--policy',str(seed)+'='+str(path)]
    env=dict(os.environ,PYTHONPATH=str(source/'src'))
    command_receipts=[]
    def run(name,command):
        binding['status']=name;write_report(root/'state.json',binding)
        output=root/'receipts'/f'{name}.json';begin=time.monotonic()
        with (root/f'{name}.stdout.json').open('w') as stdout,(root/f'{name}.stderr.log').open('w') as stderr:
            result=subprocess.run(command+['--output',str(output)],cwd=source,env=env,stdout=stdout,stderr=stderr,
                timeout=max(1,plan['driver_process_wall_ceiling_seconds']-(time.monotonic()-started)))
        command_receipts.append({'name':name,'command':command+['--output',str(output)],'seconds':time.monotonic()-begin,
                                 'returncode':result.returncode,'receipt_sha256':digest(output) if output.exists() else None})
        if result.returncode:raise RuntimeError(f'{name} failed; inspect original stderr')
        return json.loads(output.read_text())['result']
    first=run('collect-eight',base+['--families','8'])
    if first['collection']['committed_families']!=8 or first['summary']['inference_complete']:raise ValueError('partial segment semantics differ')
    second=run('resume-complete',base+['--families','32','--resume'])
    if second['collection']['status']!='complete' or second['summary']['frozen_inference_reproduced']:raise ValueError('complete unaudited segment semantics differ')
    audit=run('reload-forward-audit',[sys.executable,'-m','graygo','probe-audit','--run-dir',str(root/'evidence')])
    if not audit['audit']['verified'] or audit['audit']['checked_families']!=32:raise ValueError('original archive forwards did not reproduce')
    evidence_before={str(path):digest(path) for path in (root/'evidence').rglob('*') if path.is_file()}
    report=run('read-only-report',[sys.executable,'-m','graygo','probe-report','--run-dir',str(root/'evidence')])
    if evidence_before!={str(path):digest(path) for path in (root/'evidence').rglob('*') if path.is_file()}:raise ValueError('standalone report changed evidence')
    if before!={str(path):digest(path) for path in inputs}:raise ValueError('original policies or frozen benchmark changed')
    if source_digest()!=binding['source_sha256']:raise ValueError('source changed during the frozen diagnostic')
    db=sqlite3.connect((root/'evidence/evidence.sqlite').as_uri()+'?mode=ro',uri=True)
    try:rows=list(db.execute('SELECT idx,payload,sha256 FROM families ORDER BY idx'))
    finally:db.close()
    if [row[0] for row in rows]!=list(range(32)):raise ValueError('original complete family prefix differs')
    for _,payload,sha in rows:
        import hashlib
        if hashlib.sha256(payload.encode()).hexdigest()!=sha:raise ValueError('raw scored-family checksum differs')
        row=json.loads(payload)
        if set(row['marginals'])!={'200','201','202'}:raise ValueError('raw original seed inventory differs')
        for pair in row['marginals'].values():
            if set(pair)!={'before','after'} or any(len(roles)!=2 for roles in pair.values()):raise ValueError('raw paired root/role inventory differs')
    if not report['within_inference_budget'] or not report['frozen_inference_reproduced'] or report['clock']['reproduced_families']!=32:
        raise ValueError('frozen audit budget/coverage differs')
    for key in ('archive_policy_binding_verified','confirmation_behavioral_criterion_met','a2_complete','gray_specialization_demonstrated'):
        if report[key]:raise ValueError('zero-record diagnostic claimed a confirmation prerequisite')
    elapsed=time.monotonic()-started
    result={**binding,'status':'verified','driver_seconds':elapsed,'initialization_seconds':setup,'initial_archives':initial,
            'commands':command_receipts,'original_inputs_unchanged':True,'standalone_report_read_only':True,
            'families':32,'roots':64,'seeds':3,'role_marginals':384,'reproduced_role_marginals':384,
            'passed_families_per_seed':{str(row['seed']):row['passed_families'] for row in report['grade']['seeds']},
            'inference_contract_sha256':report['contract_sha256'],'evidence_sha256':report['evidence_sha256'],
            'clock':report['clock'],'inference_time_ceiling_seconds':600,'disk_reserve_bytes':DEFAULT_RESERVE,
            'zero_record_diagnostic_only':True,'confirmation_criterion_met':False,'gray_specialization_demonstrated':False,
            'scope':'full-shape CPU CLI inference/recovery/forward-reproduction verification; no learning records or trained behavioral evidence'}
    write_report(root/'verification.json',result);write_report(root/'state.json',result);signal.alarm(0)
    print(json.dumps({key:result[key] for key in ('status','source_commit','driver_seconds','clock','passed_families_per_seed')}))


if __name__=='__main__':main()

"""Run one preregistered original delayed neural seed under qualified CPU prices."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import time
from graygo.manifest import source_digest,write_report
from graygo.storage import digest
from graygo.task_training import NFSPTaskConfig,NFSPTaskTrainer
from tools.a2_source_cohort import read_registry,check_selected_cpu_case
from tools.nfsp_delayed_cpu_prices import read_qualified,check_case,runtime_identity
from tools.nfsp_resources import require


def case_settings(allocation,seed,proof,identity):
    require(type(seed) is int and allocation['seeds']==[1,2] and seed in allocation['seeds'],'only the two original remaining neural seeds are selected')
    require(allocation['task']=='delayed' and allocation['learner']=='neural' and allocation['target']=='one-step' and allocation['q_reuse']==4,'only primary one-step R4 delayed neural cases are selected')
    require(set(allocation['case_configs'])=={'1','2'} and set(allocation['case_roots'])=={'1','2'},'complete fixed seed/config/root inventory required')
    require(len(set(allocation['case_roots'].values()))==2,'original cases require distinct roots')
    require(all(not Path(p).is_absolute() and Path(p).parts and Path(p).parts[0]=='runs' and '..' not in Path(p).parts for p in allocation['case_roots'].values()),'case roots must stay inside project runs')
    require(allocation['additional_control_reservation_seconds']=='0' and allocation['new_accelerator_jobs']==0,'CPU neural allocation cannot spend controls or accelerator time')
    require(allocation['learner_source_sha256']==proof['target_runtime_source_sha256'],'allocated source is not priced')
    config=NFSPTaskConfig(**allocation['case_configs'][str(seed)])
    require(config.seed==seed,'requested original seed differs from case config')
    preview=dict(identity,source_sha256=allocation['learner_source_sha256'],config=config.plan());check_case(preview,proof,'C4')
    return config


@contextmanager
def fresh_case_directory(root):
    """Hold a separate lease; an existing root is never an implicit resume."""
    root=Path(root).resolve();root.parent.mkdir(parents=True,exist_ok=True)
    with (root.parent/('.'+root.name+'.case.lock')).open('a+b') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not root.exists(),'original case root exists; inspect its original handle and preserve it')
        yield root


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--repo',required=True);parser.add_argument('--allocation',required=True);parser.add_argument('--allocation-sha',required=True);parser.add_argument('--seed',type=int,required=True);args=parser.parse_args(argv)
    repo=Path(args.repo).resolve();allocation_path=Path(args.allocation).resolve();allocation=json.loads(allocation_path.read_text())
    require(digest(allocation_path)==args.allocation_sha,'committed allocation changed')
    require(source_digest()==allocation['learner_source_sha256'] and digest(Path(__file__))==allocation['runner_sha256'],'wrong frozen learner or runner')
    require(args.seed in allocation['seeds'],'seed not selected')
    root=repo/allocation['case_roots'][str(args.seed)];stopped=[]
    for name in (signal.SIGTERM,signal.SIGINT):signal.signal(name,lambda s,f:stopped.append(s))
    with fresh_case_directory(root):
        admitted=time.monotonic();price_path=repo/allocation['cpu_price_receipt'];registry_path=repo/allocation['source_registry'];plan_path=repo/allocation['own_source_plan']
        require(digest(price_path)==allocation['cpu_price_receipt_sha256'] and digest(registry_path)==allocation['source_registry_sha256'] and digest(plan_path)==allocation['own_source_plan_sha256'],'price/source registration changed')
        proof=read_qualified(price_path);config=case_settings(allocation,args.seed,proof,runtime_identity())
        body,base,plans,edges=read_registry(registry_path,repo=repo,current=True)
        own=[p for p in plans.values() if p.learner_source_sha256==allocation['learner_source_sha256']]
        require(len(own)==1 and own[0].to_dict()==json.loads(plan_path.read_text()) and own[0].multiround_reuse==4,'original own-source A2 plan is absent or changed')
        bridge=next(e['proof'] for e in edges if e['proof'].get('candidate_source_sha256')==allocation['learner_source_sha256'] and e['proof'].get('source_compatibility_qualified'))
        preview=dict(runtime_identity(),source_sha256=source_digest(),config=config.plan());check_selected_cpu_case(preview,bridge)
        require(not stopped,'stopped before original model initialization')
        root.mkdir();write_report(root/'admission.json',{'version':1,'allocation_sha256':args.allocation_sha,'source_sha256':source_digest(),'seed':args.seed,'price_receipt_sha256':digest(price_path),'source_registry_sha256':digest(registry_path),'own_source_plan_sha256':digest(plan_path),'measured_preinitialization_seconds':time.monotonic()-admitted,'fresh_prices_and_source_inputs_verified':True,'a2_complete':False,'execution_authorized_by_assessment':False})
        setup=time.monotonic();trainer=NFSPTaskTrainer(config,root);setup_seconds=time.monotonic()-setup
        check_case(trainer.manifest,proof,'C4');check_selected_cpu_case(trainer.manifest,bridge)
        write_report(root/'case-state.json',{'status':'running','seed':args.seed,'source_sha256':source_digest(),'config':config.plan(),'original_pid':os.getpid(),'initial_checkpoint_step':trainer.store.verify(trainer.store.latest())['step'],'constructor_seconds':setup_seconds})
        result=trainer.run(stop_requested=lambda:'interrupted' if stopped else False)
        write_report(root/'process-portion.json',{'constructor_seconds':setup_seconds,'loop_seconds':result['elapsed_seconds'],'known_constructor_and_loop_seconds':setup_seconds+result['elapsed_seconds'],'scope':'known constructor and complete cumulative trainer loop; pre-entry validation is separate and outer overhead remains unknown'})
        write_report(root/'case-state.json',{'status':result['status'],'seed':args.seed,'records':result['records'],'source_sha256':source_digest(),'original_pid':os.getpid(),'a2_complete':False})
        print(json.dumps({'status':result['status'],'records':result['records'],'seed':args.seed,'a2_complete':False}),flush=True)


if __name__=='__main__':main()

"""Recompute formal A2 gates from checksummed archived model weights.

Run with the pinned source directory on PYTHONPATH and that worktree as cwd.
Closed-form game values below are independent of the numerical LP evaluator.
"""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
import graygo
from graygo.matrix_training import MatrixActorCritic
from graygo.manifest import source_digest,write_report
from graygo.storage import CheckpointStore


def audit(root):
    root=Path(root)
    plan=json.loads((root/'plan.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    if plan['source_sha256']!=source_digest():raise ValueError('audit must use the original pinned source')
    if not summary['complete'] or summary['trials']!=12:raise ValueError('formal preflight is incomplete')
    if plan['records_per_trial']!=2_000_000:raise ValueError('short diagnostic cannot select a route')
    torch.set_num_threads(1)
    cases=[]
    expected_keys={(task,protocol,seed) for task in ('asymmetric','rps') for protocol in ('taper','fixed-0.1') for seed in (0,1,2)}
    actual_keys={(r['task'],r['entropy_protocol'],r['seed']) for r in summary['results']}
    if len(summary['results'])!=12 or actual_keys!=expected_keys:raise ValueError('missing or duplicate formal cases')
    for task,protocol,seed in sorted(expected_keys):
        case=root/task/protocol/f'seed-{seed}'
        manifest=json.loads((case/'run.json').read_text())
        if manifest['source_sha256']!=plan['source_sha256']:raise ValueError('source drift in trial')
        config=manifest['config']
        for key,expected in {'records':2_000_000,'rollout':1024,'eval_every':102400,'learning_rate':3e-4,'epochs':4,'minibatch':128}.items():
            if config[key]!=expected:raise ValueError('formal update plan differs')
        if manifest['ppo_base_settings']['target_kl']!=.02 or manifest['ppo_base_settings']['clip']!=.2:
            raise ValueError('PPO clipping/KL plan differs')
        a=np.array([[1.,-1.],[-1.,.5]]) if task=='asymmetric' else np.array([[0.,-1.,1.],[1.,0.,-1.],[-1.,1.,0.]])
        value=-1/7 if task=='asymmetric' else 0.
        np.testing.assert_array_equal(manifest['payoffs'],a)
        store=CheckpointStore(case)
        latest=store.latest();meta=store.verify(latest)
        if meta['step']!=2_000_000:raise ValueError('trial did not reach its planned record count')
        state=torch.load(latest/'trainer.pt',weights_only=True,map_location='cpu')
        if state['records']!=2_000_000 or state['per_role']!=[1_000_000,1_000_000]:raise ValueError('record accounting differs')
        if state['current_opponents']+state['historical_opponents']!=state['records']:raise ValueError('opponent accounting differs')
        if state['optimizer_steps']<=0:raise ValueError('no optimizer work recorded')
        mixed=[];latest_checks=[];mixture_checks=[];last=None
        for evaluation in state['evaluations']:
            step=evaluation['records'];archive=case/'weights'/f'{step:012d}'
            store.verify(archive)
            model_data=torch.load(archive/'model.pt',weights_only=True,map_location='cpu')
            model=MatrixActorCritic(max(a.shape),config['hidden'])
            model.load_state_dict(model_data['model']);model.eval()
            with torch.no_grad():
                # Reproduce the action distribution's numerical definition;
                # direct softmax differs from exp(log-softmax) by a float32 ulp.
                policies=model.policy_logits(torch.eye(2)).log_softmax(-1).exp().double().numpy()
            policies/=policies.sum(-1,keepdims=True)
            np.testing.assert_allclose(policies,evaluation['policies'],rtol=0,atol=1e-12)
            row_gap=max(0.,value-float((policies[0]@a).min()))
            col_gap=max(0.,float((a@policies[1]).max())-value)
            np.testing.assert_allclose([row_gap,col_gap],evaluation['latest_gaps'],rtol=0,atol=1e-10)
            latest_checks.append(max(row_gap,col_gap)<=.05)
            if step>=1_000_000:
                mixed.append(policies)
                mean=np.mean(mixed,axis=0);mean/=mean.sum(-1,keepdims=True)
                mg=[max(0.,value-float((mean[0]@a).min())),max(0.,float((a@mean[1]).max())-value)]
                np.testing.assert_allclose(mg,evaluation['mixture_gaps'],rtol=0,atol=1e-10)
                mixture_checks.append(max(mg)<=.05)
            else:mixture_checks.append(False)
            last=evaluation
        if len(latest_checks)<5:raise ValueError('too few final evaluations')
        latest_pass=all(latest_checks[-5:]);mixture_pass=all(mixture_checks[-5:])
        recorded=next(r for r in summary['results'] if (r['task'],r['entropy_protocol'],r['seed'])==(task,protocol,seed))
        if (latest_pass,mixture_pass)!=(recorded['latest_passed'],recorded['mixture_passed']):raise ValueError('stored gate differs from reconstructed weights')
        cases.append({'task':task,'entropy_protocol':protocol,'seed':seed,'records':state['records'],
                      'optimizer_steps':state['optimizer_steps'],'latest_passed':latest_pass,'mixture_passed':mixture_pass,
                      'last_latest_gaps':last['latest_gaps'],'last_mixture_gaps':last['mixture_gaps']})
    passing=[]
    for protocol in ('taper','fixed-0.1'):
        for representation in ('latest','mixture'):
            subset=[c for c in cases if c['entropy_protocol']==protocol]
            if all(c[representation+'_passed'] for c in subset):passing.append([protocol,representation])
    route='ppo' if passing else 'nfsp'
    if route!=summary['route']:raise ValueError('independent route decision differs')
    return {'format_version':1,'verified':True,'route':route,'passing_configurations':passing,
            'source_sha256':plan['source_sha256'],'source_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=Path(graygo.__file__).resolve().parents[2],check=True,capture_output=True,text=True).stdout.strip(),
            'total_learner_records':sum(c['records'] for c in cases),'trials':cases,
            'method':'checksummed final resume states and every archived evaluation model; closed-form game values; reconstructed episode mixtures'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_dir');parser.add_argument('--output',required=True)
    args=parser.parse_args();result=audit(args.run_dir);write_report(args.output,result)
    print(json.dumps(result,indent=2))

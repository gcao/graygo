"""Adopt head-to-head superiority with historical regression safeguards.

Reassesses complete, sealed evidence without new games. Original decisions,
evidence and learner payloads remain intact; publication is upgraded offline.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import numpy as np
from graygo.confidence import canonical_sha
from graygo.evaluation_evidence import FrozenEvaluationEvidence
from graygo.manifest import source_digest,write_report
from graygo.storage import DEFAULT_RESERVE,digest
from tools.champion_gate import GatePlan,IndependentGate,HEAD_TO_HEAD_HISTORY,POSITIVE_POOL_GAIN,LEGACY_POOL_GAIN,gate_root
from tools.champion_loop import policy
from tools.upgrade_gate_scheduler import wrapped


def reassess(stage,records):
    stage=Path(stage).resolve();league=wrapped(stage/'learner/league/state.json')['body']
    row=next(r for r in league['rounds'] if r['records']==records)
    original=row['decision']
    if original.get('status')!='complete' or 'criteria' not in original:
        raise ValueError('a complete registered promotion evaluation is required')
    old_policy=original.get('promotion_policy',POSITIVE_POOL_GAIN)
    if old_policy not in (POSITIVE_POOL_GAIN,LEGACY_POOL_GAIN):
        raise ValueError('only an original pool-gain gate can be reassessed')
    body=wrapped(stage/'plan.json')['body']
    args=dict(rounds=len(league['contract']['loop_plan']['round_ends']),
              alpha=body.get('gate_alpha',.05),seed=body.get('gate_seed',73))
    base=stage/'learner/league/evaluations'/f'{row["index"]:04d}'
    # Paths for original policies are versioned and remain immutable.
    probe=GatePlan(**args,pool_gain_policy=old_policy,improvement=.03 if old_policy==LEGACY_POOL_GAIN else 0.)
    root=gate_root(base,probe)
    contract=json.loads((root/'gate-plan.json').read_text())
    args['looks']=tuple(contract['family']['looks'])
    old=GatePlan(**args,pool_gain_policy=old_policy,improvement=.03 if old_policy==LEGACY_POOL_GAIN else 0.)
    current=GatePlan(**args,pool_gain_policy=HEAD_TO_HEAD_HISTORY)
    labels=tuple(e.split('/',1)[1] for e in contract['family']['endpoints'] if e.startswith('delta/'))
    snapshots={name:FrozenEvaluationEvidence.load(root/name) for name in ('head',*labels)}
    games=max(contract['family']['looks'])
    for name,evidence in snapshots.items():
        if evidence.sha256!=original['evidence'][name] or evidence.available_games!=games:
            raise ValueError('evaluation evidence changed or final planned look is incomplete')
        if evidence.contract['subject_sha256']!=row['candidate']['policy_sha256']:
            raise ValueError('evaluated candidate identity differs')
        if name!='head' and evidence.contract['opponent_sha256']!=contract['pool'][name]:
            raise ValueError('historical opponent identity differs')
    if snapshots['head'].contract['opponent_sha256']!=row['incumbent']['policy_sha256']:
        raise ValueError('head-to-head incumbent differs')
    for name in labels:
        if snapshots[name].contract['incumbent_sha256']!=row['incumbent']['policy_sha256']:
            raise ValueError('historical comparison incumbent differs')
    if current.family(row['index'],labels).sha256!=original['family_sha256']:
        raise ValueError('confidence family changed; existing bounds cannot be reused')
    head=[sum(float(np.sign(g['subject_utility'])) for g in r['subject'])/2 for r in snapshots['head'].rows]
    pool={name:[r['utility'] for r in snapshots[name].rows] for name in labels}
    incumbent={name:[r['incumbent_utility'] for r in snapshots[name].rows] for name in labels}
    reproduced=old.numeric(row['index'],head,pool,incumbent,games)
    decision=current.numeric(row['index'],head,pool,incumbent,games)
    for field in ('head','aggregate','opponents'):
        if canonical_sha(reproduced[field])!=canonical_sha(original[field]):
            raise ValueError('original statistics do not reproduce')
        if canonical_sha(decision[field])!=canonical_sha(original[field]):
            raise ValueError('retained confidence bounds changed')
    if reproduced['criteria']!=original['criteria']:
        raise ValueError('original decision does not reproduce')
    if not decision['numeric_pass']:raise ValueError('candidate fails the retained promotion safeguards')
    for descriptor in (row['candidate'],row['incumbent']):policy(descriptor).assert_frozen()
    for evidence in snapshots.values():evidence.assert_current()
    decision.update(status='complete',candidate_sha256=row['candidate']['policy_sha256'],
                    champion_sha256=row['incumbent']['policy_sha256'],evidence=original['evidence'],
                    experimental_promotion_permitted=True,promotion_permitted=False)
    return {'version':'graygo/promotion-policy-assessment/3','promotion_policy':HEAD_TO_HEAD_HISTORY,
            'records':records,'candidate':deepcopy(row['candidate']),'incumbent':deepcopy(row['incumbent']),
            'decision':decision,'original_decision_path':str(root/'decision.json'),
            'original_decision_sha256':digest(root/'decision.json'),'original_criteria':original['criteria'],
            'owner_approved':False,'formal_training_or_promotion_grant':False}


def apply(campaign,records,*,reserve_bytes=DEFAULT_RESERVE):
    campaign=Path(campaign).resolve();checkout=Path(__file__).resolve().parents[1];locks=[]
    def write(path,value):write_report(path,value,reserve_bytes=reserve_bytes)
    try:
        lock=(campaign/'.controller.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        campaign_plan=wrapped(campaign/'plan.json');campaign_state=json.loads((campaign/'state.json').read_text())
        if campaign_state['plan_sha256']!=campaign_plan['sha256'] or campaign_plan['body']['source_sha256']!=source_digest():
            raise ValueError('campaign identity or numerical source differs')
        if campaign_plan['body']['controller_sha256']!=digest(checkout/'tools/continuous_training.py'):
            raise ValueError('continuous controller changed')
        stage=Path(campaign_state['current_stage']);plan=wrapped(stage/'plan.json');learner=stage/'learner'
        publication=Path(plan['body']['publication']);parent=Path(plan['body']['initialization']['parent_run'])
        for path in (learner/'.execution.lock',learner/'league/.loop.lock',parent/'.execution.lock',
                     publication.with_name('.'+publication.name+'.publisher.lock')):
            lock=path.open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        clock=wrapped(learner/'execution.json')['body']['clock']
        if clock['inflight_since'] is not None:raise ValueError('stop training gracefully before adopting policy')
        league=wrapped(learner/'league/state.json');published=wrapped(publication)
        if league['body']['evaluation_inflight'] is not None:raise ValueError('evaluation is not closed')
        if published['body']['loop_contract_sha256']!=canonical_sha(league['body']['contract']):
            raise ValueError('another loop owns publication')
        for name,sha in plan['body']['tools'].items():
            if digest(Path(plan['body']['checkout'])/'tools'/name)!=sha:raise ValueError('original source changed')
            if name not in ('champion_gate.py','champion_loop.py','learning_loop.py') and digest(checkout/'tools'/name)!=sha:
                raise ValueError('upgrade changes more than promotion policy: '+name)
        if not plan['body'].get('experimental') or not published['body'].get('experimental'):
            raise ValueError('policy adoption is limited to the experimental learning loop')
        assessment=reassess(stage,records)
        state=league['body']
        if state['champion']['policy_sha256']!=assessment['incumbent']['policy_sha256']:
            raise ValueError('approved evidence is against a different current champion')
        reservation=json.loads((parent/'handoff.json').read_text())
        if reservation['destination']!=str(stage) or reservation['plan_sha256']!=plan['sha256']:
            raise ValueError('parent reservation differs')
        archive=stage/'policy-adoptions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        archive.mkdir(parents=True,exist_ok=False)
        originals={'campaign-plan.json':campaign_plan,'campaign-state.json':campaign_state,'stage-plan.json':plan,
                   'league-state.json':league,'publication.json':published,'parent-reservation.json':reservation}
        for name,value in originals.items():write(archive/name,value)
        assessment['owner_approved']=True
        assessment['adopted_at']=datetime.now(timezone.utc).isoformat();assessment_path=archive/'assessment-v3.json'
        write(assessment_path,assessment)
        gate_plan=GatePlan(len(state['contract']['loop_plan']['round_ends']),alpha=plan['body'].get('gate_alpha',.05),
                           seed=plan['body'].get('gate_seed',73),pool_gain_policy=HEAD_TO_HEAD_HISTORY)
        gate=IndependentGate(gate_plan,reserve_bytes=reserve_bytes)
        state['contract']['implementation_sha256']=digest(checkout/'tools/champion_loop.py')
        state['contract']['gate_sha256']=gate.sha256
        previous=deepcopy(state['champion']);state['history'].append(previous)
        state['champion']=deepcopy(assessment['candidate']);state['champion_assessment']=str(assessment_path)
        state['prior_promotions']=state.get('prior_promotions',0)+1
        state.setdefault('policy_reassessments',[]).append({'records':records,'policy':HEAD_TO_HEAD_HISTORY,
            'assessment':str(assessment_path),'assessment_sha256':canonical_sha(assessment)})
        pending=state.get('pending')
        if pending is not None:
            if pending.get('status') in ('evaluating','screening'):
                if clock['last_checkpoint']['records']!=pending['target']:
                    raise ValueError('pending evaluation is not at its frozen checkpoint')
                # The incumbent changed. Burn the old confidence slot instead
                # of collecting a second hypothesis under the same allocation.
                state['rounds'].append({'index':pending['index'],'records':pending['target'],
                    'candidate':pending.get('candidate'),'incumbent':pending['champion'],
                    'decision':{'status':'complete','numeric_pass':False,'promotion_permitted':False,
                        'experimental_promotion_permitted':False,'reason':'incumbent_changed_on_policy_adoption'},
                    'promoted':False,'learner_reinitialized':False,'buffers_flushed':False})
                state.update(round=pending['index']+1,completed_records=pending['target'],pending=None)
            else:
                pending['champion']=deepcopy(state['champion']);pending['history']=deepcopy(state['history'])
                pending['status']='training'
                for key in ('candidate','screen','schedule_action'):pending.pop(key,None)
        if state.get('adaptive_scheduler') is not None:
            from tools import adaptive_gate_schedule as adaptive
            learning=max(0.,clock['observed_seconds']+clock['recovery_charge_seconds']-state['evaluation_seconds'])
            gate_count=state['adaptive_scheduler']['gate_count']
            state['adaptive_scheduler']=adaptive.observe_gate(state['adaptive_scheduler'],assessment['decision'],
                learning,previous['policy_sha256'],promoted=True)
            state['adaptive_scheduler']['gate_count']=gate_count
            state['adaptive_scheduler']['last_signal']='promoted_by_reassessment'
        league['sha256']=canonical_sha(state)
        plan['body']['checkout']=str(checkout);plan['body']['promotion_policy']=HEAD_TO_HEAD_HISTORY
        for name in ('champion_gate.py','champion_loop.py','learning_loop.py'):plan['body']['tools'][name]=digest(checkout/'tools'/name)
        plan['sha256']=canonical_sha(plan['body'])
        reservation['previous_plan_sha256']=reservation['plan_sha256'];reservation['plan_sha256']=plan['sha256']
        campaign_plan['body']['checkout']=str(checkout);campaign_plan['body']['promotion_policy']=HEAD_TO_HEAD_HISTORY
        campaign_plan['sha256']=canonical_sha(campaign_plan['body']);campaign_state['plan_sha256']=campaign_plan['sha256']
        published['body'].update(champion=deepcopy(state['champion']),completed_round=state['round'],
            loop_contract_sha256=canonical_sha(state['contract']),loop_state_sha256=league['sha256'],
            promotion_policy=HEAD_TO_HEAD_HISTORY,champion_assessment=str(assessment_path))
        published['sha256']=canonical_sha(published['body'])
        write(stage/'plan.json',plan);write(parent/'handoff.json',reservation)
        write(learner/'league/state.json',league);write(learner/'league/champion.json',published)
        write(publication,published)
        write(campaign/'plan.json',campaign_plan);write(campaign/'state.json',campaign_state)
        receipt={'at':datetime.now(timezone.utc).isoformat(),'promotion_policy':HEAD_TO_HEAD_HISTORY,
            'promoted_records':records,'policy_sha256':state['champion']['policy_sha256'],
            'assessment':str(assessment_path),'archive':str(archive),'learner_checkpoint_records':clock['last_checkpoint']['records'],
            'training_checkpoint_or_optimizer_changed':False,'original_evidence_or_decision_changed':False,
            'head_win_score':assessment['decision']['head_win_score'],
            'pool_gain_lower':assessment['decision']['aggregate']['lower'],
            'pool_gain_required':False,'adoption_tool_sha256':digest(Path(__file__))}
        write(archive/'receipt.json',receipt)
        return receipt
    finally:
        for lock in reversed(locks):lock.close()


def main():
    import torch
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    parser.add_argument('--records',type=int,required=True);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if args.apply:report=apply(args.campaign,args.records)
    else:
        state=json.loads((Path(args.campaign)/'state.json').read_text())
        report=reassess(state['current_stage'],args.records)
    print(json.dumps(report,sort_keys=True))


if __name__=='__main__':main()

"""Audit and adopt one experimentally gated frozen policy release.

A completed, unused promotion slot is reserved before collecting independent
validation games. No old decision/evidence or learner payload is rewritten.
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
from tools.champion_gate import GatePlan,HEAD_TO_HEAD_HISTORY,gate_root
from tools.champion_screen import IndependentScreen,ScreenPlan
from tools.parallel_gate import ParallelGate,load_policy
from tools.upgrade_gate_scheduler import wrapped
from tools import adaptive_gate_schedule as adaptive
from tools import policy_selection
from tools.task_pipeline_parity import state_sha


def verify_bundle(descriptor):
    """The arena payload must contain the exact policy weights used by the gate."""
    if descriptor['kind']!='archive' or not descriptor.get('bundle'):return
    import torch
    bundle=Path(descriptor['bundle']);archive=Path(descriptor['archive'])
    if digest(bundle)!=descriptor['bundle_sha256']:raise ValueError('release bundle checksum differs')
    exported=torch.load(bundle,map_location='cpu',weights_only=True)
    source=torch.load(archive/'models.pt',map_location='cpu',weights_only=True)
    head=descriptor.get('head','average')
    if head not in ('average','greedy_q'):raise ValueError('unsupported release head')
    if state_sha(exported['state_dict'])!=state_sha(source['q' if head=='greedy_q' else 'average']):raise ValueError('release bundle weights differ from the gated archive')
    expected_format='graygo/arena-nfsp-policy/2' if head=='greedy_q' else 'graygo/arena-nfsp-average/1'
    if (exported['format_version']!=expected_format or exported['head']!=head
        or exported['encoding_version']!='graygo/encoding/1' or exported['board_size']!=descriptor['rules']['size']
        or (head=='greedy_q' and descriptor.get('selection') is not None)
        or exported['rules']!=descriptor['rules'] or exported['training_records']!=descriptor['records']
        or exported.get('selection')!=descriptor.get('selection')
        or exported['model_config']!={key:source['config'][key] for key in ('width','blocks','norm_groups')}
        or exported['source_archive_sha256']!=digest(archive/'manifest.json')):
        raise ValueError('release bundle metadata differs from the gated archive')


def assess(trial):
    trial=Path(trial).resolve();wrapper=wrapped(trial/'plan.json');plan=wrapper['body'];stage=Path(plan['allocation_stage'])
    if plan.get('version') not in ('graygo/selection-release-trial/1','graygo/q-release-trial/1'):raise ValueError('unknown release trial')
    is_q=plan['version']=='graygo/q-release-trial/1'
    if digest(trial/'parallel_gate.py')!=plan['parallel_runner_sha256']:raise ValueError('parallel execution source changed')
    if digest(Path(__file__).with_name('champion_gate.py'))!=plan['numeric_gate_sha256']:raise ValueError('numeric gate source changed')
    frozen_plan=trial/'allocation-stage-plan.json'
    stage_plan=wrapped(frozen_plan if frozen_plan.exists() else stage/'plan.json');state=wrapped(stage/'learner/league/state.json')['body']
    if stage_plan['sha256']!=plan['stage_plan_sha256'] or source_digest()!=plan['source_sha256']:
        raise ValueError('trial source or original stage allocation changed')
    row=next(r for r in state['rounds'] if r['index']==plan['round_index'])
    if canonical_sha(row)!=plan['original_row_sha256'] or 'criteria' in row['decision'] or row.get('promoted'):
        raise ValueError('allocated confidence slot was already used or changed')
    authority=stage/'learner/league/evaluations'/f"{plan['round_index']:04d}"/('external-q-allocation.json' if is_q else 'external-selection-allocation.json')
    if list(authority.parent.glob('external-*-allocation.json'))!=[authority]:raise ValueError('slot has conflicting external allocations')
    if json.loads(authority.read_text())['plan_sha256']!=wrapper['sha256']:raise ValueError('trial allocation differs')
    if list(authority.parent.glob('**/evidence.sqlite')):raise ValueError('ordinary gate also used this slot')
    gate_plan=GatePlan(**plan['gate_plan'])
    if gate_plan.rounds!=len(state['contract']['loop_plan']['round_ends']) or gate_plan.alpha!=stage_plan['body'].get('gate_alpha',.05):
        raise ValueError('campaign promotion error budget differs')
    if gate_plan.alpha/gate_plan.rounds!=plan['allocated_alpha']:raise ValueError('per-slot alpha differs')
    root=gate_root(trial/'evaluation',gate_plan);decision=json.loads((root/'decision.json').read_text())
    if decision.get('status')!='complete' or not decision.get('numeric_pass'):raise ValueError('candidate did not pass the full gate')
    labels=tuple(plan['pool']);family=gate_plan.family(plan['round_index'],labels)
    contract=json.loads((root/'gate-plan.json').read_text())
    if contract['family']!=family.to_dict():raise ValueError('registered confidence family differs')
    snapshots={name:FrozenEvaluationEvidence.load(root/name) for name in ('head',*labels)}
    games=decision['head']['games']
    if games not in gate_plan.looks:raise ValueError('decision is not a registered look')
    for name,evidence in snapshots.items():
        if evidence.available_games!=games or evidence.sha256!=decision['evidence'][name]:raise ValueError('incomplete or changed gate evidence')
        if evidence.contract['subject_sha256']!=plan['candidate']['policy_sha256']:raise ValueError('candidate identity differs')
        if name=='head':
            if evidence.contract['opponent_sha256']!=plan['incumbent']['policy_sha256']:raise ValueError('incumbent differs')
        elif (evidence.contract['opponent_sha256']!=plan['pool'][name]['policy_sha256']
              or evidence.contract['incumbent_sha256']!=plan['incumbent']['policy_sha256']):raise ValueError('historical identity differs')
    reproduced=gate_plan.numeric(plan['round_index'],
        [sum(float(np.sign(g['subject_utility'])) for g in r['subject'])/2 for r in snapshots['head'].rows],
        {name:[r['utility'] for r in snapshots[name].rows] for name in labels},
        {name:[r['incumbent_utility'] for r in snapshots[name].rows] for name in labels},games)
    for key in ('head','aggregate','opponents','criteria','head_win_score','numeric_pass','family_sha256'):
        if canonical_sha(reproduced[key])!=canonical_sha(decision[key]):raise ValueError('gate statistics do not reproduce')
    for descriptor in (plan['candidate'],plan['incumbent'],*plan['pool'].values()):
        load_policy(descriptor).assert_frozen();verify_bundle(descriptor)
    for evidence in snapshots.values():evidence.assert_current()
    parity=json.loads((trial/'deployment-parity.json').read_text())
    parity_input=json.loads((trial/'deployment-parity-input.json').read_text())
    if parity_input['candidate']!=plan['candidate'] or parity['states']!=len(parity_input['cases']):raise ValueError('deployment parity tested another candidate')
    if is_q:
        import math
        candidate=plan['candidate']
        if candidate.get('head')!='greedy_q' or candidate.get('selection') is not None:raise ValueError('registered Q representation differs')
        if (parity.get('candidate_sha256')!=candidate['policy_sha256'] or parity.get('head')!='greedy_q'
            or parity['states']<20 or not math.isfinite(parity['maximum_probability_error'])
            or parity['maximum_probability_error']>=1e-9):raise ValueError('greedy Q deployment parity is not verified')
        for path,sha in parity['arena_sources'].items():
            if digest(Path(parity['arena_source_checkout'])/path)!=sha:raise ValueError('tested Q arena source changed')
    else:
        if parity['selection_sha256']!=plan['candidate']['selection']['implementation_sha256'] or parity['maximum_probability_error']>=2e-5:
            raise ValueError('deployment probability parity is not verified')
        if plan['candidate']['selection']!=policy_selection.selection_spec(0.,True,1e-6):raise ValueError('untested selection profile')
    return {'version':'graygo/q-release-assessment/1' if is_q else 'graygo/selection-assessment/1','promotion_policy':HEAD_TO_HEAD_HISTORY,
        'trial':str(trial),'records':plan['candidate']['records'],'candidate':plan['candidate'],'incumbent':plan['incumbent'],
        'pool':plan['pool'],'decision':decision,'evaluation_root':str(trial/'evaluation'),
        'allocation':str(authority),'allocation_plan_sha256':wrapper['sha256'],'round_index':plan['round_index'],
        'deployment_parity':parity,'formal_training_or_promotion_grant':False}


def apply(campaign,trial,*,reserve_bytes=DEFAULT_RESERVE):
    campaign=Path(campaign).resolve();checkout=Path(__file__).resolve().parents[1];locks=[]
    def write(path,value):write_report(path,value,reserve_bytes=reserve_bytes)
    try:
        for path in (campaign/'.controller.lock',):
            lock=path.open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        campaign_plan=wrapped(campaign/'plan.json');campaign_state=json.loads((campaign/'state.json').read_text())
        if campaign_state['plan_sha256']!=campaign_plan['sha256'] or campaign_plan['body']['source_sha256']!=source_digest():raise ValueError('campaign source/plan differs')
        if campaign_plan['body']['controller_sha256']!=digest(checkout/'tools/continuous_training.py'):raise ValueError('continuous controller differs')
        stage=Path(campaign_state['current_stage']);plan=wrapped(stage/'plan.json');learner=stage/'learner';publication=Path(plan['body']['publication']);parent=Path(plan['body']['initialization']['parent_run'])
        for path in (learner/'.execution.lock',learner/'league/.loop.lock',parent/'.execution.lock',publication.with_name('.'+publication.name+'.publisher.lock')):
            lock=path.open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        clock=wrapped(learner/'execution.json')['body']['clock'];league=wrapped(learner/'league/state.json');state=league['body'];published=wrapped(publication)
        if clock['inflight_since'] is not None or state['evaluation_inflight'] is not None:raise ValueError('stop the trainer gracefully before adoption')
        if not plan['body'].get('experimental') or not published['body'].get('experimental'):raise ValueError('experimental loop only')
        if published['body']['loop_contract_sha256']!=canonical_sha(state['contract']):raise ValueError('another loop owns publication')
        allowed={'champion_loop.py','champion_screen.py','learning_loop.py','policy_selection.py','parallel_gate.py'}
        for name,sha in plan['body']['tools'].items():
            if digest(Path(plan['body']['checkout'])/'tools'/name)!=sha:raise ValueError('original pinned source changed')
            if name not in allowed and digest(checkout/'tools'/name)!=sha:raise ValueError('unapproved source change: '+name)
        allocation_plan=wrapped(Path(trial)/'plan.json')['body']
        if plan['sha256']!=allocation_plan['stage_plan_sha256']:raise ValueError('stage plan changed since trial registration')
        assessment=assess(trial)
        if Path(assessment['trial']).resolve()!=Path(trial).resolve() or Path(json.loads((Path(trial)/'plan.json').read_text())['body']['allocation_stage'])!=stage:
            raise ValueError('trial belongs to another current stage')
        if state['champion']['policy_sha256']!=assessment['incumbent']['policy_sha256']:raise ValueError('champion changed; rebase and validate with a new slot')
        if any(row.get('allocation')==assessment['allocation'] for row in state.get('external_evaluations',[])):raise ValueError('allocation already adopted')
        reservation=json.loads((parent/'handoff.json').read_text())
        if reservation['destination']!=str(stage) or reservation['plan_sha256']!=plan['sha256']:raise ValueError('parent reservation differs')
        archive=stage/'selection-adoptions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');archive.mkdir(parents=True,exist_ok=False)
        for name,value in {'campaign-plan.json':campaign_plan,'campaign-state.json':campaign_state,'stage-plan.json':plan,'league-state.json':league,'publication.json':published,'parent-reservation.json':reservation}.items():write(archive/name,value)
        assessment_path=archive/'assessment.json';assessment['adopted_at']=datetime.now(timezone.utc).isoformat();write(assessment_path,assessment)
        previous=deepcopy(state['champion']);state['history'].append(previous);state['champion']=deepcopy(assessment['candidate']);state['champion_assessment']=str(assessment_path);state['prior_promotions']=state.get('prior_promotions',0)+1
        state.setdefault('external_evaluations',[]).append({'index':assessment['round_index'],'records':assessment['records'],
            'incumbent':previous,'candidate':deepcopy(state['champion']),'decision':assessment['decision'],
            'evaluation_root':assessment['evaluation_root'],'allocation':assessment['allocation'],'external':True,'promoted':True})
        pending=state.get('pending')
        if pending:
            if pending.get('status') in ('evaluating','screening'):
                if clock['last_checkpoint']['records']!=pending['target']:raise ValueError('pending gate is not at its frozen checkpoint')
                state['rounds'].append({'index':pending['index'],'records':pending['target'],'candidate':pending.get('candidate'),'incumbent':pending['champion'],
                    'decision':{'status':'complete','numeric_pass':False,'promotion_permitted':False,'experimental_promotion_permitted':False,'reason':'incumbent_changed_on_release_adoption'},'promoted':False,'learner_reinitialized':False,'buffers_flushed':False})
                state.update(round=pending['index']+1,completed_records=pending['target'],pending=None)
            else:
                pending.update(champion=deepcopy(state['champion']),history=deepcopy(state['history']),status='training')
                for key in ('candidate','screen','schedule_action'):pending.pop(key,None)
        selection=state['champion'].get('selection');head=state['champion'].get('head','average')
        state['contract']['loop_plan'].update(release_head=head,release_temperature=None if selection is None else selection['temperature'],
            release_placement_only=True if selection is None else selection['placement_only'])
        state['contract'].update(implementation_sha256=digest(checkout/'tools/champion_loop.py'),screen_sha256=IndependentScreen(ScreenPlan()).sha256,gate_workers=8,gate_looks=[500,2000,4000])
        gate_plan=GatePlan(len(state['contract']['loop_plan']['round_ends']),looks=tuple(state['contract']['gate_looks']),alpha=plan['body'].get('gate_alpha',.05),seed=plan['body'].get('gate_seed',73))
        state['contract']['gate_sha256']=ParallelGate(gate_plan,workers=8,reserve_bytes=reserve_bytes).sha256
        if state.get('adaptive_scheduler'):
            learning=max(0.,clock['observed_seconds']+clock['recovery_charge_seconds']-state['evaluation_seconds'])
            state['adaptive_scheduler']=adaptive.observe_gate(state['adaptive_scheduler'],assessment['decision'],learning,previous['policy_sha256'],promoted=True)
        league['sha256']=canonical_sha(state);plan['body'].update(checkout=str(checkout),release_selection=selection,release_head=head,gate_workers=8,gate_looks=state['contract']['gate_looks'])
        for name in allowed:plan['body']['tools'][name]=digest(checkout/'tools'/name)
        plan['sha256']=canonical_sha(plan['body']);reservation.update(previous_plan_sha256=reservation['plan_sha256'],plan_sha256=plan['sha256'])
        campaign_plan['body'].update(checkout=str(checkout),release_selection=selection,release_head=head,gate_workers=8,gate_looks=state['contract']['gate_looks']);campaign_plan['sha256']=canonical_sha(campaign_plan['body']);campaign_state['plan_sha256']=campaign_plan['sha256']
        published['body'].update(champion=deepcopy(state['champion']),completed_round=state['round'],loop_contract_sha256=canonical_sha(state['contract']),loop_state_sha256=league['sha256'],champion_assessment=str(assessment_path))
        published['sha256']=canonical_sha(published['body'])
        for path,value in ((stage/'plan.json',plan),(parent/'handoff.json',reservation),(learner/'league/state.json',league),(learner/'league/champion.json',published),(publication,published),(campaign/'plan.json',campaign_plan),(campaign/'state.json',campaign_state)):write(path,value)
        receipt={'archive':str(archive),'candidate_records':state['champion']['records'],'champion_sha256':state['champion']['policy_sha256'],
            'learner_checkpoint_records':clock['last_checkpoint']['records'],'selection':selection,'head':head,'head_win_score':assessment['decision']['head_win_score'],
            'learner_payload_changed':False,'allocated_alpha_reused':False,'training_recipe_changed':False,'gate_workers':8,'gate_looks':state['contract']['gate_looks']}
        write(archive/'receipt.json',receipt);return receipt
    finally:
        for lock in reversed(locks):lock.close()


def main():
    import torch
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('--trial',required=True);parser.add_argument('--campaign');parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    result=apply(args.campaign,args.trial) if args.apply else assess(args.trial)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()

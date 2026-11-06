"""Apply a tools-only scheduler upgrade to a stopped continuous campaign.

Retains the original plans/contracts, all checkpoints and all evaluation evidence.
Only the scheduler implementation fingerprint and pinned checkout are amended.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
from graygo.confidence import canonical_sha
from graygo.manifest import source_digest,write_report
from graygo.storage import digest,DEFAULT_RESERVE


def wrapped(path):
    value=json.loads(path.read_text())
    if value['sha256']!=canonical_sha(value['body']):raise ValueError('checksum differs: '+str(path))
    return value


def apply(campaign,*,reserve_bytes=DEFAULT_RESERVE,evaluation_schedule=False):
    campaign=Path(campaign).resolve();checkout=Path(__file__).resolve().parents[1];locks=[]
    def write(path,value):write_report(path,value,reserve_bytes=reserve_bytes)
    try:
        controller=(campaign/'.controller.lock').open('a+b');fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(controller)
        campaign_plan=wrapped(campaign/'plan.json');campaign_state=json.loads((campaign/'state.json').read_text())
        if campaign_state['plan_sha256']!=campaign_plan['sha256']:raise ValueError('campaign state/plan differs')
        if campaign_plan['body']['source_sha256']!=source_digest():raise ValueError('numerical source changed')
        if campaign_plan['body']['controller_sha256']!=digest(checkout/'tools/continuous_training.py'):
            raise ValueError('continuous controller changed; this upgrade admits only champion scheduling')
        stage=Path(campaign_state['current_stage']);plan=wrapped(stage/'plan.json');learner=stage/'learner'
        publication=Path(plan['body']['publication']);parent=Path(plan['body']['initialization']['parent_run'])
        for path in (learner/'.execution.lock',learner/'league/.loop.lock',parent/'.execution.lock',
                     publication.with_name('.'+publication.name+'.publisher.lock')):
            lock=path.open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        clock=wrapped(learner/'execution.json')['body']['clock']
        if clock['inflight_since'] is not None:raise ValueError('stop training gracefully before upgrading')
        league=wrapped(learner/'league/state.json');published=wrapped(publication)
        if league['body']['evaluation_inflight'] is not None:raise ValueError('evaluation is not closed')
        old_contract=canonical_sha(league['body']['contract'])
        if published['body']['loop_contract_sha256']!=old_contract:raise ValueError('a different loop owns publication')
        if league['body']['contract']['implementation_sha256']!=plan['body']['tools']['champion_loop.py']:
            raise ValueError('recorded scheduler identities differ')
        allowed={'champion_loop.py','champion_screen.py'} if evaluation_schedule else {'champion_loop.py'}
        for name,sha in plan['body']['tools'].items():
            if digest(Path(plan['body']['checkout'])/'tools'/name)!=sha:raise ValueError('original source changed')
            if name not in allowed and digest(checkout/'tools'/name)!=sha:
                raise ValueError('upgrade changes more than scheduling: '+name)
        reservation=json.loads((parent/'handoff.json').read_text())
        if reservation['plan_sha256']!=plan['sha256'] or reservation['destination']!=str(stage):
            raise ValueError('parent reservation differs')
        archive=stage/'controller-upgrades'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        archive.mkdir(parents=True,exist_ok=False)
        originals={'campaign-plan.json':campaign_plan,'campaign-state.json':campaign_state,
                   'stage-plan.json':plan,'league-state.json':league,'publication.json':published,
                   'parent-reservation.json':reservation}
        for name,value in originals.items():write(archive/name,value)
        old_implementation=league['body']['contract']['implementation_sha256']
        implementation=digest(checkout/'tools/champion_loop.py')
        if evaluation_schedule:
            from tools.champion_screen import IndependentScreen
            league['body']['contract']['screen_sha256']=IndependentScreen().sha256
            pending=league['body'].get('pending')
            if pending and pending.get('status') in ('evaluating','screening'):
                # Never resume an old baseline gate under a new opponent set or
                # reuse its confidence family for another candidate.
                target=pending['target']
                if clock['last_checkpoint']['records']!=target:
                    raise ValueError('unfinished evaluation is not at its frozen checkpoint')
                league['body']['rounds'].append({'index':pending['index'],'records':target,
                    'candidate':pending.get('candidate'),'incumbent':pending['champion'],
                    'decision':{'status':'complete','numeric_pass':False,
                        'promotion_permitted':False,'experimental_promotion_permitted':False,
                        'reason':'evaluation_schedule_superseded'},'promoted':False,
                    'learner_reinitialized':False,'buffers_flushed':False})
                league['body'].update(round=pending['index']+1,completed_records=target,pending=None)
        league['body']['contract']['implementation_sha256']=implementation
        league['sha256']=canonical_sha(league['body'])
        plan['body']['checkout']=str(checkout)
        for name in allowed:plan['body']['tools'][name]=digest(checkout/'tools'/name)
        plan['sha256']=canonical_sha(plan['body'])
        reservation['previous_plan_sha256']=reservation['plan_sha256'];reservation['plan_sha256']=plan['sha256']
        campaign_plan['body']['checkout']=str(checkout);campaign_plan['sha256']=canonical_sha(campaign_plan['body'])
        campaign_state['plan_sha256']=campaign_plan['sha256']
        published['body']['loop_contract_sha256']=canonical_sha(league['body']['contract'])
        published['body']['loop_state_sha256']=league['sha256'];published['sha256']=canonical_sha(published['body'])
        write(stage/'plan.json',plan);write(parent/'handoff.json',reservation)
        write(learner/'league/state.json',league);write(publication,published)
        write(learner/'league/champion.json',published)
        write(campaign/'plan.json',campaign_plan);write(campaign/'state.json',campaign_state)
        receipt={'at':datetime.now(timezone.utc).isoformat(),'archive':str(archive),
                 'source_sha256':source_digest(),'old_scheduler_sha256':old_implementation,
                 'new_scheduler_sha256':implementation,'checkpoint_records':clock['last_checkpoint']['records'],
                 'numeric_promotion_policy_changed':False,'evaluation_evidence_changed':False,
                 'checkpoint_or_optimizer_changed':False,'checkout':str(checkout),
                 'evaluation_schedule_changed':evaluation_schedule,
                 'opponents':'champion-and-frozen-model-history' if evaluation_schedule else 'unchanged'}
        write(archive/'receipt.json',receipt)
        return receipt
    finally:
        for lock in reversed(locks):lock.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    parser.add_argument('--evaluation-schedule',action='store_true');args=parser.parse_args()
    print(json.dumps(apply(args.campaign,evaluation_schedule=args.evaluation_schedule),sort_keys=True))


if __name__=='__main__':main()

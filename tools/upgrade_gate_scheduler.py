"""Apply a tools-only scheduler upgrade to a stopped continuous campaign.

Retains the original plans/contracts, all checkpoints and all evaluation evidence.
Only the scheduler implementation fingerprint and pinned checkout are amended.
"""
import argparse
import ast
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


def screen_defaults_only(before,after,name):
    """Admit changed screening defaults, without accepting another code change."""
    def normalize(path):
        tree=ast.parse(Path(path).read_text())
        if name=='champion_screen.py':
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ScreenPlan')
            for node in cls.body:
                if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id in ('games','seconds'):
                    node.value=ast.Constant(None)
        else:
            for node in ast.walk(tree):
                if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                    and node.func.attr=='add_argument' and node.args and isinstance(node.args[0],ast.Constant)
                    and node.args[0].value in ('--screen-games','--screen-seconds')):
                    for keyword in node.keywords:
                        if keyword.arg=='default':keyword.value=ast.Constant(None)
        return ast.dump(tree)
    if normalize(before)!=normalize(after):raise ValueError('screen-size upgrade changes other code: '+name)


def apply(campaign,*,reserve_bytes=DEFAULT_RESERVE,evaluation_schedule=False,adaptive_schedule=False,screen_size=False):
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
        if screen_size and (evaluation_schedule or adaptive_schedule):raise ValueError('screen-size upgrade must be applied separately')
        allowed={'champion_loop.py','champion_screen.py'} if evaluation_schedule or screen_size else {'champion_loop.py'}
        if adaptive_schedule:allowed.update(('learning_loop.py','adaptive_gate_schedule.py'))
        for name,sha in plan['body']['tools'].items():
            if digest(Path(plan['body']['checkout'])/'tools'/name)!=sha:raise ValueError('original source changed')
            if name not in allowed and digest(checkout/'tools'/name)!=sha:
                raise ValueError('upgrade changes more than scheduling: '+name)
        if screen_size:
            from tools.champion_screen import ScreenPlan
            if (ScreenPlan().games,ScreenPlan().seconds)!=(100,600.):raise ValueError('expected 100-game screening plan')
            for name in allowed:screen_defaults_only(Path(plan['body']['checkout'])/'tools'/name,checkout/'tools'/name,name)
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
        if adaptive_schedule:
            from tools import adaptive_gate_schedule as adaptive
            if plan['body']['remaining_seconds'] is not None:
                raise ValueError('adaptive scheduling requires an unlimited campaign')
            plan['body']['adaptive_gate_schedule']=True
            league['body']['contract']['loop_plan']['adaptive_schedule']=True
            league['body']['contract']['adaptive_scheduler']=adaptive.contract()
            learning_seconds=max(0.,clock['observed_seconds']+clock['recovery_charge_seconds']-league['body']['evaluation_seconds'])
            league['body'].setdefault('adaptive_scheduler',adaptive.initial_state(learning_seconds))
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
        screen_gate_preserved=False
        if screen_size:
            from tools.champion_screen import IndependentScreen
            from graygo.storage import CheckpointStore
            CheckpointStore.verify(learner/'resume'/f"{clock['last_checkpoint']['records']:012d}")
            league['body']['contract']['screen_sha256']=IndependentScreen().sha256
            pending=league['body'].get('pending')
            screen_gate_preserved=bool(pending and pending.get('status')=='evaluating')
            if pending and pending.get('status')=='screening':
                if clock['last_checkpoint']['records']!=pending['target']:raise ValueError('unfinished screen is not at its frozen checkpoint')
                league['body']['rounds'].append({'index':pending['index'],'records':pending['target'],
                    'candidate':pending.get('candidate'),'incumbent':pending['champion'],'screen':pending.get('screen'),
                    'decision':{'status':'complete','numeric_pass':False,'promotion_permitted':False,
                        'experimental_promotion_permitted':False,'reason':'screen_size_changed'},
                    'promoted':False,'learner_reinitialized':False,'buffers_flushed':False})
                league['body'].update(round=pending['index']+1,completed_records=pending['target'],pending=None)
            scheduler=league['body'].get('adaptive_scheduler')
            if scheduler is not None:
                # A noisy old 20-game estimate is not the baseline for a new
                # 100-game estimate. Keep the full-gate clock and cooldown.
                learning=max(0.,clock['observed_seconds']+clock['recovery_charge_seconds']-league['body']['evaluation_seconds'])
                scheduler.update(previous_score=None,last_gate_score=None,history_scores={},stalled_checks=0,
                    last_screen_records=None,last_signal='screen_size_changed',last_action='rescreen_100',
                    next_screen_learning_seconds=min(scheduler['next_screen_learning_seconds'],learning))
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
                 'adaptive_schedule_enabled':adaptive_schedule,'screen_size_changed':screen_size,
                 'screen_games':100 if screen_size else None,'screen_seconds':600. if screen_size else None,
                 'inflight_gate_preserved':screen_gate_preserved,
                 'opponents':'champion-and-frozen-model-history' if evaluation_schedule else 'unchanged'}
        write(archive/'receipt.json',receipt)
        return receipt
    finally:
        for lock in reversed(locks):lock.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    parser.add_argument('--evaluation-schedule',action='store_true')
    parser.add_argument('--adaptive-schedule',action='store_true');parser.add_argument('--screen-size',action='store_true');args=parser.parse_args()
    print(json.dumps(apply(args.campaign,evaluation_schedule=args.evaluation_schedule,
        adaptive_schedule=args.adaptive_schedule,screen_size=args.screen_size),sort_keys=True))


if __name__=='__main__':main()

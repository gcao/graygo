"""Append the latest training status once, including idle/budget-finished runs.

The hourly timer invokes this observer independently of the training process.
It uses JSON metadata only and never starts a trainer or imports PyTorch.
"""
import argparse
from datetime import datetime
import json
import os
import sqlite3
from pathlib import Path
from zoneinfo import ZoneInfo

# The service also supports execution by absolute script path.
try:
    from tools.training_benchmarks import benchmark_fields, stage_roots
except ModuleNotFoundError:
    from training_benchmarks import benchmark_fields, stage_roots


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def alive(pid, root):
    if not isinstance(pid, int):
        return False
    try:
        process = Path('/proc') / str(pid)
        args = (process / 'cmdline').read_bytes().split(b'\0')
        return (process / 'stat').read_text().split()[2] != 'Z' and str(root).encode() in args
    except (OSError, ValueError):
        return False


def count(value):
    return f'{value:,}' if isinstance(value, int) else '?'


def summary(project):
    receipt = read(project / 'runs/latest-training-loop.json')
    if not receipt:
        return ['state=IDLE', 'status=no_training_run']
    root = Path(receipt['run_dir'])
    status = read(root / 'status.json')
    execution = read(root / 'learner/execution.json').get('body', {})
    clock = execution.get('clock', {})
    league = read(root / 'learner/league/state.json').get('body', {})
    watch = read(root / 'watch.json')
    running = alive(receipt.get('supervisor_pid'), Path(receipt.get('process_root',root)))
    run_status = status.get('status', 'unknown')
    if run_status in ('running', 'prepared'):
        state = 'RUNNING' if running else 'SUPERVISOR_EXITED'
    else:
        state = 'FINISHING' if running else 'STOPPED'
    pending = league.get('pending') or {}
    phase = pending.get('status', 'starting') if running else 'idle'
    if running and clock.get('status') == 'initializing':phase='restoring_checkpoint'
    records = (clock.get('last_checkpoint') or {}).get('records', status.get('total_records'))
    checkpoint = watch.get('checkpoint') or {}
    if checkpoint.get('records') != records:
        checkpoint = {}
    progress=execution.get('progress') or {}
    if progress:checkpoint=progress
    counters = checkpoint.get('counters') or status.get('counters') or {}
    if not counters and isinstance(records, int):
        weights = root / 'learner/weights' / f'{records:012d}' / 'manifest.json'
        counters = read(weights).get('metadata', {}).get('counters', {})
    fields = [f'state={state} status={run_status} phase={phase}', f'run={root.name}',
              f'checkpoint_records={count(records)}',
              f"Q_updates={count(counters.get('q_updates'))} avg_updates={count(counters.get('supervised_updates'))}"]
    if receipt.get('unlimited'):fields.append('time_limit=none')
    if progress:fields.append(f"live_records={count(progress.get('records'))}")
    if records==0 and receipt.get('parent_records',0)>0:
        fields.append(f"restoring_from={count(receipt['parent_records'])}")
    if running and pending.get('target') is not None:
        fields.append(f"next_target={count(pending['target'])}")
    if checkpoint.get('mean_q_loss') is not None:
        fields.append(f"Q_loss={checkpoint['mean_q_loss']:.6f}")
    if checkpoint.get('mean_game_rounds') is not None:
        fields.append(f"mean_rounds={checkpoint['mean_game_rounds']:.1f}")
    published = read(receipt.get('publication', root / 'learner/league/champion.json'))
    champion = published.get('body', {}).get('champion', {}).get('label', '?')
    promotions = league.get('prior_promotions',0)+sum(r['promoted'] for r in league.get('rounds', []))
    fields.append(f'promotions={promotions} champion={champion}')
    fields.extend(benchmark_fields(receipt, root, progress.get('records', records)))
    if pending.get('status')=='evaluating' and pending.get('index') is not None:
        evaluation=root/'learner/league/evaluations'/f"{pending['index']:04d}"
        if (evaluation/'positive-pool-gain-v2').is_dir():evaluation=evaluation/'positive-pool-gain-v2'
        gate=read(evaluation/'gate-plan.json');looks=gate.get('family',{}).get('looks',[500,2000])
        counts={}
        for path in sorted(evaluation.glob('*/evidence.sqlite')):
            try:
                connection=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True,timeout=.1)
                try:counts[path.parent.name]=connection.execute('SELECT COUNT(*) FROM units').fetchone()[0]*2
                finally:connection.close()
            except sqlite3.Error:pass
        if counts:
            maximum=max(counts.values());target=next((n for n in looks if n>=maximum),looks[-1])
            fields.append(f"gate_candidate_records={count(pending.get('target'))}")
            fields.extend(f'gate_{label}={games}/{target}' for label,games in counts.items())
    assessment_path=league.get('champion_assessment') or published.get('body',{}).get('champion_assessment')
    if not assessment_path:
        for previous_stage in stage_roots(receipt, root):
            assessment_path=read(previous_stage/'learner/league/state.json').get('body',{}).get('champion_assessment')
            if assessment_path:break
    assessment=read(assessment_path) if assessment_path else {}
    if assessment:
        fields.append(f"promotion_policy={assessment.get('promotion_policy',assessment.get('decision',{}).get('promotion_policy','unknown'))}")
    all_rounds=[]
    for stage in reversed(stage_roots(receipt, root)):
        all_rounds.extend(read(stage/'learner/league/state.json').get('body',{}).get('rounds',[]))
    for row in reversed(all_rounds):
        decision=row.get('decision',{})
        if 'criteria' in decision:
            if (assessment.get('version')=='graygo/promotion-policy-assessment/2' and assessment.get('records')==row['records']
                    and assessment.get('decision',{}).get('numeric_pass')):
                fields.append(f"last_gate_records={count(row['records'])} promotion_reassessment_pass=True original_gate_pass={decision['numeric_pass']}")
                break
            failed=','.join(k for k,v in decision['criteria'].items() if not v) or 'none'
            fields.append(f"last_gate_records={count(row['records'])} gate_pass={decision['numeric_pass']} failed={failed}")
            break
    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--log', type=Path)
    args = parser.parse_args()
    log = args.log or args.project / 'runs/training-watch.log'
    stamp = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z')
    try:
        fields = summary(args.project)
    except (OSError, ValueError, KeyError, TypeError) as error:
        fields = ['state=OBSERVER_ERROR', f'error={repr(error)}']
    line = f'[{stamp}] HOURLY | ' + ' | '.join(fields)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        stream.write(line + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(line)


if __name__ == '__main__':
    main()

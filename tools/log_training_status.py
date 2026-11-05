"""Append the latest training status once, including idle/budget-finished runs.

The hourly timer invokes this observer independently of the training process.
It uses JSON metadata only and never starts a trainer or imports PyTorch.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo


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
    clock = read(root / 'learner/execution.json').get('body', {}).get('clock', {})
    league = read(root / 'learner/league/state.json').get('body', {})
    watch = read(root / 'watch.json')
    running = alive(receipt.get('supervisor_pid'), root)
    run_status = status.get('status', 'unknown')
    if run_status in ('running', 'prepared'):
        state = 'RUNNING' if running else 'SUPERVISOR_EXITED'
    else:
        state = 'FINISHING' if running else 'STOPPED'
    pending = league.get('pending') or {}
    phase = pending.get('status', 'starting') if running else 'idle'
    records = (clock.get('last_checkpoint') or {}).get('records', status.get('total_records'))
    checkpoint = watch.get('checkpoint') or {}
    if checkpoint.get('records') != records:
        checkpoint = {}
    counters = checkpoint.get('counters') or status.get('counters') or {}
    if not counters and isinstance(records, int):
        weights = root / 'learner/weights' / f'{records:012d}' / 'manifest.json'
        counters = read(weights).get('metadata', {}).get('counters', {})
    fields = [f'state={state} status={run_status} phase={phase}', f'run={root.name}',
              f'checkpoint_records={count(records)}',
              f"Q_updates={count(counters.get('q_updates'))} avg_updates={count(counters.get('supervised_updates'))}"]
    if running and pending.get('target') is not None:
        fields.append(f"next_target={count(pending['target'])}")
    if checkpoint.get('mean_q_loss') is not None:
        fields.append(f"Q_loss={checkpoint['mean_q_loss']:.6f}")
    if checkpoint.get('mean_game_rounds') is not None:
        fields.append(f"mean_rounds={checkpoint['mean_game_rounds']:.1f}")
    screens = sorted((root / 'learner/league/screens').glob('*/screen.json'))
    if screens:
        screen = read(screens[-1])
        index = int(screens[-1].parent.name)
        ends = league.get('contract', {}).get('loop_plan', {}).get('round_ends', [])
        if index < len(ends):
            fields.append(f'screen_records={count(ends[index])}')
        for head, report in screen.get('heads', {}).items():
            name = 'Q' if head == 'greedy_q' else head
            if report.get('status') == 'uninitialized':
                fields.append(f'{name}=uninitialized')
                continue
            if report.get('opening_pass_probability') is not None:
                fields.append(f"{name}_opening_pass={report['opening_pass_probability']:.1%}")
            for opponent, result in report.get('opponents', {}).items():
                if result.get('games', 0):
                    fields.append(f"{name}_vs_{opponent}={result['wins']}W/{result['draws']}D/{result['losses']}L"
                                  f" score={result['win_score']:.1%} n={result['games']}")
    promotions = sum(r['promoted'] for r in league.get('rounds', []))
    published = read(receipt.get('publication', root / 'learner/league/champion.json'))
    champion = published.get('body', {}).get('champion', {}).get('label', '?')
    fields.append(f'promotions={promotions} champion={champion}')
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

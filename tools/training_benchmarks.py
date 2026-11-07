"""Read-only, standard-library summaries of learned-opponent evaluation evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def stage_roots(receipt, root):
    campaign = receipt.get('campaign_dir')
    roots = list((Path(campaign) / 'stages').glob('*')) if campaign else []
    return sorted(set([root, *[path for path in roots if path.is_dir()]]), reverse=True)


def evidence_stats(path):
    """Committed color pairs only, with no writes or training-runtime imports."""
    try:
        connection = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True, timeout=.1)
        try:
            rows = connection.execute('SELECT payload FROM units ORDER BY idx').fetchall()
        finally:
            connection.close()
        utilities = [game['subject_utility'] for row in rows for game in json.loads(row[0])['subject']]
    except (OSError, ValueError, KeyError, sqlite3.Error):
        return None
    if not utilities:
        return None
    wins = sum(value > 0 for value in utilities)
    draws = sum(value == 0 for value in utilities)
    return {'wins': wins, 'draws': draws, 'losses': len(utilities)-wins-draws,
            'games': len(utilities), 'win_score': (wins+.5*draws)/len(utilities)}


def latest_benchmark(roots):
    candidates = []
    for root in roots:
        folder = root / 'learner/league'
        league = read(folder / 'state.json').get('body', {})
        ends = league.get('contract', {}).get('loop_plan', {}).get('round_ends', [])
        known = [*league.get('history', []), league.get('bootstrap', {}), league.get('champion', {})]
        labels = {descriptor.get('policy_sha256'): descriptor.get('label', '?') for descriptor in known if descriptor.get('policy_sha256')}
        screens = list((folder / 'screens').glob('*/champion-first-v2/screen.json'))
        for path in screens:
            screen = read(path)
            opponents = screen.get('heads', {}).get('average', {}).get('opponents', {})
            if not opponents.get('champion', {}).get('games'):
                continue
            candidates.append({'records': screen.get('candidate_records') or 0,
                'selection':screen.get('heads',{}).get('average',{}).get('selection'),
                'at': path.stat().st_mtime, 'source': 'screen', 'status': screen.get('status', 'unknown') if screen.get('completed_at') else 'in_progress',
                'opponents': {name: dict(row, opponent_label=labels.get(row.get('opponent_sha256'),row.get('opponent_label','?'))) for name, row in opponents.items()
                              if name == 'champion' or name.startswith('historical-')}})
        pending = league.get('pending')
        rounds = [*league.get('rounds', []), *league.get('external_evaluations', []), *([pending] if pending else [])]
        for row in rounds:
            index = row.get('index')
            if not isinstance(index, int):
                continue
            gate = Path(row['evaluation_root']) if row.get('evaluation_root') else folder / 'evaluations' / f'{index:04d}'
            if (gate / 'head-to-head-history-v3/gate-plan.json').is_file():
                gate = gate / 'head-to-head-history-v3'
            elif (gate / 'positive-pool-gain-v2/gate-plan.json').is_file():
                gate = gate / 'positive-pool-gain-v2'
            plan = read(gate / 'gate-plan.json')
            if not plan:
                continue
            opponents = {}
            modified = []
            for name in ('head', *[name for name in plan.get('pool', {}) if name.startswith('historical-')]):
                path = gate / name / 'evidence.sqlite'
                report = evidence_stats(path)
                if report is None:
                    continue
                key = 'champion' if name == 'head' else name
                incumbent = row.get('incumbent', row.get('champion', {}))
                report['opponent_label'] = incumbent.get('label', '?') if name == 'head' else labels.get(plan['pool'][name], '?')
                opponents[key] = report
                modified.extend(p.stat().st_mtime for p in (path, Path(str(path)+'-wal')) if p.exists())
            if 'champion' not in opponents:
                continue
            decision = read(gate / 'decision.json')
            candidates.append({'records': row.get('records', row.get('target', ends[index] if index < len(ends) else 0)),
                'selection':(row.get('candidate') or {}).get('selection'),
                'at': max(modified), 'source': 'gate', 'status': decision.get('status', 'in_progress'),
                'opponents': opponents})
    return max(candidates, key=lambda row: (row['records'], row['source'] == 'gate', row['at']), default=None)


def benchmark_fields(receipt, root, live_records, *, now=None):
    benchmark = latest_benchmark(stage_roots(receipt, root))
    if benchmark is None:
        return ['benchmark=not_measured_against_learned_opponents']
    current = time.time() if now is None else now
    age = max(0., current-benchmark['at'])/3600
    lag = max(0, live_records-benchmark['records']) if isinstance(live_records, int) else '?'
    fields = [f"benchmark_records={benchmark['records']:,} source={benchmark['source']} status={benchmark['status']}",
              f"benchmark_age={age:.1f}h lag_records={lag:,}" if isinstance(lag, int) else f'benchmark_age={age:.1f}h lag_records=?',
              'benchmark_measured_at=' + datetime.fromtimestamp(benchmark['at'], timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')]
    selection=benchmark.get('selection')
    fields.append('benchmark_policy=raw_average' if selection is None else
        f"benchmark_policy=selected_average temperature:{selection['temperature']} placement_only:{selection['placement_only']}")
    for name, result in benchmark['opponents'].items():
        fields.append(f"average_vs_{name}({result.get('opponent_label', '?')})="
                      f"{result['wins']}W/{result['draws']}D/{result['losses']}L"
                      f" score={result['win_score']:.1%} n={result['games']}")
    return fields

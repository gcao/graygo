"""Small, non-promoting checks of frozen Q and average policies.

These fixed color-paired samples are a training diagnostic and a cheap rejection
filter. Passing them never establishes readiness or authorizes a promotion.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
import numpy as np
from graygo.confidence import canonical_sha
from graygo.engine import BLACK, Rules, State
from graygo.evaluation import BaselinePolicy, EvaluationPlan, EvaluationRun, FrozenNeuralPolicy, seed_for
from graygo.manifest import write_report
from graygo.storage import DEFAULT_RESERVE, digest


@dataclass(frozen=True)
class ScreenPlan:
    games: int = 20
    seconds: float = 30.
    seed: int = 107

    def __post_init__(self):
        if type(self.games) is not int or self.games < 4 or self.games % 2:
            raise ValueError('screen needs an even fixed game count of at least four')
        if type(self.seconds) not in (int, float) or not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError('screen needs a finite positive time ceiling')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('screen needs an independent nonnegative seed')


def summarize(rows):
    games = [game for row in rows for game in row['subject']]
    if not games:
        return {'games': 0}
    utilities = [g['subject_utility'] for g in games]
    wins = sum(u > 0 for u in utilities)
    draws = sum(u == 0 for u in utilities)
    return {'games': len(games), 'wins': wins, 'draws': draws, 'losses': len(games)-wins-draws,
            'win_score': (wins+.5*draws)/len(games), 'mean_utility': float(np.mean(utilities)),
            'mean_rounds': float(np.mean([g['rounds'] for g in games])),
            'endings': {reason: sum(g['termination'] == reason for g in games)
                        for reason in ('mutual_pass', 'round_limit')}}


class IndependentScreen:
    def __init__(self, plan=ScreenPlan(), *, reserve_bytes=DEFAULT_RESERVE):
        self.plan, self.reserve = plan, reserve_bytes
        self.identity = {'version': 'graygo/champion-screen/1', 'plan': asdict(plan),
                         'implementation_sha256': digest(Path(__file__))}
        self.sha256 = canonical_sha(self.identity)

    def evaluate(self, archive, *, average_ready, root, time_cap_seconds, stop_requested=lambda: False):
        if not math.isfinite(time_cap_seconds) or time_cap_seconds <= 0:
            raise ValueError('screen requires a positive remaining budget')
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic()+min(time_cap_seconds, self.plan.seconds)
        result = {'version': self.identity['version'], 'screen_sha256': self.sha256, 'heads': {},
                  'status': 'complete', 'average_eligible': False, 'promotion_permitted': False,
                  'scope': 'descriptive fixed-sample screening only; no readiness or strength claim'}
        for head in ('greedy_q', 'average'):
            if head == 'average' and not average_ready:
                result['heads'][head] = {'status': 'uninitialized', 'opponents': {}}
                continue
            if stop_requested() or time.monotonic() >= deadline:
                result['status'] = 'interrupted' if stop_requested() else 'budget_limited'
                break
            subject = FrozenNeuralPolicy.from_archive(archive, representation=head, label='screen-'+head)
            rules = Rules(**subject.identity['origin']['rules'])
            distribution = subject.initial_distribution(State(rules), BLACK, fresh_episode=True)
            report = {'status': 'complete', 'opening_pass_probability': float(distribution[-1]), 'opponents': {}}
            result['heads'][head] = report
            for name in ('random', 'tactical'):
                remaining = deadline-time.monotonic()
                if stop_requested() or remaining <= 0:
                    report['status'] = result['status'] = 'interrupted' if stop_requested() else 'budget_limited'
                    break
                plan = EvaluationPlan('champion-screen', subject.label, name, rules,
                                      seed_for(self.plan.seed, head, name), (self.plan.games,), self.sha256)
                path = root/head/name
                run = EvaluationRun(plan, path, subject, BaselinePolicy(name),
                                    resume=(path/'plan.json').exists(), reserve_bytes=self.reserve)
                progress = run.run(self.plan.games, time_cap_seconds=remaining, stop_requested=stop_requested)
                report['opponents'][name] = dict(summarize(run.records()), status=progress['status'],
                                                plan_sha256=run.sha256)
                if progress['status'] != 'complete':
                    report['status'] = result['status'] = progress['status']
                    break
            if result['status'] != 'complete':
                break
        average = result['heads'].get('average', {})
        opponents = average.get('opponents', {})
        result['average_eligible'] = (result['status'] == 'complete' and average.get('status') == 'complete'
            and set(opponents) == {'random', 'tactical'}
            and all(r['games'] == self.plan.games and r['mean_utility'] > 0 for r in opponents.values()))
        write_report(root/'screen.json', result, reserve_bytes=self.reserve)
        return result

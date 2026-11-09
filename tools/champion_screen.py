"""Non-promoting screens against frozen champion and historical models."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import time
import numpy as np
from graygo.confidence import canonical_sha
from graygo.engine import BLACK, Rules, State
from graygo.evaluation import EvaluationPlan, EvaluationRun, FrozenNeuralPolicy, seed_for
from graygo.manifest import write_report
from graygo.storage import DEFAULT_RESERVE, digest


@dataclass(frozen=True)
class ScreenPlan:
    games: int = 100
    seconds: float = 600.
    seed: int = 107
    refresh_seconds: float = 3600.

    def __post_init__(self):
        if type(self.games) is not int or self.games < 4 or self.games % 2:
            raise ValueError('screen needs an even fixed game count of at least four')
        if type(self.seconds) not in (int, float) or not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError('screen needs a finite positive time ceiling')
        if not math.isfinite(self.refresh_seconds) or self.refresh_seconds <= 0:
            raise ValueError('positive benchmark refresh interval required')
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
        self.identity = {'version': 'graygo/champion-screen/2', 'plan': asdict(plan),
                         'implementation_sha256': digest(Path(__file__))}
        self.sha256 = canonical_sha(self.identity)

    def evaluate(self, archive, *, average_ready, root, time_cap_seconds, stop_requested=lambda: False,
                 reference_opponents=None, round_index=0, candidate_records=None, subject_policy=None):
        if not math.isfinite(time_cap_seconds) or time_cap_seconds <= 0:
            raise ValueError('screen requires a positive remaining budget')
        if type(round_index) is not int or round_index < 0:
            raise ValueError('nonnegative candidate index required')
        references = dict(reference_opponents or {})
        if references and ('champion' not in references or any(
                name != 'champion' and not name.startswith('historical-') for name in references)):
            raise ValueError('screens accept champion and historical references only')
        if average_ready and not references:
            raise ValueError('an initialized average needs a champion reference')
        # A new namespace leaves every v1 report and its evidence untouched.
        root = Path(root) / 'champion-first-v2'
        root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic()+min(time_cap_seconds, self.plan.seconds)
        result = {'version': self.identity['version'], 'screen_sha256': self.sha256, 'heads': {},
                  'status': 'complete', 'average_eligible': False, 'promotion_permitted': False,
                  'candidate_records': candidate_records, 'round_index': round_index,
                  'started_at': datetime.now(timezone.utc).isoformat(),
                  'scope': 'descriptive fixed-sample screening only; no readiness or strength claim'}
        # The deployed average policy gets the useful matchups first. Q remains
        # diagnostic and reports its opening pass probability without games.
        for head in ('average', 'greedy_q'):
            if head == 'average' and not average_ready:
                result['heads'][head] = {'status': 'uninitialized', 'opponents': {}}
                continue
            if stop_requested() or time.monotonic() >= deadline:
                result['status'] = 'interrupted' if stop_requested() else 'budget_limited'
                break
            subject = subject_policy if head=='average' and subject_policy is not None else FrozenNeuralPolicy.from_archive(archive, representation=head, label='screen-'+head)
            rules = Rules(**subject.identity['origin']['rules'])
            distribution = subject.initial_distribution(State(rules), BLACK, fresh_episode=True)
            report = {'status': 'complete', 'opening_pass_probability': float(distribution[-1]), 'opponents': {},
                      'policy_sha256':subject.sha256,'selection':subject.identity.get('selection'),
                      'representation':subject.identity['kind']}
            result['heads'][head] = report
            opponents = dict(references) if head == 'average' else {}
            for name, opponent in opponents.items():
                remaining = deadline-time.monotonic()
                if stop_requested() or remaining <= 0:
                    report['status'] = result['status'] = 'interrupted' if stop_requested() else 'budget_limited'
                    break
                plan = EvaluationPlan('champion-screen', subject.label, name, rules,
                                      seed_for(self.plan.seed, round_index, head, name), (self.plan.games,), self.sha256)
                path = root/head/name
                run = EvaluationRun(plan, path, subject, opponent,
                                    resume=(path/'plan.json').exists(), reserve_bytes=self.reserve)
                progress = run.run(self.plan.games, time_cap_seconds=remaining, stop_requested=stop_requested)
                report['opponents'][name] = dict(summarize(run.records()), status=progress['status'],
                    plan_sha256=run.sha256, opponent_sha256=opponent.sha256,
                    opponent_label=getattr(opponent, 'label', getattr(opponent, 'name', name)),
                    measured_at=datetime.now(timezone.utc).isoformat())
                write_report(root/'screen.json', result, reserve_bytes=self.reserve)
                if progress['status'] != 'complete':
                    report['status'] = result['status'] = progress['status']
                    break
            if result['status'] != 'complete':
                break
        average = result['heads'].get('average', {})
        opponents = average.get('opponents', {})
        result['average_eligible'] = (result['status'] == 'complete' and average.get('status') == 'complete'
            and set(opponents) == set(references) and bool(references)
            and all(row['games'] == self.plan.games for row in opponents.values())
            and opponents['champion']['mean_utility'] > 0)
        result['completed_at'] = datetime.now(timezone.utc).isoformat()
        write_report(root/'screen.json', result, reserve_bytes=self.reserve)
        return result

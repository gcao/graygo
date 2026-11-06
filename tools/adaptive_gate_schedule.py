"""Persisted evaluation timing heuristics; never a promotion decision."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path


@dataclass(frozen=True)
class AdaptiveGatePlan:
    initial_seconds: float = 3600.
    minimum_seconds: float = 1800.
    maximum_seconds: float = 14400.
    monitor_seconds: float = 3600.
    strong_score: float = .70
    progress_delta: float = .05
    history_floor: float = .45
    history_drop: float = .15
    stalled_checks: int = 2

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (self.minimum_seconds,self.initial_seconds,self.maximum_seconds,self.monitor_seconds)):
            raise ValueError('finite adaptive intervals required')
        if not (0 < self.minimum_seconds <= self.initial_seconds <= self.maximum_seconds
                and 0 < self.monitor_seconds <= self.maximum_seconds):
            raise ValueError('ordered positive adaptive intervals required')
        if not (.5 < self.strong_score <= 1 and 0 < self.progress_delta < .5
                and 0 <= self.history_floor <= .5 and 0 < self.history_drop < 1
                and type(self.stalled_checks) is int and self.stalled_checks >= 2):
            raise ValueError('invalid adaptive screening thresholds')


PLAN = AdaptiveGatePlan()


def contract(plan=PLAN):
    return {'version': 'graygo/adaptive-gate-schedule/1', 'plan': asdict(plan),
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def initial_state(now, plan=PLAN):
    return {'version': 'graygo/adaptive-gate-state/1', 'started_learning_seconds': now,
            'interval_seconds': plan.initial_seconds, 'next_screen_learning_seconds': now,
            'next_gate_learning_seconds': now+plan.initial_seconds,
            'incumbent_sha256': None, 'previous_score': None, 'history_scores': {},
            'stalled_checks': 0, 'gate_count': 0, 'last_gate_learning_seconds': None,
            'last_gate_incumbent_sha256': None, 'last_gate_score': None,
            'last_signal': 'new_schedule', 'last_action': 'screen_at_next_checkpoint'}


def screen_due(state, now):
    return now >= state['next_screen_learning_seconds']


def _score(row):
    score = row.get('win_score')
    if (row.get('status', 'complete') != 'complete' or row.get('games', 0) < 4
            or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError('complete finite color-paired screen result required')
    return float(score)


def observe_screen(state, screen, now, incumbent_sha256, plan=PLAN):
    result = deepcopy(state)
    average = screen.get('heads', {}).get('average', {})
    opponents = average.get('opponents', {})
    if screen.get('status') != 'complete' or 'champion' not in opponents:
        result.update(last_signal='incomplete_screen', last_action='train_then_rescreen',
                      next_screen_learning_seconds=now+plan.minimum_seconds)
        return result, {'run_gate': False, 'reason': 'adaptive_screen_incomplete'}
    score = _score(opponents['champion'])
    if opponents['champion'].get('opponent_sha256', incumbent_sha256) != incumbent_sha256:
        raise ValueError('adaptive screen was measured against another champion')
    if result['incumbent_sha256'] != incumbent_sha256:
        result.update(incumbent_sha256=incumbent_sha256, previous_score=None,
                      stalled_checks=0, interval_seconds=plan.initial_seconds)
    previous = result['previous_score']
    history_stable = True
    history = {}
    for name, row in opponents.items():
        if name == 'champion':
            continue
        if not name.startswith('historical-'):
            raise ValueError('adaptive schedule accepts learned opponents only')
        value = _score(row)
        identity = row.get('opponent_sha256', name)
        old = result['history_scores'].get(identity)
        history_stable &= value >= plan.history_floor and (old is None or value >= old-plan.history_drop)
        history[identity] = value
    result['history_scores'].update(history)
    result['previous_score'] = score
    result['last_screen_records'] = screen.get('candidate_records')
    strong = score >= plan.strong_score and history_stable
    fresh_strength = (result['last_gate_incumbent_sha256'] != incumbent_sha256
                      or result['last_gate_score'] is None
                      or score > result['last_gate_score']+plan.progress_delta)
    if strong and fresh_strength:
        result['interval_seconds'] = max(plan.minimum_seconds, result['interval_seconds']/2)
        result.update(stalled_checks=0, last_signal='strong_progress')
    elif not history_stable:
        result['stalled_checks'] += 1
        result['last_signal'] = 'history_regression'
    elif previous is None or score > previous+plan.progress_delta:
        result.update(stalled_checks=0, last_signal='improving' if score > .5 else 'learning')
        result['interval_seconds'] = min(result['interval_seconds'], plan.initial_seconds)
        result['next_gate_learning_seconds'] = min(result['next_gate_learning_seconds'], now+plan.initial_seconds)
    else:
        result['stalled_checks'] += 1
        result['last_signal'] = 'flat_or_behind'
    if result['stalled_checks'] >= plan.stalled_checks:
        result['interval_seconds'] = min(plan.maximum_seconds, result['interval_seconds']*2)
        result['next_gate_learning_seconds'] = now+result['interval_seconds']
        result['stalled_checks'] = 0
    last_gate = result['last_gate_learning_seconds']
    origin = last_gate if last_gate is not None else result['started_learning_seconds']
    cooldown_cleared = last_gate is None or now-last_gate >= plan.minimum_seconds
    timed_due = now >= result['next_gate_learning_seconds'] or now-origin >= plan.maximum_seconds
    run_gate = bool(screen.get('average_eligible') and history_stable and score > .5
                    and cooldown_cleared and ((strong and fresh_strength) or timed_due))
    result['next_screen_learning_seconds'] = now+min(plan.monitor_seconds, result['interval_seconds'])
    result['last_action'] = 'full_gate' if run_gate else 'train_more'
    return result, {'run_gate': run_gate,
                    'reason': 'adaptive_gate_ready' if run_gate else 'adaptive_train_more',
                    'signal': result['last_signal'], 'interval_seconds': result['interval_seconds']}


def observe_gate(state, decision, now, incumbent_sha256, *, promoted=False, plan=PLAN):
    if decision.get('status') != 'complete':
        raise ValueError('unfinished gates retain their frozen candidate and schedule')
    result = deepcopy(state)
    result['gate_count'] += 1
    result.update(last_gate_learning_seconds=now, last_gate_incumbent_sha256=incumbent_sha256,
                  last_gate_score=result['previous_score'], stalled_checks=0)
    if promoted:
        result.update(interval_seconds=plan.minimum_seconds, incumbent_sha256=None, previous_score=None,
                      last_gate_score=None, last_signal='promoted')
    else:
        result['interval_seconds'] = min(plan.maximum_seconds,
                                        max(plan.initial_seconds, result['interval_seconds']*2))
        result['last_signal'] = 'gate_not_passed'
    result['next_gate_learning_seconds'] = now+result['interval_seconds']
    result['next_screen_learning_seconds'] = now+min(plan.monitor_seconds, result['interval_seconds'])
    result['last_action'] = 'train_more'
    return result

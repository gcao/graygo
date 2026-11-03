"""Execute only the thirteen explicitly approved original table controls.

Imports stay lightweight until approval is checked. The independent watchdog
uses this same file with Python -S and survives a launcher/worker-parent exit.
"""
import argparse
from contextlib import contextmanager
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

VERSION = 'graygo/fixed-control-campaign/1'
DECISION_VERSION = 'graygo/control-completion-owner-decision/1'
CONTROLLER_RESERVE = Decimal(900)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    created = False
    try:
        with temporary.open('x') as stream:
            created = True
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if created:
            try:temporary.unlink()
            except FileNotFoundError:pass
        raise
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def boot_time():
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def load_campaign(path):
    wrapper = load(path); body = wrapper.get('body')
    require(isinstance(body, dict) and body.get('version') == VERSION and wrapper.get('sha256') == canonical(body),
            'control ledger checksum/version changed; preserve it for review')
    return body


def save_campaign(path, state):
    write(path, {'body': state, 'sha256': canonical(state)})


def process_identity(pid):
    try:
        fields = Path('/proc/' + str(pid) + '/stat').read_text().rsplit(')', 1)[1].split()
        command = Path('/proc/' + str(pid) + '/cmdline').read_bytes()
    except FileNotFoundError:
        return None
    if fields[0] in ('Z', 'X'):
        return None
    return {'pid': pid, 'start_ticks': fields[19], 'process_group': int(fields[2]),
            'cmdline_sha256': hashlib.sha256(command).hexdigest(),
            'birth_boottime_seconds': int(fields[19]) / os.sysconf('SC_CLK_TCK')}


def decision_limit(proposal, decision):
    # This check precedes Torch imports, campaign locks, directories and models.
    require(isinstance(decision, dict) and decision.get('version') == DECISION_VERSION
            and decision.get('approved') is True, 'explicit matching owner approval is required before starting table controls')
    text = decision.get('decision_record')
    require(isinstance(text, str) and text.strip()
            and hashlib.sha256(text.encode()).hexdigest() == decision.get('decision_record_sha256'),
            'retain the actual owner answer and its exact fingerprint')
    require(decision.get('proposal_sha256') == canonical(proposal)
            and decision.get('control_ceiling_seconds') == proposal['proposed_control_ceiling_seconds']
            and decision.get('case_ids') == [case['id'] for case in proposal['cases']],
            'owner decision changed the exact proposal, ceiling or thirteen-case scope')
    from tools.nfsp_control_budget import approved_control_limit
    return approved_control_limit(proposal, decision)


def validate_cases(proposal):
    from graygo.task_training import NFSPTaskConfig
    expected = {('primary', task, seed, 4) for task in ('branching', 'gray-pure', 'gray-mixed') for seed in (0, 1, 2)}
    expected |= {(role, 'delayed', seed, reuse) for role, reuse in (('primary', 4), ('comparison', 1)) for seed in (1, 2)}
    require(len(proposal['cases']) == 13 and proposal['case_count'] == 13, 'all thirteen original controls must remain selected together')
    observed = set(); ids = set(); configs = {}
    for case in proposal['cases']:
        role, task, seed, reuse = case['cohort_role'], case['task'], case['seed'], case['q_reuse']
        key = (role, task, seed, reuse)
        require(type(seed) is int and type(reuse) is int and key in expected and key not in observed,
                'unknown, duplicate, filtered or replacement control case')
        identifier = role + '/' + task + '/tabular/' + str(seed) + '/R' + str(reuse)
        require(case['id'] == identifier and identifier not in ids and case['learner'] == 'tabular', 'original table case identity differs')
        cap = Decimal(case['complete_process_ceiling_seconds'])
        require(cap > 0 and cap == int(cap), 'complete integer process cap required')
        config = NFSPTaskConfig(**case['config'])
        required = NFSPTaskConfig(task=task, learner='tabular', seed=seed, q_reuse=reuse,
            time_cap_seconds=int(cap), budget_profile_sha256=case['raw_budget_sha256'])
        require(config.formal_protocol() and config.plan() == required.plan(), 'full original table recipe or funded ceiling changed')
        configs[identifier] = config; observed.add(key); ids.add(identifier)
    require(observed == expected and sum(int(case['complete_process_ceiling_seconds']) for case in proposal['cases']) == 213300
            and proposal['additional_control_reservation_seconds'] == '213300', 'complete original aggregate cap inventory required')
    return configs


def bundle(repo, allocation_path, expected_sha, *, fresh=True):
    repo = Path(repo).resolve(); allocation_path = Path(allocation_path).resolve()
    require(sha(allocation_path) == expected_sha, 'committed control allocation changed')
    a = load(allocation_path)
    campaign_root = (repo / a['campaign_root']).resolve()
    require(campaign_root.is_relative_to((repo / 'runs').resolve()), 'campaign must remain inside project runs')
    require(sha(__file__) == a['runner_sha256'] and a['max_parallel_cases'] == 1
            and a['controller_reserve_seconds'] == str(CONTROLLER_RESERVE), 'frozen serial controller or overhead reservation changed')
    proposal_path = repo / a['proposal']; proposal = load(proposal_path)
    require(sha(proposal_path) == a['proposal_file_sha256'] and canonical(proposal) == a['proposal_sha256'], 'original scope proposal changed')
    configs = validate_cases(proposal)
    require(set(a['cases']) == set(configs), 'allocation must bind all thirteen original cases')
    roots = []
    for case in proposal['cases']:
        binding = a['cases'][case['id']]; root = (repo / binding['root']).resolve()
        require(root.is_relative_to(campaign_root / 'cases') and root not in roots
                and all(root not in p.parents and p not in root.parents for p in roots), 'distinct run-owned control roots required')
        roots.append(root)
        require(binding['source_sha256'] == case['source_sha256'] and binding['source_package'] == case['source_package_path']
                and sha(repo / case['raw_budget']) == case['raw_budget_sha256'], 'original control source/price binding changed')
        require(sha(repo / binding['plan']) == binding['plan_file_sha256'], 'original source-specific control plan changed')
    usage = load(repo / a['prior_usage'])
    require(sha(repo / a['prior_usage']) == a['prior_usage_sha256'], 'original control/accounting inventory changed')
    from tools.nfsp_resources import Usage, measured_usage
    review = measured_usage([Usage(**row) for row in usage['inventory']['measurements']])
    require(review['cpu_categories']['control']['committed_process_seconds'] == proposal['current_control_commitment_seconds'],
            'new controls require a fresh allocation after another control commitment')
    if fresh:
        from tools.nfsp_control_budget import build_proposal
        from tools.a2_source_cohort import read_registry
        from tools.a2_cohort import NFSPA2Plan
        from tools.nfsp_delayed_cpu_prices import runtime_identity
        require(canonical(build_proposal(repo, repo / a['price_evidence'])) == canonical(proposal), 'fresh direct source-bound control prices differ')
        require(runtime_identity() == proposal['runtime_identity'], 'live numerical/host/runtime identity differs')
        require(sha(repo / a['source_registry']) == a['source_registry_sha256'], 'source registry changed')
        read_registry(repo / a['source_registry'], repo=repo, current=True)
        for case in proposal['cases']:
            binding = a['cases'][case['id']]
            plan = NFSPA2Plan.from_dict(load(repo / binding['plan']))
            require(plan.learner_source_sha256 == case['source_sha256'] and plan.multiround_reuse == case['q_reuse'],
                    'keep each control under its original source/reuse plan')
    return a, proposal, configs, usage


def preview(repo, allocation_path, expected_sha, decision=None):
    a, proposal, configs, usage = bundle(repo, allocation_path, expected_sha, fresh=True)
    limit = Decimal(43200) if decision is None else decision_limit(proposal, decision)
    projected = Decimal(proposal['current_control_commitment_seconds']) + Decimal(213300) + CONTROLLER_RESERVE
    return {'version': VERSION, 'status': 'owner_approval_missing' if decision is None else 'exact_controls_ready_within_selected_ceiling',
        'case_count': len(configs), 'primary_case_count': 11, 'comparison_case_count': 2,
        'case_process_reservation_seconds': '213300', 'controller_reserve_seconds': str(CONTROLLER_RESERVE),
        'projected_control_commitment_seconds': str(projected), 'selected_control_ceiling_seconds': str(limit),
        'within_selected_ceiling': projected <= limit, 'default_control_ceiling_seconds': '43200',
        'owner_approved': decision is not None, 'execution_authorized': False, 'a2_complete': False,
        'scope': 'Read-only exact thirteen-case preview. The default remains twelve aggregate CPU process-hours; '
                 'an explicit proposal-bound owner decision is required before campaign initialization. No case '
                 'or source is replaced, and the two R1 comparisons stay outside primary R4 slots.'}


@contextmanager
def campaign_lock(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with (root / '.campaign.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def claim_case(repo, a, proposal, decision, decision_sha, case_id):
    limit = decision_limit(proposal, decision)
    require(case_id in a['cases'], 'case is outside the exact approved scope')
    root = (Path(repo) / a['campaign_root']).resolve()
    projected = Decimal(proposal['current_control_commitment_seconds']) + Decimal(213300) + CONTROLLER_RESERVE
    require(projected <= limit, 'all case caps plus bounded controller overhead must fit the selected ceiling')
    identity = process_identity(os.getpid())
    require(identity is not None and identity['process_group'] == os.getpid(), 'launch each worker in its own new process session')
    global_root = Path(repo) / 'runs' / '.control-completion-registry'
    with campaign_lock(global_root), campaign_lock(root):
        registry_path = global_root / 'allocation.json'
        registry = {'version': VERSION, 'campaign_root': str(root), 'allocation_sha256': a['allocation_sha256'],
                    'owner_decision_file_sha256': decision_sha, 'proposal_sha256': canonical(proposal)}
        if registry_path.exists():
            require(load(registry_path) == registry, 'this proposal already owns its original campaign; no duplicate budget/root allocation')
        else:
            write(registry_path, registry)
        path = root / 'campaign.json'
        if path.exists():
            state = load_campaign(path)
            require(state['allocation_sha256'] == a['allocation_sha256'] and state['owner_decision_file_sha256'] == decision_sha,
                    'campaign allocation/decision cannot change or purchase a new budget')
        else:
            require(all(not (Path(repo) / entry['root']).exists() for entry in a['cases'].values()), 'an original control root exists; preserve its handle and data')
            state = {'version': VERSION, 'allocation_sha256': a['allocation_sha256'], 'owner_decision_file_sha256': decision_sha,
                'proposal_sha256': canonical(proposal), 'control_ceiling_seconds': str(limit),
                'base_control_commitment_seconds': proposal['current_control_commitment_seconds'],
                'controller_reservation_seconds': str(CONTROLLER_RESERVE), 'cases': {case['id']: {'status': 'reserved',
                    'complete_process_ceiling_seconds': case['complete_process_ceiling_seconds']} for case in proposal['cases']},
                'a2_complete': False, 'scope': 'Exact original controls; reserve every complete worker cap before initialization.'}
        require(set(state['cases']) == set(a['cases']) and state['proposal_sha256'] == canonical(proposal)
                and state['control_ceiling_seconds'] == str(limit)
                and state['base_control_commitment_seconds'] == proposal['current_control_commitment_seconds']
                and state['controller_reservation_seconds'] == str(CONTROLLER_RESERVE)
                and all(state['cases'][c['id']]['complete_process_ceiling_seconds'] == c['complete_process_ceiling_seconds'] for c in proposal['cases']),
                'original complete control ledger scope or caps changed')
        require(state['cases'][case_id]['status'] == 'reserved', 'original control was already claimed; no implicit resume or replacement')
        require(not state.get('budget_violations'), 'preserve the original budget violation; further work needs explicit review')
        require(not any(row['status'] in ('claimed', 'worker_terminal_pending_guard') for row in state['cases'].values()),
                'serial campaign still owns an original worker/guard; inspect that same handle')
        require(not (Path(repo) / a['cases'][case_id]['root']).exists(), 'original case root exists; never overwrite or implicitly resume it')
        prior_guard_seconds = sum(Decimal(str(row.get('guard_active_seconds', 0))) for row in state['cases'].values())
        require(prior_guard_seconds < CONTROLLER_RESERVE, 'the reserved controller overhead is exhausted')
        state['cases'][case_id].update(status='claimed', original_worker=identity,
                                      guard_active_budget_seconds=str(CONTROLLER_RESERVE - prior_guard_seconds))
        save_campaign(path, state)
    return root, identity


def guard(identity, cap_seconds, output, *, graceful_headroom=30., hard_headroom=1., interval=.1, max_active_seconds=900.):
    require(identity['process_group'] == identity['pid'], 'watchdog requires an original session-owned worker group')
    require(0 < hard_headroom < graceful_headroom < cap_seconds and interval > 0, 'bounded graceful/hard process deadlines required')
    require(max_active_seconds > 0, 'positive remaining controller overhead required')
    require(identity['birth_boottime_seconds'] == int(identity['start_ticks']) / os.sysconf('SC_CLK_TCK'),
            'deadline must start at the actual kernel process birth, not a reset main-function clock')
    start = boot_time(); active = 0.; last_save = 0.; signaled = False; killed = False
    state = {'version': VERSION, 'status': 'monitoring_original_worker', 'original_guard_pid': os.getpid(),
             'original_worker': identity, 'cap_seconds': cap_seconds, 'graceful_headroom_seconds': graceful_headroom,
             'hard_headroom_seconds': hard_headroom, 'new_training_records': 0}
    while True:
        tick = time.monotonic(); now = boot_time(); current = process_identity(identity['pid'])
        elapsed = now - identity['birth_boottime_seconds']
        if current is None:
            state.update(status='original_worker_terminal', observed_process_elapsed_upper_seconds=elapsed,
                         hard_budget_stop=killed, graceful_budget_signal=signaled)
        elif any(current[key] != identity[key] for key in ('pid', 'start_ticks', 'cmdline_sha256', 'process_group', 'birth_boottime_seconds')):
            state.update(status='original_identity_changed_no_signal', observed_process_elapsed_upper_seconds=elapsed)
        else:
            try:
                if (elapsed >= cap_seconds - graceful_headroom or active >= max(0., max_active_seconds - 5)) and not signaled:
                    os.killpg(identity['pid'], signal.SIGUSR1); signaled = True
                if (elapsed >= cap_seconds - hard_headroom or active >= max(0., max_active_seconds - 1)) and not killed:
                    os.killpg(identity['pid'], signal.SIGKILL); killed = True
            except ProcessLookupError:
                # The original exited between observation and signalling.
                pass
        active += time.monotonic() - tick
        if state['status'] != 'monitoring_original_worker' or now - last_save >= 10:
            tick = time.monotonic(); state.update(active_monitor_seconds=active, guard_wall_seconds=now - start)
            try:
                write(output, state)
            except OSError as error:
                # Lost logging must never disable the independent deadline.
                state['last_logging_error'] = repr(error)
            active += time.monotonic() - tick; last_save = now
        if state['status'] != 'monitoring_original_worker':
            return state
        time.sleep(interval)


def settle_case(repo, a, case_id, decision_sha):
    root = Path(repo) / a['campaign_root']; output = root / 'guards' / (canonical(case_id) + '.json')
    require(output.exists(), 'wait for the original guard; absence is not permission to restart')
    observed = load(output)
    require(observed['status'] == 'original_worker_terminal', 'the exact original guard has not established terminal state')
    with campaign_lock(root):
        path = root / 'campaign.json'; state = load_campaign(path); row = state['cases'][case_id]
        require(state['allocation_sha256'] == a['allocation_sha256'] and state['owner_decision_file_sha256'] == decision_sha,
                'original campaign allocation or owner decision changed')
        require(observed['original_worker'] == row['original_worker'], 'terminal guard belongs to another original worker')
        require(row['status'] in ('claimed', 'worker_terminal_pending_guard'), 'original control was already settled')
        row.update(status='budget_stopped_unreviewed' if observed['hard_budget_stop'] else 'terminal_unreviewed',
                   measured_complete_process_upper_seconds=observed['observed_process_elapsed_upper_seconds'],
                   guard_active_seconds=observed['active_monitor_seconds'], guard_sha256=sha(output))
        violations = []
        if Decimal(str(row['measured_complete_process_upper_seconds'])) > Decimal(row['complete_process_ceiling_seconds']):
            violations.append({'case_id': case_id, 'kind': 'whole_process_cap_exceeded', 'actual_seconds': row['measured_complete_process_upper_seconds']})
        if sum(Decimal(str(r.get('guard_active_seconds', 0))) for r in state['cases'].values()) > CONTROLLER_RESERVE:
            violations.append({'case_id': case_id, 'kind': 'controller_overhead_reserve_exceeded'})
        state.setdefault('budget_violations', []).extend(violations)
        save_campaign(path, state)
        require(not violations, 'original budget overrun was retained; stop further control work and review it')
    return state


def run_case(repo, allocation_path, expected_sha, decision_path, decision_sha, case_id):
    # Approval precedes bundle verification, locks, guards and model creation.
    repo = Path(repo).resolve(); allocation_path = Path(allocation_path).resolve(); a0 = load(allocation_path)
    require(sha(allocation_path) == expected_sha and sha(decision_path) == decision_sha, 'original allocation/owner-decision file changed')
    decision = load(decision_path); decision_limit(load(repo / a0['proposal']), decision)
    require(case_id in a0['cases'], 'unknown original control case')
    a, proposal, configs, usage = bundle(repo, allocation_path, expected_sha, fresh=False)
    a['allocation_sha256'] = expected_sha
    case = next(case for case in proposal['cases'] if case['id'] == case_id)
    from graygo.manifest import source_digest, write_report
    from graygo.task_training import NFSPTaskTrainer
    from tools.nfsp_delayed_cpu_prices import runtime_identity, IDENTITY_KEYS
    from tools.run_nfsp_cpu_delayed_case import fresh_case_directory, retain_initialization
    require(source_digest() == case['source_sha256'], 'worker must import its exact original frozen source')
    require(runtime_identity() == proposal['runtime_identity'], 'worker host/numerical/runtime differs from its price')
    campaign, identity = claim_case(repo, a, proposal, decision, decision_sha, case_id)
    guard_budget = load_campaign(campaign / 'campaign.json')['cases'][case_id]['guard_active_budget_seconds']
    guard_output = campaign / 'guards' / (canonical(case_id) + '.json')
    guard_output.parent.mkdir(parents=True, exist_ok=True)
    identity_path = campaign / 'guards' / (canonical(case_id) + '-identity.json'); write(identity_path, identity)
    stopped = []
    for name in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(name, lambda s, f: stopped.append('budget_limited' if s == signal.SIGUSR1 else 'interrupted'))
    command = [sys.executable, '-S', str(Path(__file__).resolve()), 'guard', '--identity', str(identity_path),
               '--cap', case['complete_process_ceiling_seconds'], '--output', str(guard_output), '--max-active-seconds', guard_budget]
    with (campaign / 'guards' / (canonical(case_id) + '.log')).open('xb') as log:
        watchdog = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 10
    while not guard_output.exists():
        require(watchdog.poll() is None and time.monotonic() < deadline, 'original watchdog failed to become ready; preserve this claim')
        time.sleep(.05)
    require(load(guard_output)['original_worker'] == identity and not stopped, 'exact watchdog/worker admission failed')
    # The independent deadline already covers expensive source/profile checks.
    _, fresh_proposal, _, _ = bundle(repo, allocation_path, expected_sha, fresh=True)
    require(canonical(fresh_proposal) == canonical(proposal), 'source/price proposal changed after the original claim')
    binding = a['cases'][case_id]; config = configs[case_id]; root = repo / binding['root']
    with fresh_case_directory(root):
        require(not stopped, 'stopped before original control initialization')
        root.mkdir(); write_report(root / 'admission.json', {'version': VERSION, 'case_id': case_id,
            'allocation_sha256': expected_sha, 'owner_decision_file_sha256': decision_sha,
            'source_sha256': source_digest(), 'guard_original_pid': watchdog.pid,
            'whole_process_cap_seconds': case['complete_process_ceiling_seconds'],
            'cohort_role': case['cohort_role'], 'primary_R4_slots_fillable': case['cohort_role'] == 'primary'})
        tick = time.monotonic(); trainer = NFSPTaskTrainer(config, root); setup = time.monotonic() - tick
        require(trainer.manifest['source_sha256'] == case['source_sha256'] and trainer.manifest['config'] == case['config']
                and {k: trainer.manifest[k] for k in IDENTITY_KEYS} == proposal['runtime_identity'], 'emitted control manifest differs')
        import torch
        require(torch.count_nonzero(trainer.backend.q.values).item() == 0
                and torch.equal(trainer.backend.q.values, trainer.backend.target.values)
                and not trainer.backend.q_optimizer.state and trainer.backend.average_optimizer is None,
                'fresh unbiased table values, equal target and empty optimizer moments required')
        initial = retain_initialization(trainer, root)
        write_report(root / 'case-state.json', {'status': 'running', 'case_id': case_id, 'original_pid': os.getpid(),
            'source_sha256': source_digest(), 'retained_initial_manifest_sha256': initial['retained_initial_checkpoint_manifest_sha256'],
            'constructor_seconds': setup, 'cohort_role': case['cohort_role']})
        result = trainer.run(stop_requested=lambda: stopped[0] if stopped else False)
        write_report(root / 'case-state.json', {'status': result['status'], 'records': result['records'], 'case_id': case_id,
            'original_pid': os.getpid(), 'source_sha256': source_digest(), 'cohort_role': case['cohort_role'], 'a2_complete': False})
    with campaign_lock(campaign):
        state = load_campaign(campaign / 'campaign.json'); state['cases'][case_id].update(status='worker_terminal_pending_guard',
            result_status=result['status'], records=result['records'], result_sha256=sha(root / 'result.json'))
        save_campaign(campaign / 'campaign.json', state)
    print(json.dumps({'case_id': case_id, 'status': result['status'], 'records': result['records'], 'a2_complete': False}), flush=True)


def run_serial(repo, allocation_path, expected_sha, decision_path, decision_sha, continue_unstarted=False):
    repo = Path(repo).resolve(); allocation_path = Path(allocation_path).resolve(); a0 = load(allocation_path)
    require(sha(allocation_path) == expected_sha and sha(decision_path) == decision_sha, 'original allocation/owner decision changed')
    decision = load(decision_path); decision_limit(load(repo / a0['proposal']), decision)
    a, proposal, _, _ = bundle(repo, allocation_path, expected_sha, fresh=True); a['allocation_sha256'] = expected_sha
    campaign = repo / a['campaign_root']; state_path = campaign / 'campaign.json'
    require(continue_unstarted or not state_path.exists(), 'original campaign exists; inspect it and explicitly continue only unstarted cases')
    # Capture actual source-specific argv/settings; never resume an existing model.
    env_base = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for case in proposal['cases']:
        identifier = case['id']
        if state_path.exists():
            row = load_campaign(state_path)['cases'][identifier]
            if row['status'] != 'reserved':
                require(continue_unstarted, 'an original control was already enrolled')
                if row['status'] in ('claimed', 'worker_terminal_pending_guard'):
                    original = process_identity(row['original_worker']['pid'])
                    require(original is None, 'original control worker remains live; poll its exact handle, never launch a replacement')
                    settle_case(repo, a, identifier, decision_sha)
                continue
        require(sha(decision_path) == decision_sha, 'the original owner decision changed before the next case')
        logs = campaign / 'worker-logs'; logs.mkdir(parents=True, exist_ok=True)
        label = canonical(identifier)
        command = [str(repo / '.venv/bin/python'), str(Path(__file__).resolve()), 'run-case', '--repo', str(repo),
                   '--allocation', str(allocation_path), '--allocation-sha', expected_sha,
                   '--decision', str(Path(decision_path).resolve()), '--decision-sha', decision_sha, '--case-id', identifier]
        env = dict(env_base, PYTHONPATH=str(repo / case['source_package_path']) + os.pathsep
                   + str(repo / a['source_verifier_checkout']) + os.pathsep + str(Path(__file__).resolve().parents[1]))
        with (logs / (label + '.stdout')).open('xb') as out, (logs / (label + '.stderr')).open('xb') as err:
            child = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], env=env,
                                     stdout=out, stderr=err, start_new_session=True)
            try:
                code = child.wait(timeout=float(case['complete_process_ceiling_seconds']) + 1)
            except subprocess.TimeoutExpired:
                # This is the declared worker deadline, not a UI observation timeout.
                if child.poll() is None:os.killpg(child.pid, signal.SIGKILL)
                child.wait(); raise RuntimeError('original worker exceeded its hard deadline; preserve and review it')
        deadline = time.monotonic() + 60
        guard_path = campaign / 'guards' / (label + '.json')
        while not guard_path.exists() or load(guard_path)['status'] == 'monitoring_original_worker':
            require(time.monotonic() < deadline, 'original guard is unresolved; preserve its handle and wait, without replacement')
            time.sleep(.2)
        settle_case(repo, a, identifier, decision_sha)
        require(code == 0, 'original control process failed; preserve its data and stop further launches')
    print(json.dumps({'status': 'fixed_original_controls_terminal_pending_independent_audits',
                      'campaign_root': str(campaign), 'a2_complete': False}), flush=True)


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    for name in ('preview', 'run-case', 'settle', 'run-serial'):
        p = sub.add_parser(name); p.add_argument('--repo', required=True); p.add_argument('--allocation', required=True); p.add_argument('--allocation-sha', required=True)
        if name != 'preview':
            p.add_argument('--decision', required=True); p.add_argument('--decision-sha', required=True)
            if name != 'run-serial':p.add_argument('--case-id', required=True)
            else:p.add_argument('--continue-unstarted', action='store_true')
        else:
            p.add_argument('--output', required=True)
    g = sub.add_parser('guard'); g.add_argument('--identity', required=True); g.add_argument('--cap', type=float, required=True); g.add_argument('--output', required=True)
    g.add_argument('--max-active-seconds', type=float, default=900.)
    args = parser.parse_args()
    if args.command == 'guard':
        guard(load(args.identity), args.cap, args.output, max_active_seconds=args.max_active_seconds)
    elif args.command == 'preview':
        result = preview(args.repo, args.allocation, args.allocation_sha); output = Path(args.output).resolve()
        a = load(args.allocation)
        require(output.is_relative_to((Path(args.repo) / a['preview_root']).resolve())
                and not output.is_relative_to((Path(args.repo) / a['campaign_root']).resolve())
                and not output.exists(), 'new preview must stay outside the planned campaign and original inputs')
        write(output, result); print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == 'run-case':
        run_case(args.repo, args.allocation, args.allocation_sha, args.decision, args.decision_sha, args.case_id)
    elif args.command == 'run-serial':
        run_serial(args.repo, args.allocation, args.allocation_sha, args.decision, args.decision_sha, args.continue_unstarted)
    else:
        a0 = load(args.allocation); require(sha(args.allocation) == args.allocation_sha and sha(args.decision) == args.decision_sha, 'original settlement bindings changed')
        decision_limit(load(Path(args.repo) / a0['proposal']), load(args.decision)); a0['allocation_sha256'] = args.allocation_sha
        settle_case(args.repo, a0, args.case_id, args.decision_sha)


if __name__ == '__main__':
    main()

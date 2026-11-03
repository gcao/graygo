from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

repo = Path('/home/gcao/proj/graygo')
root = Path(__file__).resolve().parent
frozen = repo / '.worktrees/task-actor-boundary-53aa73c'
allocation = frozen / 'docs/experiments/task-actor-boundary-allocation.json'
a = json.loads(allocation.read_text())
tool = frozen / a['producer_path']
started = time.monotonic()
child = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def snapshot():
    files, directories = {}, []
    for archive in a['trained_archives'].values():
        original = (repo / archive['path']).parent.parent
        for p in sorted(original.rglob('*')):
            assert not p.is_symlink()
            name = str(p.relative_to(repo))
            if p.is_file():
                files[name] = sha(p)
            elif p.is_dir():
                directories.append(name)
    return {'files': files, 'directories': directories}


assert sha(allocation) == '6903c6e96b0f3a9abd431aa6984efa02002dd4070f6d463a0c701b052f99cf91'
assert sha(tool) == a['producer_sha256']
assert not subprocess.check_output(['git', '-C', str(frozen), 'status', '--porcelain'], text=True)
for role in ('reference', 'candidate'):
    package = (repo / a[role + '_package_path']).parent
    assert not subprocess.check_output(['git', '-C', str(package), 'status', '--porcelain'], text=True)
before = snapshot()
(root / 'original-input-snapshot.json').write_text(json.dumps(before, indent=2, sort_keys=True) + '\n')
body = {'version': 1, 'status': 'running', 'original_driver_pid': os.getpid(), 'allocation_sha256': sha(allocation),
        'tool_sha256': sha(tool), 'tool_commit': subprocess.check_output(['git', '-C', str(frozen), 'rev-parse', 'HEAD'], text=True).strip(),
        'driver_sha256': sha(__file__), 'original_input_snapshot_sha256': canonical(before),
        'commands': [], 'new_training_records': 0, 'new_evaluation_games': 0, 'new_accelerator_job': False,
        'source_compatibility_qualified': False, 'a2_evidence_reusable': False, 'execution_authorized': False,
        'outer_watchdog_seconds': a['outer_watchdog_seconds']}


def save():
    tmp = root / 'job.tmp'
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True) + '\n')
    tmp.replace(root / 'job.json')


def alarm(*args):
    raise TimeoutError('actor boundary original driver watchdog exhausted')


signal.signal(signal.SIGALRM, alarm)
signal.alarm(1230)
save()
try:
    for learner, archive in a['trained_archives'].items():
        bundle = repo / archive['path']
        assert sha(bundle / 'manifest.json') == archive['manifest_sha256']
        assert sha(bundle / 'models.pt') == archive['models_sha256']
        for role in ('reference', 'candidate'):
            label = learner + '-' + role
            output = root / (label + '.json')
            cmd = [str(repo / '.venv/bin/python'), '-m', 'tools.task_actor_boundary', '--archive', str(bundle),
                   '--expected-source-sha', a[role + '_source_sha256'], '--expected-archive-source-sha', a['reference_source_sha256'],
                   '--expected-archive-manifest-sha', archive['manifest_sha256'], '--time-cap-seconds', str(a['per_probe_time_ceiling_seconds']),
                   '--output', str(output)]
            env = dict(os.environ, PYTHONPATH=str(repo / a[role + '_package_path']) + os.pathsep + str(frozen))
            entry = {'label': label, 'command': cmd}
            body['commands'].append(entry)
            remaining = a['process_time_ceiling_seconds'] - (time.monotonic() - started) - 20
            assert remaining > 0
            with (root / (label + '.stdout')).open('w') as stdout, (root / (label + '.stderr')).open('w') as stderr:
                tick = time.monotonic()
                child = subprocess.Popen(cmd, cwd=frozen, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                entry['pid'] = child.pid
                body['current_probe'] = label
                save()
                code = child.wait(timeout=min(a['per_probe_time_ceiling_seconds'] + 15, remaining))
                entry.update(returncode=code, seconds=time.monotonic() - tick)
                if code != 0:
                    raise RuntimeError(label + ' failed with returncode ' + str(code))
                entry['result_sha256'] = sha(output)
                save()
    sys.path.insert(0, str(frozen))
    from tools.task_actor_boundary import verify_pair
    comparisons = {}
    for learner, archive in a['trained_archives'].items():
        old = json.loads((root / (learner + '-reference.json')).read_text())
        new = json.loads((root / (learner + '-candidate.json')).read_text())
        comparisons[learner] = verify_pair(old, new, reference_source=a['reference_source_sha256'],
                                           candidate_source=a['candidate_source_sha256'], archive_manifest_sha=archive['manifest_sha256'])
    assert snapshot() == before
    result = {'version': 1, 'status': 'both_finite_actor_boundaries_verified', 'comparisons': comparisons,
              'source_compatibility_qualified': False, 'a2_evidence_reusable': False, 'execution_authorized': False}
    (root / 'comparison.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    body.update(status='verified', comparison_sha256=sha(root / 'comparison.json'), current_probe=None)
except BaseException as exc:
    body.update(status='failed', error=repr(exc))
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    body['original_inputs_unchanged'] = snapshot() == before
    if not body['original_inputs_unchanged']:
        body.update(status='failed', error='original input snapshot changed')
    body['measured_driver_seconds'] = time.monotonic() - started
    save()
    signal.alarm(0)

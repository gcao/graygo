from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time

repo = Path('/home/gcao/proj/graygo')
root = Path(__file__).resolve().parent
frozen = repo / '.worktrees/task-pipeline-parity-d724e06'
allocation = frozen / 'docs/experiments/task-pipeline-parity-allocation.json'
a = json.loads(allocation.read_text())
started = time.monotonic()
child = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def package_sha(path):
    package = path / 'graygo'
    h = hashlib.sha256()
    for p in sorted([*package.glob('*.py'), *(package / 'native').glob('*.cpp')]):
        h.update(p.relative_to(package).as_posix().encode() + b'\0' + p.read_bytes() + b'\0')
    return h.hexdigest()


assert sha(allocation) == '2380f7de0c7d4a43d1d99314cc6b276f5f43f56f146f015339f4a454e372a666'
assert sha(frozen / a['profile_producer']) == a['profile_producer_sha256']
assert sha(frozen / a['comparison_tool']) == a['comparison_tool_sha256']
assert sha(frozen / 'tools/task_actor_boundary.py') == a['transcript_helper_sha256']
assert not subprocess.check_output(['git', '-C', str(frozen), 'status', '--porcelain'], text=True)
for role in ('reference', 'candidate'):
    path = repo / a[role + '_package_path']
    assert package_sha(path) == a[role + '_source_sha256']
    assert not subprocess.check_output(['git', '-C', str(path.parent), 'status', '--porcelain'], text=True)
old_inventory = json.loads((repo / 'docs/verification/task-source-compatibility-ca45-c3b8/actor-boundary/original-input-snapshot.json').read_text())
assert all(sha(repo / name) == digest for name, digest in old_inventory['files'].items())
body = {'version': 1, 'status': 'running', 'original_driver_pid': os.getpid(),
        'tool_commit': subprocess.check_output(['git', '-C', str(frozen), 'rev-parse', 'HEAD'], text=True).strip(),
        'driver_sha256': sha(__file__), 'allocation_sha256': sha(allocation), 'profile_producer_sha256': a['profile_producer_sha256'],
        'comparison_tool_sha256': a['comparison_tool_sha256'], 'control_reserved_seconds': a['additional_control_commitment_seconds'],
        'commands': [], 'new_formal_A2_records': 0, 'new_scored_evaluation_games': 0, 'new_accelerator_job': False,
        'source_compatibility_qualified': False, 'a2_evidence_reusable': False, 'execution_authorized': False,
        'outer_watchdog_seconds': a['outer_watchdog_seconds']}


def save():
    tmp = root / 'job.tmp'
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True) + '\n')
    tmp.replace(root / 'job.json')


def run(label, cmd, env, ceiling):
    global child
    remaining = a['process_time_ceiling_seconds'] - (time.monotonic() - started) - 20
    assert remaining > 0
    row = {'label': label, 'command': cmd}
    body['commands'].append(row)
    with (root / (label + '.stdout')).open('w') as out, (root / (label + '.stderr')).open('w') as err:
        tick = time.monotonic()
        child = subprocess.Popen(cmd, cwd=frozen, env=env, stdout=out, stderr=err, start_new_session=True)
        row['pid'] = child.pid
        body['current_phase'] = label
        save()
        code = child.wait(timeout=min(ceiling, remaining))
    row.update(returncode=code, seconds=time.monotonic() - tick)
    save()
    if code != 0:
        raise RuntimeError(label + ' returned ' + str(code))


def alarm(*args):
    raise TimeoutError('original matched-pipeline watchdog exhausted')


signal.signal(signal.SIGALRM, alarm)
signal.alarm(1830)
save()
try:
    for role in ('reference', 'candidate'):
        env = dict(os.environ, PYTHONPATH=str(repo / a[role + '_package_path']) + os.pathsep + str(frozen))
        cmd = [str(repo / '.venv/bin/python'), str(frozen / a['profile_producer']), '--run-dir', str(root / role),
               '--expected-source-sha', a[role + '_source_sha256'], '--records', '65536', '--device', 'cpu',
               '--case-minutes', '3', '--task', 'delayed', '--learners', 'neural', 'tabular', '--reuses', '4']
        run(role, cmd, env, a['per_profile_outer_seconds'])
        status = json.loads((root / role / 'state.json').read_text())
        plan = json.loads((root / role / 'plan.json').read_text())
        assert status['status'] == 'complete' and status['completed_cases'] == 4 and plan['cases'] == a['cases']
    env = dict(os.environ, PYTHONPATH=str(repo / a['candidate_package_path']) + os.pathsep + str(frozen))
    cmd = [str(repo / '.venv/bin/python'), '-m', 'tools.task_pipeline_parity', '--reference', str(root / 'reference'),
           '--candidate', str(root / 'candidate'), '--allocation', str(allocation), '--expected-allocation-sha', sha(allocation),
           '--output', str(root / 'comparison.json')]
    run('comparison', cmd, env, 180)
    result = json.loads((root / 'comparison.json').read_text())
    assert result['status'] == 'four_finite_complete_pipelines_verified' and result['aggregate_new_diagnostic_records'] == 524288
    body.update(status='verified', current_phase=None, comparison_sha256=sha(root / 'comparison.json'),
                aggregate_new_diagnostic_records=524288)
except BaseException as exc:
    body.update(status='failed', error=repr(exc))
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    body['original_learning_inputs_unchanged'] = all(sha(repo / name) == digest for name, digest in old_inventory['files'].items())
    body['original_source_packages_unchanged'] = all(package_sha(repo / a[role + '_package_path']) == a[role + '_source_sha256'] for role in ('reference', 'candidate'))
    if not body['original_learning_inputs_unchanged'] or not body['original_source_packages_unchanged']:
        body.update(status='failed', error='original input/source changed')
    body['measured_driver_seconds'] = time.monotonic() - started
    save()
    signal.alarm(0)

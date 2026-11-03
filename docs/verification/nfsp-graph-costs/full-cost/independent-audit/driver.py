from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time

repo = Path('/home/gcao/proj/graygo')
root = Path(__file__).resolve().parent
allocation = repo / 'docs/experiments/nfsp-graph-cost-independent-audit-allocation.json'
a = json.loads(allocation.read_text())
frozen = repo / a['frozen_tooling_path']
original = repo / a['original_run']
started = time.monotonic()
child = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


assert sha(allocation) == 'fc96bec27ab758f2237793e2c53e2718979c2dd5da7ccee8a408fb16ae39f6fb'
assert not subprocess.check_output(['git', '-C', str(frozen), 'status', '--porcelain'], text=True)
for name, key in [('audit_nfsp_graph_costs.py', 'auditor_sha256'),
                  ('audit_nfsp_cost.py', 'core_auditor_sha256'),
                  ('nfsp_graph_costs.py', 'producer_sha256'),
                  ('nfsp_cost_profile.py', 'core_producer_sha256'),
                  ('nfsp_batch_costs.py', 'components_sha256')]:
    assert sha(frozen / 'tools' / name) == a[key]
measurement_job = json.loads((original / 'job.json').read_text())
assert measurement_job['status'] == 'complete_measurement_unreviewed'
assert measurement_job['returncode'] == 0
assert measurement_job['original_driver_pid'] == a['original_driver_pid']
assert measurement_job['profile_pid'] == a['original_profile_pid']
inputs = sorted(p for p in original.rglob('*') if p.is_file() and not p.name.endswith('.lock'))
before = {str(p): sha(p) for p in inputs}
body = {'version': 1, 'status': 'running', 'original_driver_pid': os.getpid(),
        'allocation_sha256': sha(allocation), 'driver_sha256': sha(__file__),
        'auditor_sha256': a['auditor_sha256'], 'original_measurement_job_sha256': sha(original / 'job.json'),
        'new_training_records': 0, 'new_evaluation_games': 0,
        'new_accelerator_job': False, 'a2_complete': False, 'execution_authorized': False,
        'watchdog_seconds': a['outer_watchdog_seconds']}


def save():
    tmp = root / 'job.tmp'
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True) + '\n')
    tmp.replace(root / 'job.json')


def alarm(*args):
    raise TimeoutError('independent graph audit watchdog exhausted')


signal.signal(signal.SIGALRM, alarm)
signal.alarm(640)
save()
try:
    cmd = [str(repo / '.venv/bin/python'), '-m', 'tools.audit_nfsp_graph_costs',
           '--run-dir', str(original), '--repo', str(repo),
           '--allocation', str(frozen / a['original_measurement_allocation']),
           '--producer', str(frozen / 'tools/nfsp_graph_costs.py'),
           '--core-producer', str(frozen / 'tools/nfsp_cost_profile.py'),
           '--components', str(frozen / 'tools/nfsp_batch_costs.py'),
           '--expected-source-sha', a['learner_source_sha256'],
           '--expected-producer-sha', a['producer_sha256'],
           '--expected-allocation-sha', a['original_measurement_allocation_sha256'],
           '--output-dir', str(root / 'result')]
    body['command'] = cmd
    env = dict(os.environ, PYTHONPATH=str(repo / '.worktrees/frozen-cuda-c3b8b5f/src') + os.pathsep + str(frozen))
    with (root / 'stdout.json').open('w') as stdout, (root / 'stderr.log').open('w') as stderr:
        child = subprocess.Popen(cmd, cwd=frozen, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        body['audit_pid'] = child.pid
        save()
        code = child.wait(timeout=a['process_time_ceiling_seconds'])
    body['returncode'] = code
    if code == 0:
        result = json.loads((root / 'result/audit.json').read_text())
        assert result['status'] == 'verified_empirical_accounting'
        assert not result['execution_authorized'] and not result['a2_complete']
        body.update(status='verified', audit_sha256=sha(root / 'result/audit.json'))
    else:
        body['status'] = 'failed'
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
    body['original_inputs_unchanged'] = before == {str(p): sha(p) for p in inputs}
    if not body['original_inputs_unchanged']:
        body.update(status='failed', error='original measurement inputs changed')
    body['measured_driver_seconds'] = time.monotonic() - started
    save()
    signal.alarm(0)

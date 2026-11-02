"""Bounded real-CLI integration and independent saved-state execution audit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from graygo.buffers import ReplayBuffer, ReservoirBuffer
from graygo.confidence import canonical_sha
from graygo.engine import Rules
from graygo.manifest import make_manifest, source_digest, write_report
from graygo.rule_state import RuleStateSpec
from graygo.storage import CheckpointStore, digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(a, b, path='runtime'):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=path)
    elif isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), path + ' keys differ')
        for key in a:
            exact(a[key], b[key], path + '/' + str(key))
    elif isinstance(a, (list, tuple)):
        require(type(a) == type(b) and len(a) == len(b), path + ' sequence differs')
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, path + '/' + str(i))
    else:
        require(a == b, path + ' differs')


def audit(root):
    root = Path(root)
    plan = json.loads((root / 'plan.json').read_text())
    require(plan['source_sha256'] == source_digest(), 'use the measured package source')
    require(digest(Path(plan['native_library'])) == plan['native_library_sha256'], 'measured native binary changed')
    schema = RuleStateSpec(Rules(), plan['native_library']).to_dict()
    states = []
    checks = {}
    for name, sessions in (('full', 1), ('resumed', 3)):
        directory = root / name
        wrapper = json.loads((directory / 'execution.json').read_text())
        body = wrapper['body']
        require(wrapper['sha256'] == canonical_sha(body), 'clock checksum differs')
        clock, contract = body['clock'], body['contract']
        require(contract['source_sha256'] == source_digest() and contract['runtime']['native_library'] == plan['native_library'], 'source/decoder differs')
        config = contract['runtime']
        require(contract['native_schema'] == schema and (config['width'], config['blocks'], config['batch'],
                config['capacity'], config['learning_start'], config['q_reuse'], config['target_every'])
                == (64, 8, 512, 500000, 4096, 4, 12800), 'full model/cadence/native identity differs')
        require(contract['journal']['scope'] == 'diagnostic' and contract['journal']['records'] == 16384, 'diagnostic scope differs')
        require(clock['inflight_since'] is None and clock['sessions'] == sessions and clock['unclean_recoveries'] == 0, 'clock/session mismatch')
        require(clock['observed_seconds'] > 0 and clock['recovery_charge_seconds'] == 0, 'invalid observed/recovery clock')
        require(clock['observed_seconds'] <= contract['execution']['time_ceiling_seconds'] == 300., 'diagnostic time ceiling exceeded')
        store = CheckpointStore(directory)
        latest = store.latest()
        seal = store.verify(latest)
        require(seal['step'] == clock['last_checkpoint']['records'] == 16384, 'endpoint differs')
        require(digest(latest / 'manifest.json') == clock['last_checkpoint']['manifest_sha256'], 'clock/bundle binding differs')
        require(digest(directory / 'run.json') == clock['training_run_sha256'], 'clock/run binding differs')
        control = json.loads((latest / 'controller.json').read_text())
        require(seal['metadata']['controller_sha256'] == canonical_sha(control), 'controller seal differs')
        require(len(list((directory / 'resume').glob('[0-9]*'))) == 1, 'final retention differs')
        require(sorted(int(p.name) for p in (directory / 'weights').glob('[0-9]*')) == [0, 4096, 8192, 12288, 16384], 'permanent archives differ')
        for point in (0, 4096, 8192, 12288, 16384):
            public = directory / 'weights' / f'{point:012d}'
            require(set(store.verify(public)['files']) == {'models.pt'}, 'public archive copied replay')
            require(control['archived'][str(point)] == digest(public / 'manifest.json'), 'archive/control binding differs')
        decision = json.loads((directory / 'opening-decision.json').read_text())
        require(decision['sha256'] == canonical_sha(decision['body']) and decision['body']['kind'] == 'diagnostic'
                and decision['body']['records'] == 8192 and decision['body']['status'] == 'pass', 'opening differs')
        require(control['opening_sha256'] == digest(directory / 'opening-decision.json'), 'decision/controller differs')
        require(decision['body']['archive_manifest_sha256'] == digest(directory / 'weights' / '000000008192' / 'manifest.json'), 'decision model differs')
        state = torch.load(latest / 'runtime.pt', weights_only=True, map_location='cpu')
        require(state['config'] == config, 'saved runtime configuration differs')
        public = torch.load(directory / 'weights' / '000000016384' / 'models.pt', weights_only=True, map_location='cpu')
        exact(public['q'], state['backend']['q'], 'final public Q')
        exact(public['average'], state['backend']['average'], 'final public average')
        replay = ReplayBuffer.load(latest / 'replay.npz')
        reservoir = ReservoirBuffer.load(latest / 'reservoir.npz')
        require(replay.insertions == replay.size == 16384 and reservoir.seen == reservoir.size > 0, 'buffer accounting differs')
        require((reservoir.arrays['record'][:reservoir.size] >= 8192).all(), 'early labels')
        require(state['backend']['counters']['q_updates'] == 100 and state['backend']['counters']['q_loss_targets'] == 51200, 'Q reuse differs')
        require(state['backend']['counters']['supervised_updates'] > 0, 'no averaging update')
        require(state['backend']['counters']['supervised_targets'] == 512 * state['backend']['counters']['supervised_updates'], 'SL target clock differs')
        require(state['collector']['metrics']['per_role_records'] == [8192, 8192], 'unbalanced roles')
        state.pop('timings')
        states.append((state, replay, reservoir))
        checks[name] = {'observed_seconds': clock['observed_seconds'], 'sessions': sessions,
                        'final_manifest_sha256': digest(latest / 'manifest.json'), 'labels': reservoir.seen,
                        'counters': state['backend']['counters'], 'clock_sha256': digest(directory / 'execution.json')}
    exact(states[0][0], states[1][0])
    for a, b in zip(states[0][1:], states[1][1:]):
        for key in a.arrays:
            np.testing.assert_array_equal(a.arrays[key], b.arrays[key])
    return {'verified': True, 'purpose': 'diagnostic', 'source_sha256': source_digest(), 'auditor_sha256': digest(Path(__file__)),
            'plan_sha256': digest(root / 'plan.json'), 'checks': checks,
            'exact_complete_saved_state_and_buffers': True, 'early_labels_absent': True,
            'live_tensor_device_layout': 'covered by separate CPU/CUDA executor tests; saved-state audit loads tensors on CPU',
            'formal_scope_approved': False, 'a2_complete': False}


def collect(root, native):
    root = Path(root).resolve()
    require(not root.exists(), 'diagnostic already exists; inspect original process/output, never restart on observation timeout')
    root.mkdir(parents=True)
    native = str(Path(native).resolve())
    require(Path(native).is_file(), 'explicit native decoder required')
    plan = {'purpose': 'diagnostic', 'source_sha256': source_digest(), 'native_library': native,
            'native_library_sha256': digest(Path(native)), 'records_per_trajectory': 16384,
            'aggregate_record_cap': 32768, 'cumulative_time_ceiling_seconds_per_trajectory': 300.,
            'commands': [['full', '--opening-decision', 'pass'], ['resumed', '--until-records', '4096'],
                         ['resumed', '--resume'], ['resumed', '--resume', '--opening-decision', 'pass']],
            'formal_scope_approved': False, 'a2_complete': False}
    manifest = make_manifest(seed=17, command={'check_board_execution': plan}, rules={'size': 9, 'max_rounds': 200, 'collision': 'gray'})
    require(manifest['git_dirty'] is False, 'commit the measured source before collection')
    write_report(root / 'plan.json', plan)
    write_report(root / 'run.json', manifest)
    started = time.monotonic()
    observed = []
    for i, (name, *extra) in enumerate(plan['commands']):
        command = [sys.executable, '-m', 'graygo', 'run-board-diagnostic', '--run-dir', str(root / name),
                   '--records', '16384', '--seed', '17', '--device', 'cuda', '--q-reuse', '4',
                   '--time-cap-hours', str(300 / 3600), '--native-library', native, *extra]
        log = root / f'command-{i}.json'
        with log.open('w') as output:
            subprocess.run(command, stdout=output, check=True)
        result = json.loads(log.read_text())['result']
        observed.append({'status': result['status'], 'records': result['records'], 'clock': result['clock']})
        expected = [('complete', 16384), ('checkpoint_boundary', 4096), ('opening_pending', 8192), ('complete', 16384)][i]
        require((result['status'], result['records']) == expected, 'command stopped; inspect this original diagnostic instead of starting a replacement')
    elapsed = time.monotonic() - started
    result = audit(root)
    write_report(root / 'audit.json', result)
    write_report(root / 'report.json', {'verified': True, 'purpose': 'diagnostic', 'elapsed_collection_seconds': elapsed,
                 'commands': observed, 'source_sha256': source_digest(), 'aggregate_records': 32768,
                 'formal_scope_approved': False, 'a2_complete': False})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    parser.add_argument('--native-library')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if not args.audit_only and not args.native_library:
        parser.error('--native-library is required for collection')
    result = audit(args.run_dir) if args.audit_only else collect(args.run_dir, args.native_library)
    print(json.dumps(result, sort_keys=True))

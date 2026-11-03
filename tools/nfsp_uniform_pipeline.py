"""Dedicated bounded uniform-fallback hardware pipeline; no formal training grant."""
from dataclasses import replace
import fcntl
import gc
import math
from pathlib import Path
import time
import torch
from graygo.board_runtime import BoardNFSPRuntime, BoardRuntimeConfig, RUNTIME_VERSION, configure_runtime
from graygo.board_profile import _memory, _synchronize
from graygo.engine import State
from graygo.evaluation import BaselinePolicy, FrozenNeuralPolicy, evaluate_pair, seed_for
from graygo.manifest import make_manifest, source_digest, write_report
from graygo.storage import CheckpointStore

PROFILE_VERSION = "graygo/uniform-board-hardware-profile/1"

def profile_board(config, root, *, records, time_cap_seconds, stop_requested=lambda: False):
    """Measure both closed and early-open diagnostic phases with fixed full models.

    Each phase starts a new learner and uses at most 32,768 fresh records. The
    early-open phase deliberately exercises supervised/average work cheaply; its
    game lengths and acting mix cannot price a trained 5M/7M/80M campaign. This
    non-resumable measurement cannot emit A2, pilot, or promotion approval.
    """
    if (type(records) is not int or records % config.batch or not config.learning_start <= records <= 32768
            or not math.isfinite(time_cap_seconds) or not 0 < time_cap_seconds <= 3600):
        raise ValueError('profile needs a batch-aligned endpoint <=32768 per phase and <=1 hour total')
    if config.rules.size != 9 or config.rules.max_rounds != 200 or config.rules.collision != 'gray':
        raise ValueError('hardware profile uses the official 9x9/200-round gray rules')
    if config.model_config() != BoardRuntimeConfig().model_config():
        raise ValueError('hardware profile requires the full 64-channel/eight-block network')
    if config.batch != 512 or config.learning_start != 4096 or config.target_every != 12800:
        raise ValueError('hardware profile requires the declared production logical cadence')
    if config.reservoir_open != 5_000_000 or config.handoff_end != 7_000_000 or not config.uniform_comparison:
        raise ValueError('uniform reference must retain its declared opening and hand-off schedule')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.profile.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'plan.json').exists():
            raise FileExistsError('profile already exists; inspect its original process/output')
        CheckpointStore(root / 'closed' / 'checkpoints').ensure_space()
        phases = {'closed': config, 'early_open_diagnostic': replace(config, reservoir_open=0, handoff_end=2)}
        plan = {
            'version': PROFILE_VERSION, 'runtime_version': RUNTIME_VERSION, 'source_sha256': source_digest(),
            'records_per_phase': records, 'total_record_cap': records * 2, 'total_time_ceiling_seconds': time_cap_seconds,
            'phases': {name: value.plan() for name, value in phases.items()},
            'evaluation': {'representations': ['greedy_q', 'average'], 'opponents': ['random', 'tactical'],
                           'color_pairs_per_match': 1, 'strength_claim': False},
            'scope': 'bounded untrained pipeline measurement; early opening is diagnostic only',
            'a2_complete': False, 'formal_learning_run': False, 'promotion_permitted': False,
            'budget_boundary': 'checked between collector chunks and complete color pairs; atomic IO may overrun',
        }
        write_report(root / 'plan.json', plan)
        manifest = make_manifest(seed=config.seed, command={'profile': plan}, rules=config.plan()['rules'])
        write_report(root / 'run.json', manifest)
        started = time.monotonic()
        output = {'version': PROFILE_VERSION, 'plan': plan, 'phases': {}, 'status': 'running'}
        write_report(root / 'progress.json', output)
        for name, phase in phases.items():
            if stop_requested() or time.monotonic() - started >= time_cap_seconds:
                output['status'] = 'interrupted' if stop_requested() else 'budget_limited'
                break
            phase_started = time.perf_counter()
            device = torch.device(phase.device)
            configure_runtime(device, phase.deterministic)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            setup = time.perf_counter()
            runtime = BoardNFSPRuntime(phase)
            _synchronize(device)
            setup = time.perf_counter() - setup
            result = runtime.run_profile(records, time_cap_seconds=max(.001, time_cap_seconds - (time.monotonic() - started)),
                                         stop_requested=stop_requested)
            _synchronize(device)
            result.update({'setup_seconds': setup, 'settings': runtime.settings, 'native_schema': runtime.spec.to_dict(),
                           'q_parameters': sum(p.numel() for p in runtime.backend.q.parameters()),
                           'average_parameters': sum(p.numel() for p in runtime.backend.average.parameters()),
                           'buffer_array_bytes': sum(a.nbytes for buffer in (runtime.replay, runtime.reservoir) for a in buffer.arrays.values()),
                           'reservoir_seen': runtime.reservoir.seen, 'reservoir_size': runtime.reservoir.size})
            # Save even on a cooperative budget/signal stop; preserve the actual state.
            saved_at = time.perf_counter()
            checkpoint = runtime.save(root / name / 'checkpoints')
            result['checkpoint_write_seconds'] = time.perf_counter() - saved_at
            sealed = CheckpointStore.verify(checkpoint)
            result['checkpoint_payload_bytes'] = sum(f['bytes'] for f in sealed['files'].values())
            result['checkpoint'] = str(checkpoint)
            # Restore in place to measure strict decoding and retain one model allocation.
            restored_at = time.perf_counter()
            runtime.restore(checkpoint)
            _synchronize(device)
            result['checkpoint_restore_seconds'] = time.perf_counter() - restored_at
            result['restore_record_match'] = runtime.records == result['records']
            CheckpointStore(root / name / 'checkpoints').finalize()
            result['evaluation_pairs'] = []
            evaluation_started = time.perf_counter()
            for representation in ('greedy_q', 'average'):
                model = runtime.backend.q if representation == 'greedy_q' else runtime.backend.average
                subject = FrozenNeuralPolicy(model, representation=representation, label='hardware-only')
                for opponent_name in ('random', 'tactical'):
                    if stop_requested() or time.monotonic() - started >= time_cap_seconds:
                        break
                    pair = evaluate_pair(subject, BaselinePolicy(opponent_name), State(phase.rules),
                                         unit_seed=seed_for(config.seed, 'hardware-evaluation', representation, opponent_name))
                    result['evaluation_pairs'].append({'representation': representation, 'opponent': opponent_name, 'pair': pair})
                    write_report(root / name / 'evaluation.json', {'scope': 'timing diagnostic; no confidence or strength decision',
                                                                 'pairs': result['evaluation_pairs']})
                del subject
            _synchronize(device)
            result['evaluation_seconds'] = time.perf_counter() - evaluation_started
            result.update(_memory(device))
            result['total_phase_seconds'] = time.perf_counter() - phase_started
            result['covered'] = {
                'closed_collection': name != 'closed' or result['reservoir_seen'] == 0,
                'q_update': result['counters']['q_updates'] > 0,
                'target_refresh': result['counters']['target_refreshes'] > 0,
                'supervised_update': name == 'closed' or result['counters']['supervised_updates'] > 0,
                'average_acting': name == 'closed' or result['metrics']['acting_counts']['average'] > 0,
                'paired_evaluation': len(result['evaluation_pairs']) == 4,
                'restore': result['restore_record_match'],
            }
            output['phases'][name] = result
            output['elapsed_seconds'] = time.monotonic() - started
            write_report(root / 'progress.json', output)
            del runtime
            gc.collect()
            if result['status'] != 'complete':
                output['status'] = result['status']
                break
        if output['status'] == 'running':
            output['status'] = 'complete' if len(output['phases']) == 2 and all(all(r['covered'].values()) for r in output['phases'].values()) else 'incomplete_coverage'
        output['elapsed_seconds'] = time.monotonic() - started
        output['full_campaign_price_established'] = False
        output['a2_complete'] = output['promotion_permitted'] = output['formal_learning_run'] = False
        write_report(root / 'report.json', output)
        write_report(root / 'progress.json', output)
        return output

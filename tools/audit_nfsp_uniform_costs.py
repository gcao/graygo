"""Read-only independent uniform-reference component/price accounting."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from decimal import Decimal
import fcntl
import hashlib
import json
import math
from pathlib import Path
from graygo.manifest import source_digest, write_report
from graygo.storage import CheckpointStore
from tools.audit_nfsp_cost import require, sha, load, close
from tools import audit_nfsp_graph_costs as original_graph_audit

VERSION = 'graygo/uniform-reference-cost-audit/1'


def calculate(base, measurements, setup):
    """Independent 8M/three-seed equation; never import the new producer."""
    n = 8000000; prior = base['forecast']['per_seed']['pilot']; rates = base['core_report']['rates']
    require(prior['records'] == n and prior['evaluation_games'] == 36000 and prior['evaluation_endpoints'] == 18,
            'complete fixed reference exposure/evaluation scope differs')
    require(prior['snapshot_publication_upper'] == 15626 and prior['graph_capture_upper'] == 164
            and prior['weight_pack_upper'] == 31320, 'source publication/cold count changed')
    observers = measurements['pipeline_observers']; phases = measurements['pipeline']['phases']
    collection = [rates['collection_per_record']]
    require(len(observers) == 2 and {o['phase'] for o in observers} == set(phases), 'actual phase observer inventory differs')
    for observer in observers:
        phase = phases[observer['phase']]
        expected = phase['timings']['collection'] - observer['actor_seconds'] - observer['snapshot_copy_seconds']
        require(expected >= -.001 and observer['records'] == phase['records'], 'nested uniform scopes overlap')
        close(observer['resolver_storage_bookkeeping_seconds'], max(0., expected), 'uniform residual')
        collection.append(max(0., expected) / phase['records'])
    dense = measurements['dense']
    close(dense['resolver_storage_bookkeeping_seconds'], max(0., dense['total_seconds'] - dense['actor_seconds']), 'dense uniform residual')
    collection.append(dense['resolver_storage_bookkeeping_seconds'] / dense['new_profile_records'])
    actor = measurements['actors']; durations = []
    require(set(actor['cases']) == {'shared_uniform', 'distinct_q_uniform'} and actor['requests_per_sample'] == 32,
            'both exact uniform private actor layouts required')
    seen_events = set()
    for rows in actor['cases'].values():
        require(len(rows) == 4, 'fixed actual actor timing sample count differs')
        for row in rows:
            first, last = row['graph_event_span']
            require(type(first) is int and type(last) is int and 0 <= first <= last <= len(setup['events']), 'invalid source setup span')
            indices = set(range(first, last)); require(not indices & seen_events, 'a cold setup event was subtracted twice')
            seen_events.update(indices)
            seconds = math.fsum(event['seconds'] for event in setup['events'][first:last])
            close(row['graph_setup_seconds'], seconds, 'uniform actor source setup')
            close(row['steady_seconds'], max(0., row['raw_seconds'] - seconds), 'uniform actor steady scope')
            durations.append(row['steady_seconds'])
    captures = [event['seconds'] for event in setup['events'] if event['kind'] == 'capture']
    packs = [event['seconds'] for event in setup['events'] if event['kind'] == 'weight_pack']
    require(captures and packs and all(x >= 0 and math.isfinite(x) for x in captures + packs + durations), 'actual finite cold/steady samples required')
    close(setup['capture_max_seconds'], max(captures), 'capture maximum')
    close(setup['weight_pack_max_seconds'], max(packs), 'weight-pack maximum')
    require(all(pool['eager_fallbacks'] == 0 for pool in setup['pools'].values()), 'reference escaped its selected graph path')
    parts = dict(prior['components_seconds'])
    parts['collection'] = n * max(collection)
    parts['private_actors'] = n * max(base['actors']['steady_actor_seconds_per_record'], max(durations) / actor['requests_per_sample'])
    parts['frozen_snapshots'] = 15626 * max(max(base['actors']['snapshot_copy_seconds']), max(actor['snapshot_copy_seconds']))
    parts['frozen_graph_captures'] = 164 * max(base['graph_setup']['capture_max_seconds'], max(captures))
    parts['frozen_parameter_packs'] = 31320 * max(base['graph_setup']['weight_pack_max_seconds'], max(packs))
    parts['setup'] = max(parts['setup'], max(row['setup_seconds'] for row in phases.values()))
    total = math.fsum(parts.values())
    return {'components_seconds_per_seed': parts, 'projected_seconds_before_margin': total,
            'per_seed_ceiling_seconds': 2 * total, 'H_U_device_hours_per_seed': 2 * total / 3600,
            'B_U_board_device_hours': 3 * 2 * total / 3600}


def audit(repo, job_root, allocation, expected_allocation_sha, producer, pipeline_producer):
    repo = Path(repo).resolve(); job_root = Path(job_root).resolve(); root = job_root / 'profile'; allocation = Path(allocation).resolve()
    declaration = load(allocation); job = load(job_root / 'job.json')
    require(sha(allocation) == expected_allocation_sha and sha(producer) == declaration['producer_sha256']
            and sha(pipeline_producer) == declaration['pipeline_producer_sha256'], 'original uniform source/tool/allocation changed')
    require(source_digest() == declaration['learner_source_sha256'], 'use the exact original measured package')
    require(job['status'] == 'complete_uniform_measurement_pending_audit' and job['returncode'] == 0
            and job['allocation_sha256'] == expected_allocation_sha and job['original_inputs_unchanged'], 'original uniform measurement not complete')
    require(job['measured_process_seconds'] <= declaration['outer_watchdog_seconds'], 'original GPU process exceeded its whole-job budget')
    start = Decimal(job['allocation_interval_start_epoch']); end = Decimal(job['allocation_interval_end_epoch'])
    require(end > start and abs(end - start - Decimal(str(job['measured_process_seconds']))) < Decimal('.01'), 'physical GPU interval differs')
    require(job['physical_device_uuid'] == declaration['physical_device_uuid'], 'original reference used another physical device')
    with ExitStack() as locks:
        fcntl.flock(locks.enter_context((root / '.uniform-cost.lock').open('rb')), fcntl.LOCK_SH | fcntl.LOCK_NB)
        report = load(root / 'report.json'); plan = load(root / 'plan.json'); manifest = load(root / 'run.json')
        require(report['status'] == 'complete_measurement_pending_independent_audit' and report['plan'] == plan == manifest['command']
                and sha(root / 'report.json') == job['report_sha256'], 'actual source-bound uniform report differs')
        base_root = repo / declaration['base_root']; base = load(base_root / 'report.json')
        tools = repo / declaration['base_tooling_root']
        base_audit = original_graph_audit.audit(base_root, repo, repo / declaration['base_allocation'],
            tools / 'tools/nfsp_graph_costs.py', tools / 'tools/nfsp_cost_profile.py', tools / 'tools/nfsp_batch_costs.py',
            expected_source=declaration['learner_source_sha256'], expected_producer=declaration['base_producer_sha256'],
            expected_allocation=declaration['base_allocation_sha256'])
        require(base_audit['status'] == 'verified_empirical_accounting', 'common components did not freshly requalify')
        expected = dict(base['plan']['runtime'], uniform_comparison=True)
        require(plan['runtime'] == expected and plan['source_sha256'] == manifest['source_sha256'] == declaration['learner_source_sha256'],
                'reference changed more than declared uniform fallback')
        phases = report['measurements']['pipeline']['phases']
        pipe = load(root / 'pipeline/report.json')
        require(pipe == report['measurements']['pipeline'] and pipe['plan']['phases'] ==
            {'closed': expected, 'early_open_diagnostic': dict(expected, reservoir_open=0, handoff_end=2)}, 'actual diagnostic phase recipe differs')
        for name, phase in phases.items():
            require(phase['status'] == 'complete' and phase['records'] == 32768
                    and phase['per_role_records'] == [16384, 16384] and all(phase['covered'].values()), 'incomplete real uniform pipeline')
            counters = phase['counters']
            require(counters['fresh_records'] == counters['fresh_q'] == counters['fresh_ordinary'] == 32768
                    and counters['q_updates'] == 228 and counters['q_loss_targets'] == 116736
                    and counters['target_refreshes'] == 2 and counters['supervised_targets'] == 512 * counters['supervised_updates'], 'real logical clocks differ')
            acting = phase['metrics']['acting_counts']
            require(acting['uniform'] > 0 and acting['q'] > 0, 'reference omitted actual fallback/BR behavior')
            if name == 'closed':require(phase['reservoir_seen'] == counters['supervised_updates'] == acting['average'] == 0, 'closed reference learned labels or used average')
            else:require(counters['supervised_updates'] > 0 and acting['average'] > 0, 'post reference omitted learned averaging')
            bundle = Path(phase['checkpoint']); seal = CheckpointStore.verify(bundle)
            require(seal['step'] == 32768 and seal['metadata']['source_sha256'] == declaration['learner_source_sha256'], 'actual pipeline checkpoint differs')
            require(phase['settings'] == plan['settings'], 'reference numerical flags changed')
        dense = report['measurements']['dense']
        require(dense['new_profile_records'] == 256 and dense['acting_counts']['uniform'] > 0 and dense['acting_counts']['average'] == 0,
                'actual dense closed fallback coverage missing')
        for memory in [report['memory'], *phases.values()]:
            allocated = memory['peak_cuda_allocated_bytes']; reserved = memory['peak_cuda_reserved_bytes']
            require(0 < allocated <= reserved <= declaration['reported_vram_mib'] * 1024**2,
                    'measured uniform pipeline/actor memory exceeds the selected device')
        calculated = calculate(base, report['measurements'], report['graph_setup'])
        forecast = report['forecast']
        require(forecast['seeds'] == [100, 101, 102] and forecast['records_per_seed'] == 8000000
                and forecast['aggregate_reference_records'] == 24000000 and forecast['evaluation_games_per_seed'] == 36000
                and forecast['evaluation_endpoints_per_seed'] == 18 and forecast['margin'] == 2., 'reference scope or original margin changed')
        require(forecast['A2_uniform_subset_CPU_price_qualified'] is False and forecast['full_B_U_scope_qualified'] is False
                and forecast['execution_authorized'] is False and report['a2_complete'] is False and report['formal_learning_run'] is False,
                'component forecast became full scope or training authority')
        for key, value in calculated['components_seconds_per_seed'].items():close(forecast['components_seconds_per_seed'][key], value, 'reference component ' + key)
        for key in ('projected_seconds_before_margin', 'per_seed_ceiling_seconds', 'H_U_device_hours_per_seed', 'B_U_board_device_hours'):
            close(forecast[key], calculated[key], key)
        inventory = dict(base_audit['input_files_sha256'])
        for directory in (job_root,):
            for path in directory.rglob('*'):
                if path.is_file():inventory[str(path)] = sha(path)
        inventory[str(allocation)] = sha(allocation); inventory[str(Path(producer).resolve())] = sha(producer)
        inventory[str(Path(pipeline_producer).resolve())] = sha(pipeline_producer)
        require(all(sha(p) == value for p, value in inventory.items()), 'actual uniform evidence changed during audit')
    return {'version': VERSION, 'status': 'verified_uniform_board_component_forecast', 'source_sha256': declaration['learner_source_sha256'],
        'producer_sha256': sha(producer), 'pipeline_producer_sha256': sha(pipeline_producer), 'auditor_sha256': sha(__file__),
        'input_files_sha256': inventory, 'original_inputs_unchanged': True, 'independent_calculation': calculated,
        'new_diagnostic_records': 65792, 'physical_allocation_seconds': str(end - start),
        'B_U_board_component_qualified': True, 'A2_uniform_subset_CPU_price_qualified': False,
        'full_B_U_scope_qualified': False, 'a2_complete': False, 'execution_authorized': False,
        'scope': 'Independent source-compatible common-component reuse plus actual uniform collection/actor/cold measurements. Three 8M board-reference forecasts only; A2 comparison subset and applicable CPU budgets remain separate. No runtime theorem, learned strength, full scope or launch.'}


def main():
    parser = argparse.ArgumentParser()
    for name in ('repo', 'run-dir', 'allocation', 'expected-allocation-sha', 'producer', 'pipeline-producer', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(); require(not Path(args.output).exists(), 'preserve original uniform audit output')
    result = audit(args.repo, args.run_dir, args.allocation, args.expected_allocation_sha, args.producer, args.pipeline_producer)
    write_report(args.output, result); print(json.dumps({'status': result['status'], 'B_U_board_device_hours': result['independent_calculation']['B_U_board_device_hours']}))


if __name__ == '__main__':main()

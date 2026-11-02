"""Read-only, independently calculated review of a complete NFSP cost profile.

The producer is deliberately not imported. Prices are empirical component
forecasts scoped to one source/device, never convergence or execution grants.
"""
import argparse
from decimal import Decimal
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

VERSION = 'graygo/full-buffer-nfsp-cost-audit/1'
PRODUCER = 'graygo/full-buffer-nfsp-cost-measurement/1'
STAGES = {'pilot': (8_000_000, 7), 'screening_cumulative': (20_000_000, 8), 'confirmation': (80_000_000, 9)}
RATE_NAMES = ('collection_per_record', 'actor_per_record', 'snapshot', 'q_update', 'supervised_update',
              'target_copy', 'checkpoint_write', 'checkpoint_restore', 'baseline_eval_round',
              'neural_eval_round', 'public_archive', 'public_model_load', 'policy_integrity', 'setup')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON number')))


def positive(value, name, *, zero=False):
    require(type(value) in (int, float) and math.isfinite(value) and (value >= 0 if zero else value > 0),
            'invalid ' + name)
    return value


def close(actual, expected, name):
    positive(actual, name, zero=True)
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), name + ' differs from independent calculation')


def samples(values, count, name):
    require(isinstance(values, list) and len(values) == count and count > 0, name + ' sample count differs')
    return max(positive(value, name) for value in values)


def recompute(report, pipeline):
    """Rebuild rates and every budget term without calling the producer."""
    plan = report['plan']; config = plan['runtime']; batch = config['batch']; records = plan['records_per_pipeline_phase']
    require(type(batch) is int and batch > 0 and type(records) is int and records >= config['learning_start']
            and records % batch == 0, 'invalid logical pipeline cadence')
    require(report['status'] == pipeline['status'] == 'complete', 'incomplete profile')
    require(plan['version'] == report['version'] == PRODUCER, 'unknown producer schema')
    require(pipeline['plan']['source_sha256'] == plan['source_sha256'], 'pipeline learner source differs')
    require(pipeline['plan']['phases']['closed'] == config, 'closed runtime differs')
    early = dict(config, reservoir_open=0, handoff_end=2)
    require(pipeline['plan']['phases']['early_open_diagnostic'] == early, 'post runtime changed more than diagnostic opening')
    require(set(pipeline['phases']) == {'closed', 'early_open_diagnostic'}, 'missing pipeline phase')
    observers = report['reference_pipeline_timing_observers']
    require(len(observers) == 2 and {x['phase'] for x in observers} == set(pipeline['phases']), 'timing observer phase coverage differs')
    collection = []
    for observer in observers:
        name = observer['phase']; phase = pipeline['phases'][name]; counters = phase['counters']
        require(phase['status'] == 'complete' and phase['records'] == observer['records'] == records, 'pipeline endpoint differs')
        require(counters['fresh_records'] == counters['fresh_q'] == counters['fresh_ordinary'] == records, 'fresh clocks differ')
        updates = (records // batch - config['learning_start'] // batch + 1) * config['q_reuse']
        require(counters['q_updates'] == updates and counters['q_loss_targets'] == updates * batch
                and counters['target_refreshes'] == records // config['target_every'], 'pipeline Q/target clocks differ')
        require(counters['supervised_targets'] == counters['supervised_updates'] * batch, 'supervised target count differs')
        require(phase['per_role_records'] == [records // 2] * 2 and phase['metrics']['physical_rounds'] * 2 == records, 'paired records differ')
        require(phase['restore_record_match'] is True and all(x is True for x in phase['covered'].values()), 'pipeline coverage/restore incomplete')
        if name == 'closed':
            require(phase['reservoir_seen'] == counters['supervised_updates'] == 0, 'closed pipeline learned average labels')
        else:
            require(counters['supervised_updates'] > 0 and phase['metrics']['acting_counts']['average'] > 0, 'post pipeline lacks average work')
        total = positive(phase['timings']['collection'], 'pipeline collection')
        close(observer['collection_seconds'], total, 'collection timing')
        actor = positive(observer['actor_seconds'], 'pipeline actors')
        snapshot = positive(observer['snapshot_copy_seconds'], 'pipeline snapshots')
        require(actor + snapshot <= total, 'nested timing scopes overlap outside collection')
        residual = total - actor - snapshot
        close(observer['resolver_storage_bookkeeping_seconds'], residual, 'pipeline residual')
        collection.append(residual / records)
    dense = report['dense_collection']; n_dense = plan['dense_collection_records']
    require(dense['new_profile_records'] == dense['collector_local_records'] == n_dense
            and dense['buffer_record_offset'] == config['capacity'], 'dense new-record accounting differs')
    require(dense['root_rounds_played'] == [config['rules']['max_rounds'] - 1]
            and dense['completed_games'] == n_dense // 2, 'dense collection did not finish declared last-round roots')
    require(dense['snapshot_seconds_included'] is False, 'dense snapshot work would be double priced')
    residual = positive(dense['total_seconds'], 'dense total') - positive(dense['actor_seconds'], 'dense actor')
    require(residual >= 0, 'negative dense residual')
    close(dense['resolver_storage_bookkeeping_seconds'], residual, 'dense residual')
    collection.append(residual / n_dense)
    require(report['fresh_profile_records'] == 2 * records + n_dense, 'fresh profile count includes offline rows or misses collection')
    kernel = report['kernels']; iterations = plan['kernel_iterations_each']; warmup = plan['kernel_warmup_each']
    require(kernel['measured_updates_each'] == iterations and kernel['warmup_updates_each'] == warmup
            and kernel['actual_optimizer_updates_each'] == iterations + warmup, 'kernel update counts differ')
    require(kernel['measured_q_loss_targets'] == kernel['measured_supervised_targets'] == iterations * batch
            and kernel['new_fresh_learner_records'] == 0 and kernel['replay_and_augmentation_included'] is True,
            'offline optimizer targets/fresh clocks differ')
    actors = report['actors']; envs = config['envs']
    require(actors['paired_public_states'] is True and actors['requests_per_sample'] == actors['distinct_mixed_groups'] == 2 * envs
            and actors['snapshots'] == envs, 'private actor grouping differs')
    require(set(actors['samples_seconds']) == {'shared_q', 'shared_average', 'distinct_mixed'}, 'actor layouts missing')
    actor_max = max(samples(v, plan['actor_iterations_each'], 'private actor') for v in actors['samples_seconds'].values())
    evaluation = report['evaluation']; matches = evaluation['matches']
    require({x['representation'] for x in matches} == {'q/random', 'q/tactical', 'average/random', 'average/tactical', 'average/average'}
            and len(matches) == 5 and evaluation['strength_claim'] is False, 'evaluation coverage differs')
    for match in matches:
        games = match['pair']['subject']
        require(match['paired_games'] == len(games) == 2 and match['physical_rounds'] == sum(x['rounds'] for x in games), 'evaluation physical rounds differ')
        require(all(0 < x['rounds'] <= config['rules']['max_rounds'] and x['black_score_units'] + x['white_score_units'] == 4 * config['rules']['size']**2 for x in games), 'evaluation endings invalid')
        rate = positive(match['seconds'], 'evaluation elapsed') / match['physical_rounds']
        values = evaluation['neural_seconds_per_round' if match['representation'] == 'average/average' else 'baseline_seconds_per_round']
        require(any(math.isclose(rate, x, rel_tol=1e-12, abs_tol=1e-12) for x in values), 'whole-game evaluation rate missing')
    io = report['io']
    require(io['buffer_roundtrip_verified'] is True and io['model_roundtrip_verified'] is True
            and io['training_resume_or_public_agent'] is False and io['frozen_snapshots_in_payload'] == envs
            and io['metadata_log_length_each'] == config['rules']['max_rounds'] - 1, 'full IO coverage differs')
    rates = dict(collection_per_record=max(collection), actor_per_record=actor_max / (2 * envs),
                 snapshot=samples(actors['snapshot_copy_seconds'], envs, 'snapshot copy'),
                 q_update=samples(kernel['q_seconds'], iterations, 'Q kernel'),
                 supervised_update=samples(kernel['supervised_seconds'], iterations, 'supervised kernel'),
                 target_copy=samples(kernel['target_copy_seconds'], max(2, min(iterations, 8)), 'target copy'),
                 checkpoint_write=positive(io['full_checkpoint_write_seconds'], 'checkpoint write'),
                 checkpoint_restore=positive(io['full_checkpoint_restore_seconds'], 'checkpoint restore'),
                 baseline_eval_round=samples(evaluation['baseline_seconds_per_round'], 16, 'baseline evaluation'),
                 neural_eval_round=samples(evaluation['neural_seconds_per_round'], 4, 'neural evaluation'),
                 public_archive=positive(io['public_payload_seconds'], 'public archive'),
                 public_model_load=positive(io['public_model_load_seconds'], 'public load'),
                 policy_integrity=samples(evaluation['policy_integrity_seconds'], 2, 'policy integrity'),
                 setup=positive(pipeline['phases']['closed']['setup_seconds'], 'setup'))
    require(set(report['rates_seconds']) == set(RATE_NAMES), 'cost component missing or unrecognized')
    for name, value in rates.items():
        close(report['rates_seconds'][name], value, 'rate ' + name)
    margin = positive(plan['margin'], 'margin'); require(margin >= 1, 'runtime margin below one')
    counts = plan['evaluation_game_budgets_per_seed']; require(counts == report['evaluation_game_budgets_per_seed'], 'evaluation scope differs')
    require(set(counts) == set(STAGES) and all(type(x) is int and x >= 0 for x in counts.values())
            and counts['screening_cumulative'] >= counts['pilot'], 'invalid cumulative evaluation budget')
    forecast = report['forecast']; require(set(forecast['per_seed']) == set(STAGES), 'forecast stages differ')
    computed = {}
    for name, (records, milestones) in STAGES.items():
        # Count update/publication boundaries from zero, not from the offline corpus.
        q = len(range(config['learning_start'] // batch, records // batch + 1)) * config['q_reuse']
        sl = records // batch - 5_000_000 // batch
        copies = records // config['target_every']; checkpoints = (records + plan['checkpoint_every_records'] - 1) // plan['checkpoint_every_records'] + milestones
        terms = {'collection': records * rates['collection_per_record'], 'private_actors': records * rates['actor_per_record'],
                 'frozen_snapshots': ((records + batch - 1) // batch + 1) * rates['snapshot'],
                 'Q': q * rates['q_update'], 'supervised_average': sl * rates['supervised_update'],
                 'target_refresh': copies * rates['target_copy'], 'checkpoints': checkpoints * rates['checkpoint_write'],
                 'one_restore': rates['checkpoint_restore'], 'public_archives': milestones * rates['public_archive'],
                 'evaluation': counts[name] * config['rules']['max_rounds'] * rates['baseline_eval_round'],
                 'evaluation_policy_integrity': 2 * counts[name] * rates['policy_integrity'], 'setup': rates['setup']}
        row = forecast['per_seed'][name]
        require((row['records'], row['q_updates'], row['supervised_update_upper'], row['target_refreshes'], row['full_checkpoint_count_upper'], row['evaluation_games'])
                == (records, q, sl, copies, checkpoints, counts[name]), 'forecast event count differs')
        require(set(row['components_seconds']) == set(terms), 'forecast omitted a component')
        for component, amount in terms.items():
            close(row['components_seconds'][component], amount, 'forecast component ' + component)
        total = math.fsum(terms.values()); hours = total * margin / 3600
        for key, expected in (('projected_seconds_before_margin', total), ('ceiling_seconds', total * margin), ('allocated_process_hours', hours)):
            close(row[key], expected, name + ' ' + key)
        cuda = config['device'].split(':')[0] == 'cuda'
        if cuda: close(row['allocated_single_device_hours'], hours, name + ' device hours')
        else: require(row['allocated_single_device_hours'] is None, 'CPU projection became device hours')
        computed[name] = hours
    scientific = margin * (19_200_000 * rates['neural_eval_round'] + 768 * rates['actor_per_record']
                           + 192_512 * rates['policy_integrity'] + 12 * rates['public_model_load'])
    close(forecast['scientific_and_probe_seconds_with_margin'], scientific, 'science/probe forecast')
    for term, stage in (('H_P', 'pilot'), ('H_D', 'screening_cumulative'), ('H_N', 'confirmation')):
        if cuda: close(forecast[term], computed[stage], term)
        else: require(forecast[term] is None, 'CPU forecast supplied ' + term)
    if cuda: close(forecast['scientific_and_probe_device_hours'], scientific / 3600, 'science/probe device hours')
    else: require(forecast['scientific_and_probe_device_hours'] is None, 'CPU science price')
    require(forecast['concurrency'] == 1 and forecast['margin'] == margin, 'unsupported amortization or margin')
    return {'rates_seconds': rates, 'per_seed_allocated_process_hours': computed,
            'scientific_and_probe_allocated_process_hours': scientific / 3600}


def production_shape(report):
    plan = report['plan']; c = plan['runtime']
    return bool(plan['diagnostic'] is False and c['device'].split(':')[0] == 'cuda'
                and (c['width'], c['blocks'], c['norm_groups'], c['capacity'], c['batch'], c['learning_start'], c['target_every'])
                    == (64, 8, 8, 500_000, 512, 4096, 12800)
                and c['rules'] == {'size': 9, 'max_rounds': 200, 'collision': 'gray'}
                and c['augmentation'] is True and c['deterministic'] is True and c['native_library'] is not None
                and c['reservoir_open'] == 5_000_000 and c['handoff_end'] == 7_000_000 and c['uniform_comparison'] is False
                and plan['records_per_pipeline_phase'] >= 25600 and plan['kernel_iterations_each'] >= 32
                and plan['kernel_warmup_each'] >= 4 and plan['actor_iterations_each'] >= 4 and plan['dense_collection_records'] >= 256
                and plan['target'] == 'one-step' and c['q_reuse'] in (1, 4)
                and report['full_buffers']['replay_size'] == report['full_buffers']['reservoir_size'] == 500_000)


def _seal(path, inventory):
    from graygo.storage import CheckpointStore, CorruptCheckpoint
    path = Path(path).resolve()
    try:
        manifest = CheckpointStore.verify(path)
    except CorruptCheckpoint as error:
        raise ValueError(str(error)) from error
    for item in ['manifest.json', 'manifest.sha256', *manifest['files']]:
        file = path / item; inventory[str(file)] = sha(file)
    return manifest


def audit(root, repo, allocation, tool, *, expected_source, expected_tool, expected_allocation):
    import numpy as np
    import torch
    from graygo.buffers import ReplayBuffer, ReservoirBuffer
    from graygo.manifest import source_digest
    from graygo.models import ModelConfig, QNet, PolicyNet
    torch.set_num_threads(1)
    root = Path(root).resolve(); repo = Path(repo).resolve(); allocation = Path(allocation).resolve(); tool = Path(tool).resolve()
    require(sha(allocation) == expected_allocation and sha(tool) == expected_tool and source_digest() == expected_source, 'caller-pinned source/allocation differs')
    inventory = {str(p): sha(p) for p in [allocation, tool, root/'driver.py', root/'job.json', root/'plan.json', root/'run.json', root/'report.json', root/'pipeline/report.json']}
    a = load(allocation); job = load(root/'job.json'); report = load(root/'report.json'); plan = report['plan']; manifest = load(root/'run.json')
    require(job['status'] == 'complete_measurement_unreviewed' and job['returncode'] == 0 and job['original_inputs_unchanged'] is True, 'original measurement not terminal and successful')
    require(job['allocation_sha256'] == expected_allocation and job['tool_sha256'] == plan['tool_sha256'] == expected_tool
            and job['driver_sha256'] == sha(root/'driver.py') and job['report_sha256'] == sha(root/'report.json'), 'original job bindings differ')
    require(plan == load(root/'plan.json') == manifest['command'] and manifest['git_dirty'] is False
            and manifest['source_sha256'] == plan['source_sha256'] == job['learner_source_sha256'] == a['learner_source_sha256'] == expected_source, 'plan/source manifest differs')
    require(manifest['git_commit'] == a['learner_source_commit'] and job['physical_device_uuid'] == a['physical_device_uuid'], 'source commit/device binding differs')
    require(job['production_training_selected'] is False and a['production_learning_selected'] is False
            and a['pilot_selected'] is False and a['confirmation_selected'] is False, 'measurement allocation is not isolated')
    for key in ('formal_learning_run', 'a2_complete', 'execution_authorized'):
        require(report[key] is False, 'profile claims ' + key)
    require(production_shape(report) and report['candidate_price_forecast_available'] is True and report['full_component_coverage'] is True, 'not the full selected CUDA profile')
    c = plan['runtime']
    for field, expected in [('selected_reuse', c['q_reuse']), ('target', plan['target']), ('device', c['device']),
                            ('reference_pipeline_records_per_phase', plan['records_per_pipeline_phase']), ('dense_collection_records', plan['dense_collection_records']),
                            ('total_new_profile_own_records', report['fresh_profile_records']), ('kernel_iterations_each', plan['kernel_iterations_each']),
                            ('kernel_warmup_updates_each', plan['kernel_warmup_each']), ('actor_iterations_each', plan['actor_iterations_each']),
                            ('forecast_margin', plan['margin']), ('process_time_ceiling_seconds', plan['time_ceiling_seconds']),
                            ('disk_reserve_bytes', plan['reserve_bytes']), ('outer_watchdog_seconds', job['outer_watchdog_seconds'])]:
        require(a[field] == expected, 'allocation mismatch: ' + field)
    span = Decimal(str(job['allocation_interval_end_epoch'])) - Decimal(str(job['allocation_interval_start_epoch']))
    require(span > 0 and abs(span - Decimal(str(job['measured_driver_seconds']))) < Decimal('.01')
            and job['measured_driver_seconds'] <= a['outer_watchdog_seconds'] and report['elapsed_seconds'] <= a['process_time_ceiling_seconds'], 'invalid original allocation interval/cap')
    source = repo / a['buffer_checkpoint']; native = repo / a['native_library']; benchmark = repo / a['dense_benchmark']
    require(str(native.resolve()) == c['native_library'] and sha(native) == a['native_library_sha256'] == plan['native_schema']['native_decoder']['sha256'], 'native decoder differs')
    require(sha(benchmark) == a['dense_benchmark_file_sha256'], 'dense benchmark input differs')
    for p in (native, benchmark): inventory[str(p.resolve())] = sha(p)
    source_seal = _seal(source, inventory)
    require(sha(source/'manifest.json') == plan['input_checkpoint_manifest_sha256'], 'input corpus seal differs')
    pipeline = load(root/'pipeline/report.json')
    require(sha(root/'pipeline/report.json') == report['reference_pipeline_report_sha256'], 'pipeline reference differs')
    for phase in pipeline['phases'].values():
        checkpoint = Path(phase['checkpoint']).resolve(); require(checkpoint.is_relative_to(root/'pipeline'), 'pipeline checkpoint escaped run')
        _seal(checkpoint, inventory)
    io_path = (root/'full-io/weights'/f"{c['capacity']+plan['dense_collection_records']:012}").resolve()
    require(Path(report['io']['checkpoint']).resolve() == io_path and not (io_path/'runtime.pt').exists() and not (io_path/'models.pt').exists(), 'cost IO became training artifacts')
    seal = _seal(io_path, inventory)
    require(sha(io_path/'manifest.json') == report['io']['checkpoint_manifest_sha256'] and set(seal['files']) == {'cost-payload.pt', 'replay.npz', 'reservoir.npz'}
            and sum(x['bytes'] for x in seal['files'].values()) == report['io']['payload_bytes'], 'full IO seal/size differs')
    r = ReplayBuffer.load(io_path/'replay.npz'); s = ReservoirBuffer.load(io_path/'reservoir.npz'); original = ReplayBuffer.load(source/'replay.npz')
    require(r.size == s.size == original.size == c['capacity'] and r.insertions == c['capacity'] + plan['dense_collection_records']
            and r.allocated_bytes == report['full_buffers']['replay_allocated_bytes'] and s.allocated_bytes == report['full_buffers']['reservoir_allocated_bytes'], 'full buffer payload sizes differ')
    require(r.spec.to_dict() == s.spec.to_dict() == original.spec.to_dict() == plan['native_schema'] == source_seal['metadata']['schema'], 'buffer rule/codec schemas differ')
    require(r.q_insertions == r.insertions and r.q_size == r.size and r.arrays['value_mask'].all(), 'materialized/dense Q eligibility differs')
    materialized = report['materialized_dataset']
    require(materialized['materialized_rows'] == materialized['source_retained_rows'] == a['input_materialized_rows'] == c['capacity']
            and materialized['new_environment_actions'] == a['new_fresh_records_for_materialization'] == 0
            and materialized['source_original_insertions'] == original.insertions
            and materialized['source_historical_q_count_known'] is (original.q_insertions is not None), 'offline history was invented')
    offset = plan['dense_collection_records']; order = np.argsort(original.arrays['ordinal'][:original.size])
    # Slots before offset contain the actual newly collected dense records;
    # all other data fields must retain their true original sorted transitions.
    for name in original.arrays.keys() - {'ordinal', 'next_index', 'next_ordinal'}:
        require(np.array_equal(r.arrays[name][offset:], original.arrays[name][order[offset:]]), 'offline transition field changed: ' + name)
    require(np.array_equal(r.arrays['ordinal'][:offset], np.arange(c['capacity'], c['capacity'] + offset))
            and np.array_equal(r.arrays['ordinal'][offset:], np.arange(offset, c['capacity'])), 'offline/new record namespaces differ')
    old = s.arrays['record'] < c['capacity']; old_records = s.arrays['record'][old]
    require((old_records >= 0).all() and c['capacity'] <= s.seen <= c['capacity'] + offset, 'supervised materialization/offer counters differ')
    for name in ('spatial', 'globals', 'legal'):
        require(np.array_equal(s.arrays[name][old], original.arrays[name][order[old_records]]), 'offline supervised observation changed')
    expected = np.zeros_like(s.arrays['greedy'][old]); actions = original.arrays['action'][order[old_records]].astype(np.int64)
    expected[np.arange(len(actions)), actions//8] = 1 << (actions%8)
    require(np.array_equal(s.arrays['greedy'][old], expected), 'synthetic executed-action labels differ')
    payload = torch.load(io_path/'cost-payload.pt', map_location='cpu', weights_only=True)
    require(payload['profile_only'] is True, 'cost-only payload flag missing')
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        model_config = ModelConfig(c['width'], c['blocks'], c['norm_groups'])
        models = {'q': QNet(model_config), 'target': QNet(model_config), 'average': PolicyNet(model_config)}
    for name, model in models.items():
        model.load_state_dict(payload[name], strict=True)
        require(all(torch.isfinite(value).all().item() for value in payload[name].values()), 'nonfinite model payload')
    for name in ('q_optimizer', 'average_optimizer'):
        require(len(payload[name]['state']) == len(list(models[name.removesuffix('_optimizer')].parameters()))
                and all(int(x['step'].item()) == plan['kernel_iterations_each'] + plan['kernel_warmup_each'] for x in payload[name]['state'].values()), 'saved optimizer steps differ')
    frozen = payload['frozen_snapshot_stress']; logs = payload['maximum_length_log_metadata_stress']
    require(set(frozen) == set(range(c['envs'])) and len(logs) == c['envs'] and all(len(events) == c['rules']['max_rounds'] - 1 for events in logs), 'snapshot/log IO stress missing')
    pointers = set()
    for index, snapshot in frozen.items():
        require(snapshot['version'] == 3_000_000 + index and snapshot['average_ready'] is True, 'frozen snapshot metadata differs')
        for name in ('q', 'average'):
            require(set(snapshot[name]) == set(payload[name]), 'snapshot parameter layout differs')
            for key, tensor in snapshot[name].items():
                require(torch.equal(tensor, payload[name][key]) and tensor.untyped_storage().data_ptr() not in pointers, 'snapshot payload aliases or differs')
                pointers.add(tensor.untyped_storage().data_ptr())
    require(len({id(event['actions']) for events in logs for event in events}) == c['envs'] * (c['rules']['max_rounds'] - 1), 'log lists alias and undercount IO')
    public_path = root/'full-io/cost-public-payload.pt'; inventory[str(public_path)] = sha(public_path)
    public = torch.load(public_path, map_location='cpu', weights_only=True)
    require(public['profile_only'] is True and all(set(public[name]) == set(payload[name])
            and all(torch.equal(value, payload[name][key]) for key, value in public[name].items()) for name in ('q', 'average')), 'public-load timing payload differs')
    computed = recompute(report, pipeline)
    memory = report['memory']; require(0 < memory['peak_cuda_allocated_bytes'] <= memory['peak_cuda_reserved_bytes'] <= a['reported_vram_mib'] * 1024**2, 'memory profile exceeds declared device')
    violations = []
    if computed['per_seed_allocated_process_hours']['pilot'] > 24: violations.append('H_P exceeds initial 24-device-hour per-seed pilot ceiling')
    if computed['scientific_and_probe_allocated_process_hours'] > 16: violations.append('scientific/probe forecast exceeds 16-device-hour component')
    require(source_digest() == expected_source and all(sha(path) == value for path, value in inventory.items()), 'inputs changed during audit')
    return {'version': VERSION, 'status': 'verified_empirical_accounting', 'source_sha256': expected_source, 'producer_sha256': expected_tool,
            'auditor_sha256': sha(Path(__file__)), 'input_files_sha256': inventory, 'original_inputs_unchanged': True,
            'original_driver_pid': job['original_driver_pid'], 'original_profile_pid': job['profile_pid'], 'physical_device_uuid': job['physical_device_uuid'],
            'allocation_interval_start_epoch': job['allocation_interval_start_epoch'], 'allocation_interval_end_epoch': job['allocation_interval_end_epoch'],
            'physical_allocation_seconds': str(span), 'physical_allocation_device_hours': str(span/3600),
            'independent_calculation': computed, 'memory': memory, 'component_limit_violations': violations,
            'empirical_component_forecast_verified': True, 'selected_scope_qualified': False, 'runtime_bound_proven': False,
            'a2_complete': False, 'execution_authorized': False,
            'scope': 'source/device-bound empirical forecast and retained artifact accounting; no rerun, learned strength, whole-project consumption or scope approval'}


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('run-dir', 'repo', 'allocation', 'producer', 'expected-source-sha', 'expected-tool-sha', 'expected-allocation-sha', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv); started = time.monotonic()
    try:
        root = Path(args.run_dir).resolve(); output = Path(args.output_dir).resolve()
        require(not output.exists() and not output.is_relative_to(root), 'audit output must be new and outside original run')
        with (root/'.cost-profile.lock').open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            report = audit(root, args.repo, args.allocation, args.producer, expected_source=args.expected_source_sha,
                           expected_tool=args.expected_tool_sha, expected_allocation=args.expected_allocation_sha)
            report['audit_process_seconds'] = time.monotonic() - started
            output.mkdir(parents=True, exist_ok=False)
            (output/'audit.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
        print(json.dumps({key: report[key] for key in ('status', 'component_limit_violations', 'physical_allocation_device_hours', 'audit_process_seconds')}, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, str(error)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())

"""Measure the uniform-fallback reference and conservatively price its 8M runs.

Only compatible common update/IO/evaluation measurements are reused. Actual
uniform pipeline, dense collection and private actor work are measured afresh.
"""
import argparse
from dataclasses import replace
import fcntl
import gc
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from graygo.board_runtime import BoardNFSPRuntime, BoardRuntimeConfig, configure_runtime
from graygo.buffers import ReplayBuffer, ReservoirBuffer
from graygo.engine import Rules
from graygo.evaluation import seed_for
from graygo.frozen_forward import FrozenForwardPool
from graygo.manifest import source_digest, make_manifest, write_report
from graygo.models import QNet, PolicyNet
from graygo.nfsp import NFSPBackend
from graygo.nfsp_collection import FrozenPolicies, EpisodeActor, commit_many
from graygo.rule_state import RuleObservation
from graygo.schedules import EpisodeChoice
from graygo.storage import CheckpointStore, DEFAULT_RESERVE
from tools import nfsp_cost_profile as core, nfsp_graph_costs as graph, nfsp_uniform_pipeline as pipeline

VERSION = 'graygo/uniform-reference-cost-measurement/1'


class ClosedReservoirSink(core._ReservoirSink):
    """Closed local timing view over an already populated offline corpus."""
    def __init__(self, reservoir, base, opening_record):
        super().__init__(reservoir, base)
        self.opening_record = opening_record

    def offer_index(self, *args, **kwargs):
        raise ValueError('closed dense timing must not insert labels into the offline corpus')


def require(condition, message):
    if not condition:raise ValueError(message)


def sha(path):
    return core.digest(Path(path))


def load(path):
    return json.loads(Path(path).read_text())


def compatible_config(base):
    data = dict(base); data['rules'] = Rules(**data['rules']); original = BoardRuntimeConfig(**data)
    require(not original.uniform_comparison and original.cuda_graphs and original.device.split(':')[0] == 'cuda'
            and original.rules == Rules() and original.q_reuse == 4, 'qualified primary single-GPU R4 graph configuration required')
    reference = replace(original, uniform_comparison=True)
    require(reference.backend_config() == original.backend_config() and reference.model_config() == original.model_config()
            and reference.schedule() == original.schedule(), 'uniform reference changed shared learning/model/schedule components')
    return reference


def checked_base(repo, declaration):
    root = repo / declaration['base_root']; audit_path = repo / declaration['base_audit']
    require(sha(audit_path) == declaration['base_audit_sha256'], 'original independent graph audit changed')
    audit = load(audit_path)
    require(audit['status'] == 'verified_empirical_accounting' and audit['source_sha256'] == declaration['learner_source_sha256']
            and audit['original_inputs_unchanged'] and audit['empirical_component_forecast_verified'], 'qualified common component audit required')
    require(all(sha(path) == value for path, value in audit['input_files_sha256'].items()), 'qualified common component inputs changed')
    require(sha(root / 'report.json') == declaration['base_report_sha256'], 'original measured graph components changed')
    base = load(root / 'report.json'); config = compatible_config(base['plan']['runtime'])
    require(source_digest() == declaration['learner_source_sha256'] == base['plan']['source_sha256'], 'reference must use the original measured source package')
    require(sha(core.__file__) == base['plan']['core_producer_sha256']
            and sha(graph.__file__) == base['plan']['tool_sha256'], 'compatible original measurement helpers changed')
    require(sha(repo / declaration['dense_benchmark']) == declaration['dense_benchmark_file_sha256'], 'qualified independent dense benchmark changed')
    return root, base, audit, config


def pipeline_measurements(config, root, *, records, time_cap_seconds, check_budget=lambda: None):
    import graygo.board_collection as collection
    original_class = pipeline.BoardNFSPRuntime; original_commit = collection.commit_many
    measurements = []; active = [None]
    class Instrumented(original_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.observer = {'actor_seconds': 0., 'snapshot_copy_seconds': 0., 'snapshot_copies': 0}
            measurements.append(self.observer)
        def _snapshot(self):
            version = self.backend.counters.q_updates + self.backend.counters.supervised_updates
            changed = self.snapshot is None or self.snapshot.version != version
            result, seconds = core.timed(self.backend.device, lambda: super(Instrumented, self)._snapshot())
            if changed:
                self.observer['snapshot_copy_seconds'] += seconds; self.observer['snapshot_copies'] += 1
            return result
        def advance(self, *args, **kwargs):
            check_budget(); active[0] = self
            try:return super().advance(*args, **kwargs)
            finally:active[0] = None
    def observed(requests):
        require(active[0] is not None, 'uniform actor timing escaped its actual collector')
        result, seconds = core.timed(active[0].backend.device, lambda: original_commit(requests))
        active[0].observer['actor_seconds'] += seconds; return result
    pipeline.BoardNFSPRuntime = Instrumented; collection.commit_many = observed
    try:
        result = pipeline.profile_board(config, root, records=records, time_cap_seconds=time_cap_seconds)
    finally:
        pipeline.BoardNFSPRuntime = original_class; collection.commit_many = original_commit
    require(result['status'] == 'complete' and len(measurements) == 2, 'both complete actual uniform phases required')
    for (phase, data), observed in zip(result['phases'].items(), measurements):
        residual = data['timings']['collection'] - observed['actor_seconds'] - observed['snapshot_copy_seconds']
        require(residual >= -.001, 'uniform actor/snapshot scopes overlap collection')
        observed.update(phase=phase, records=data['records'], collection_seconds=data['timings']['collection'],
                        resolver_storage_bookkeeping_seconds=max(0., residual))
    return result, measurements


def uniform_actors(config, backend, states, tracker, *, iterations, check_budget=lambda: None):
    pool = FrozenForwardPool(config.device); snapshots = []; copies = []
    for index in range(config.envs):
        check_budget()
        item, seconds = core.timed(backend.device, lambda index=index: FrozenPolicies(backend.q, backend.average,
            version=3000000 + index, average_ready=True, forward_pool=pool))
        snapshots.append(item); copies.append(seconds)
    cases = {}
    for layout in ('shared_uniform', 'distinct_q_uniform'):
        actors = []; samples = []
        for env in range(config.envs):
            for role in (1, 2):
                kind = 'q' if layout == 'distinct_q_uniform' and role == 1 else 'uniform'
                choice = EpisodeChoice(0, kind == 'q', kind, .08875, 0., 0. if kind == 'q' else .5, .5)
                actors.append(EpisodeActor(snapshots[env] if layout == 'distinct_q_uniform' else snapshots[0],
                    choice, np.random.default_rng(seed_for(config.seed, 'uniform-cost-actor', env, role))))
        for _ in range(iterations):
            check_budget(); first = tracker.mark()
            requests = [(actor, RuleObservation(states[i // 2], i % 2 + 1), actor.choice.best_response_mode) for i, actor in enumerate(actors)]
            values, raw = core.timed(backend.device, lambda: commit_many(requests))
            last = tracker.mark(); setup = tracker.seconds_between(first, last)
            require(raw >= setup - .000001 and len(values) == 2 * config.envs, 'uniform actor setup/count scopes differ')
            require(all(value.action in states[i // 2].legal_actions(i % 2 + 1) for i, value in enumerate(values)), 'uniform benchmark chose an illegal move')
            samples.append({'raw_seconds': raw, 'steady_seconds': max(0., raw - setup),
                            'graph_setup_seconds': setup, 'graph_event_span': [first, last]})
        cases[layout] = samples
    return {'cases': cases, 'requests_per_sample': 2 * config.envs, 'snapshot_copy_seconds': copies,
            'pool_stats': pool.stats(), 'scope': 'actual uniform and mixed-Q/uniform private commitments; no NFSP fresh clock fabricated'}


def dense_uniform(config, backend, replay, reservoir, states, *, records, check_budget=lambda: None):
    import graygo.board_collection as collection
    from graygo.board_collection import BoardCollector
    from graygo.evaluation import FixedStarts
    collector = BoardCollector(config.rules, config.envs, config.seed + 1801,
        starts=FixedStarts(states, label='uniform-cost-only-complete-public-roots'))
    sink = core._ReplaySink(replay); labels = ClosedReservoirSink(reservoir, sink.base, config.reservoir_open)
    rng = np.random.default_rng(config.seed + 1802)
    snapshot = FrozenPolicies(backend.q, backend.average, version=4000000, average_ready=True, forward_pool=FrozenForwardPool(config.device))
    schedule = config.schedule(); original = collection.commit_many; actor_seconds = [0.]
    def observed(requests):
        value, seconds = core.timed(backend.device, lambda: original(requests)); actor_seconds[0] += seconds; return value
    collection.commit_many = observed; core._synchronize(backend.device); started = time.perf_counter()
    try:
        while collector.records < records:
            check_budget()
            collector.collect(min(2 * config.envs, records - collector.records), snapshot, sink, labels, rng, schedule, uniform_comparison=True)
    finally:collection.commit_many = original
    core._synchronize(backend.device); elapsed = time.perf_counter() - started
    require(sink.insertions == collector.records == records and replay.size == reservoir.size == config.capacity,
            'actual uniform dense overwrite accounting differs')
    require(collector.metrics['acting_counts']['uniform'] > 0 and collector.metrics['acting_counts']['average'] == 0,
            'closed uniform dense measurement did not exercise the fallback')
    return {'new_profile_records': records, 'total_seconds': elapsed, 'actor_seconds': actor_seconds[0],
            'resolver_storage_bookkeeping_seconds': max(0., elapsed - actor_seconds[0]),
            'acting_counts': dict(collector.metrics['acting_counts']), 'local_clock_below_opening': True}


def reference_forecast(base, measured, setup, *, seeds=(100, 101, 102)):
    require(tuple(seeds) == (100, 101, 102), 'three matched original development seeds required')
    old = base['forecast']['per_seed']['pilot']; n = 8000000
    require(old['records'] == n and old['evaluation_games'] == 36000 and old['evaluation_endpoints'] == 18,
            'matched 8M/six-window reference evaluation scope differs')
    rates = base['core_report']['rates']; copies = measured['actors']['snapshot_copy_seconds']
    residual = max(r['resolver_storage_bookkeeping_seconds'] / r['records'] for r in measured['pipeline_observers'])
    dense = measured['dense']; collection = max(rates['collection_per_record'], residual, dense['resolver_storage_bookkeeping_seconds'] / dense['new_profile_records'])
    actor = max(base['actors']['steady_actor_seconds_per_record'],
                max(s['steady_seconds'] for values in measured['actors']['cases'].values() for s in values) / measured['actors']['requests_per_sample'])
    snapshot = max(max(base['actors']['snapshot_copy_seconds']), max(copies))
    capture = max(base['graph_setup']['capture_max_seconds'], setup['capture_max_seconds'])
    pack = max(base['graph_setup']['weight_pack_max_seconds'], setup['weight_pack_max_seconds'])
    components = dict(old['components_seconds'])
    components.update(collection=n * collection, private_actors=n * actor,
        frozen_snapshots=old['snapshot_publication_upper'] * snapshot,
        frozen_graph_captures=old['graph_capture_upper'] * capture,
        frozen_parameter_packs=old['weight_pack_upper'] * pack,
        setup=max(components['setup'], max(p['setup_seconds'] for p in measured['pipeline']['phases'].values())))
    total = math.fsum(components.values()); ceiling = 2 * total
    return {'scope': 'three separate matched uniform-fallback development references through 8M, fixed 5M opening/5M-7M ramp and full diagnostic windows; no performance filtering',
        'seeds': list(seeds), 'records_per_seed': n, 'aggregate_reference_records': 3 * n,
        'evaluation_games_per_seed': 36000, 'evaluation_endpoints_per_seed': 18,
        'components_seconds_per_seed': components, 'projected_seconds_before_margin': total,
        'margin': 2., 'per_seed_ceiling_seconds': ceiling, 'H_U_device_hours_per_seed': ceiling / 3600,
        'B_U_board_device_hours': 3 * ceiling / 3600,
        'compatible_common_component_reuse': ['Q', 'supervised_average', 'target_refresh', 'checkpoints', 'one_restore',
            'public_archives', 'evaluation_groups', 'evaluation_endpoint_metadata', 'evaluation_policy_integrity',
            'evaluation_model_loads', 'cold_process_startups'],
        'A2_uniform_subset_CPU_price_qualified': False, 'full_B_U_scope_qualified': False,
        'runtime_bound_proven': False, 'execution_authorized': False,
        'limits': 'Compatible common kernels/IO/evaluation retained from independently audited identical source/device/model/cadence; collection/actor/snapshot/setup rates use larger original/current observations. Finite untrained phases and empirical twofold envelope are not a runtime theorem or convergence claim. A separately declared A2 subset and applicable CPU ceilings remain required.'}


def _profile(repo, allocation, expected_sha, root):
    repo = Path(repo).resolve(); allocation = Path(allocation).resolve(); root = Path(root).resolve()
    require(sha(allocation) == expected_sha, 'declared uniform profile allocation changed')
    declaration = load(allocation)
    require(sha(__file__) == declaration['producer_sha256'] and sha(pipeline.__file__) == declaration['pipeline_producer_sha256'], 'frozen uniform producers changed')
    _, base, base_audit, config = checked_base(repo, declaration)
    started = time.monotonic(); torch.set_num_threads(1)
    settings = configure_runtime(config.device, True)
    base_pipeline = load(repo / declaration['base_root'] / 'core/pipeline/report.json')
    require(settings == base_pipeline['phases']['closed']['settings'], 'uniform profile numerical execution differs from compatible common components')
    def check():
        require(time.monotonic() - started < declaration['process_time_ceiling_seconds'], 'original uniform profile time allocation exhausted')
    plan = {'version': VERSION, 'runtime': config.plan(), 'source_sha256': source_digest(), 'producer_sha256': sha(__file__),
        'pipeline_producer_sha256': sha(pipeline.__file__), 'allocation_sha256': expected_sha,
        'base_root': str(repo / declaration['base_root']), 'base_audit_sha256': sha(repo / declaration['base_audit']),
        'records_per_phase': declaration['records_per_phase'], 'dense_records': declaration['dense_records'],
        'actor_iterations': declaration['actor_iterations'], 'time_ceiling_seconds': declaration['process_time_ceiling_seconds'],
        'settings': settings, 'new_formal_A2_records': 0, 'production_learning_selected': False, 'scope': 'bounded hardware diagnostics and compatible-component envelope only'}
    manifest = make_manifest(seed=config.seed, command=plan, rules=config.plan()['rules'])
    original_manifest = load(repo / declaration['base_root'] / 'run.json')
    for key in ('hostname', 'cpu', 'python', 'platform', 'dependencies'):
        require(manifest[key] == original_manifest[key], 'uniform/common component host or numerical dependencies differ: ' + key)
    write_report(root / 'plan.json', plan); write_report(root / 'run.json', manifest)
    with graph.GraphSetupObserver().installed() as tracker:
        measured_pipeline, observers = pipeline_measurements(config, root / 'pipeline', records=declaration['records_per_phase'],
            time_cap_seconds=declaration['process_time_ceiling_seconds'] - (time.monotonic() - started), check_budget=check)
        write_report(root / 'pipeline-observers.json', observers)
        check(); io_root = Path(base['graph_io']['checkpoint'])
        payload = torch.load(io_root / 'cost-payload.pt', map_location='cpu', weights_only=True)
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            backend = NFSPBackend(QNet(config.model_config()), PolicyNet(config.model_config()), config.backend_config(), device=config.device)
        for name in ('q', 'target', 'average'):getattr(backend, name).load_state_dict(payload[name])
        del payload
        replay = ReplayBuffer.load(io_root / 'replay.npz'); reservoir = ReservoirBuffer.load(io_root / 'reservoir.npz')
        require(replay.size == reservoir.size == config.capacity, 'original compatible full buffers required')
        states = core.sample_states(replay, config.envs, np.random.default_rng(config.seed + 1901))
        actors = uniform_actors(config, backend, states, tracker, iterations=declaration['actor_iterations'], check_budget=check)
        write_report(root / 'actors.json', actors)
        from graygo.probe_benchmark import ProbeBenchmark
        benchmark = ProbeBenchmark.from_dict(load(repo / declaration['dense_benchmark']))
        require(benchmark.family_coverage_complete and benchmark.sha256 == base['core_report']['plan']['dense_benchmark_sha256'],
                'uniform collection must use the exact independently qualified public roots')
        dense_states = tuple(node.state for family in benchmark.families for node in (family.before, family.after))
        dense = dense_uniform(config, backend, replay, reservoir, dense_states, records=declaration['dense_records'], check_budget=check)
        setup = tracker.report()
    measured = {'pipeline': measured_pipeline, 'pipeline_observers': observers, 'actors': actors, 'dense': dense}
    forecast = reference_forecast(base, measured, setup); check()
    require(all(sha(p) == h for p, h in base_audit['input_files_sha256'].items()), 'reused original graph inputs changed')
    result = {'version': VERSION, 'status': 'complete_measurement_pending_independent_audit', 'plan': plan,
        'measurements': measured, 'graph_setup': setup, 'forecast': forecast,
        'new_diagnostic_learner_records': 2 * declaration['records_per_phase'] + declaration['dense_records'],
        'elapsed_seconds': time.monotonic() - started, 'memory': core._memory(torch.device(config.device)),
        'a2_complete': False, 'execution_authorized': False, 'formal_learning_run': False}
    write_report(root / 'report.json', result)
    print(json.dumps({'status': result['status'], 'B_U_board_device_hours': forecast['B_U_board_device_hours']}))


def profile(repo, allocation, expected_sha, root):
    root = Path(root).resolve(); require(not root.exists(), 'preserve the original profile root and handle')
    root.mkdir(parents=True)
    with (root / '.uniform-cost.lock').open('xb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _profile(repo, allocation, expected_sha, root)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--repo', required=True); parser.add_argument('--allocation', required=True)
    parser.add_argument('--expected-allocation-sha', required=True); parser.add_argument('--run-dir', required=True)
    args = parser.parse_args(); profile(args.repo, args.allocation, args.expected_allocation_sha, args.run_dir)


if __name__ == '__main__':main()

"""Source-bound CPU prices and fixed original neural Gray Go task cases."""
import argparse
from contextlib import ExitStack
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import signal
import time
from graygo.manifest import source_digest, write_report
from graygo.storage import CheckpointStore, digest as _digest
from graygo.task_training import NFSPTaskConfig, NFSPTaskTrainer
from tools import cpu_cache_admission, nfsp_control_budget
from tools.a2_source_cohort import read_registry
from tools.nfsp_delayed_cpu_prices import IDENTITY_KEYS, runtime_identity
from tools.nfsp_resources import number, require
from tools.run_nfsp_cpu_delayed_case import fresh_case_directory, retain_initialization

VERSION = 'graygo/gray-cpu-runtime-prices/1'
TASKS = ('gray-pure', 'gray-mixed')


def digest(path):
    return _digest(Path(path))


def load(path):
    return json.loads(Path(path).read_text())


def qualify(repo, specification):
    started = time.monotonic(); repo = Path(repo).resolve(); specification = Path(specification).resolve()
    spec = load(specification)
    require(spec['tasks'] == list(TASKS) and spec['learner'] == 'neural' and spec['q_reuse'] == 4,
            'only both primary neural Gray Go task prices are in scope')
    paths = {k: repo / spec[k] for k in ('source_admission', 'source_registry', 'own_source_plan')}
    require(all(digest(p) == spec[k + '_sha256'] for k, p in paths.items()), 'registered source evidence changed')
    admission = load(paths['source_admission']); source = admission['candidate_source_sha256']
    require(admission['version'] == cpu_cache_admission.VERSION
            and admission['status'] == 'qualified_cpu_3x3_raw_cache'
            and source == spec['learner_source_sha256'], 'qualified raw-cache candidate required')
    require(cpu_cache_admission.package_sha(repo / spec['source_package'] / 'graygo') == source,
            'priced frozen package changed')
    registry, _, plans, edges = read_registry(paths['source_registry'], repo=repo, current=True)
    require(any(e['proof'] == admission for e in edges), 'exact cache receipt is absent from registry')
    own = [p for p in plans.values() if p.learner_source_sha256 == source]
    require(len(own) == 1 and own[0].to_dict() == load(paths['own_source_plan']), 'original own-source plan changed')
    comparison_root = Path(admission['evidence']['comparison_root'])
    audits_root = Path(admission['evidence']['audit_root'])
    inputs = dict(admission['input_files_sha256'])
    inputs.update({str(specification): digest(specification), **{str(p): digest(p) for p in paths.values()},
                   str(Path(nfsp_control_budget.__file__).resolve()): digest(nfsp_control_budget.__file__)})
    prices = {}; identities = []
    folders = [comparison_root / 'profiles' / task / ('neural-R4-' + phase)
               for task in TASKS for phase in ('pre', 'post')]
    with ExitStack() as stack:
        for folder in sorted(folders):
            fcntl.flock(stack.enter_context((folder / '.trainer.lock').open('rb')), fcntl.LOCK_SH | fcntl.LOCK_NB)
        for task in TASKS:
            profile_root = comparison_root / 'profiles' / task
            profile = load(profile_root / 'profile.json'); plan = load(profile_root / 'plan.json')
            budget_path = profile_root / 'budgets/neural-R4.json'; budget = load(budget_path)
            require(load(profile_root / 'state.json')['status'] == 'complete' and profile['plan'] == plan
                    and plan['source_sha256'] == source and len(profile['results']) == 2,
                    'original complete candidate profile pair required')
            rows = profile['results']; configs = {}
            require(budget['profile_cases'] == rows and budget['source_sha256'] == source
                    and budget['profile_plan_sha256'] == digest(profile_root / 'plan.json')
                    and budget['complete_pipeline'] is True and budget['device'] == 'cpu'
                    and budget['storage'] == 'features' and budget['native_library_sha256'] is None
                    and budget['target'] == 'one-step' and budget['q_reuse'] == 4 and budget['margin'] == 2.,
                    'raw budget differs from complete CPU/features candidate pipeline')
            for row in rows:
                phase = row['phase']; folder = profile_root / row['case']; key = task + '/' + phase
                manifest = load(folder / 'run.json'); result = load(folder / 'result.json')
                audit_path = audits_root / (key.replace('/', '-') + '.json'); audit = load(audit_path)
                bound = admission['independent_actual_prefix_audits'][key]
                require(folder.resolve() == Path(bound['original_root']).resolve()
                        and digest(audit_path) == bound['audit_sha256'] and audit['verified'] is True
                        and audit['source_sha256'] == source
                        and audit['training_manifest_sha256'] == digest(folder / 'run.json')
                        and audit['result_sha256'] == digest(folder / 'result.json')
                        and audit['episode_audit_sha256'] == digest(folder / 'episodes.bin'),
                        'independent actual prefix binding changed')
                bundle = CheckpointStore(folder).latest(); CheckpointStore.verify(bundle)
                require(audit['checkpoint_manifest_sha256'] == digest(bundle / 'manifest.json'),
                        'independent checkpoint binding changed')
                require(manifest['source_sha256'] == result['source_sha256'] == source
                        and audit['task'] == task and audit['learner'] == 'neural' and audit['seed'] == 0
                        and not audit['formal_protocol'] and result['status'] == 'complete'
                        and row['records'] == result['records'] == audit['records']
                        and row['counters'] == result['counters']
                        and row['acting_policy_counts'] == result['acting_policy_counts'],
                        'actual completed profile work differs from timing rows')
                require(abs(number(row['seconds_including_setup']) - number(result['elapsed_seconds'])
                            - number(result['timings']['setup'])) < Decimal('.00000001'),
                        'profile timing omitted setup or loop')
                configs[phase] = manifest['config']; identities.append({k: manifest[k] for k in IDENTITY_KEYS})
            price = nfsp_control_budget.price_pair(task, 'neural', 4, rows, configs)
            require(number(price['cpu_process_ceiling_seconds']) == number(budget['proposed_time_ceiling_seconds'])
                    and abs(number(price['phase_weighted_15M_seconds'])
                            - number(budget['phase_weighted_15M_seconds'])) < Decimal('.000001'),
                    'independently recomputed twofold price differs')
            prices[task] = price | {'source_sha256': source, 'raw_budget': str(budget_path),
                                   'raw_budget_sha256': digest(budget_path), 'forecast_limits': budget['limits']}
            for p in (profile_root / 'profile.json', profile_root / 'plan.json',
                      profile_root / 'state.json', budget_path):
                inputs[str(p)] = digest(p)
        require(all(identity == identities[0] for identity in identities) and runtime_identity() == identities[0],
                'current host/numerical/runtime identity differs from measured profiles')
        require(all(digest(p) == h for p, h in inputs.items()), 'price inputs changed during qualification')
    return {'version': VERSION, 'status': 'qualified_cpu_gray_neural_prices', 'prices': prices,
            'learner_source_sha256': source, 'runtime_identity': identities[0],
            'verifier_sha256': digest(__file__), 'input_files_sha256': inputs,
            'source_admission_sha256': digest(paths['source_admission']),
            'source_registry_sha256': digest(paths['source_registry']),
            'own_source_plan_sha256': digest(paths['own_source_plan']),
            'evidence': {'repo': str(repo), 'specification': str(specification)},
            'qualification_seconds': time.monotonic() - started, 'a2_complete': False,
            'execution_authorized': False,
            'scope': 'Direct complete source-bound CPU/features neural pre/post prices for both bounded '
                     'Gray Go tasks, original R4 recipe, twofold runtime margin and seeds 0/1/2. '
                     'Partial buffers and scaled overhead remain forecast limitations. Zero accelerator '
                     'hours applies only to CPU execution; complete CPU caps require separate reservations. '
                     'No CUDA/native/9x9 price, control-budget change, pilot or study selection.'}


def read_qualified(path):
    proof = load(path)
    require(proof.get('version') == VERSION and proof['verifier_sha256'] == digest(__file__),
            'receipt-bound Gray Go CPU price verifier required')
    evidence = proof['evidence']; fresh = qualify(evidence['repo'], evidence['specification'])
    left, right = dict(proof), dict(fresh)
    left.pop('qualification_seconds'); right.pop('qualification_seconds')
    require(left == right, 'fresh source-bound Gray Go CPU price changed')
    return proof


def case_settings(allocation, task, seed, proof, identity):
    require(proof.get('version') == VERSION and proof.get('status') == 'qualified_cpu_gray_neural_prices'
            and proof.get('execution_authorized') is False, 'qualified Gray Go neural CPU prices required')
    require(allocation['tasks'] == list(TASKS) and allocation['seeds'] == [0, 1, 2]
            and task in TASKS and type(seed) is int and seed in allocation['seeds'],
            'both complete original three-seed families must be selected')
    require(allocation['learner'] == 'neural' and allocation['target'] == 'one-step' and allocation['q_reuse'] == 4,
            'only the primary neural R4 recipe is selected')
    keys = {t + '/' + str(s) for t in TASKS for s in (0, 1, 2)}
    require(set(allocation['case_roots']) == set(allocation['case_configs']) == keys, 'complete six-case inventory required')
    roots = [Path(p) for p in allocation['case_roots'].values()]
    require(len(set(roots)) == 6 and all(not p.is_absolute() and len(p.parts) > 1
            and p.parts[0] == 'runs' and '..' not in p.parts for p in roots)
            and all(a not in b.parents for a in roots for b in roots if a != b),
            'distinct non-overlapping original roots inside runs required')
    require(allocation['additional_control_reservation_seconds'] == '0' and allocation['new_accelerator_jobs'] == 0,
            'CPU neural cases cannot spend table-control or GPU time')
    require(allocation['learner_source_sha256'] == proof['learner_source_sha256']
            and identity == proof['runtime_identity'], 'priced candidate source/host/runtime required')
    caps = allocation['per_case_process_ceiling_seconds']
    require(set(caps) == set(TASKS) and all(type(caps[t]) is int
            and str(caps[t]) == proof['prices'][t]['cpu_process_ceiling_seconds'] for t in TASKS)
            and allocation['aggregate_neural_cpu_reservation_seconds'] == str(3 * sum(caps.values())),
            'reserve all six complete process caps at independently qualified prices')
    for t in TASKS:
        price = proof['prices'][t]
        require(price['source_sha256'] == proof['learner_source_sha256'] and price['device_hours'] == '0'
                and price['task'] == t and price['learner'] == 'neural' and price['q_reuse'] == 4,
                'price task/source/backend binding changed')
        for s in (0, 1, 2):
            config = NFSPTaskConfig(**allocation['case_configs'][t + '/' + str(s)])
            expected = NFSPTaskConfig(task=t, learner='neural', seed=s, q_reuse=4,
                time_cap_seconds=caps[t], budget_profile_sha256=price['raw_budget_sha256'])
            require(config.formal_protocol() and config.plan() == expected.plan(), 'full original Gray Go recipe differs')
    return NFSPTaskConfig(**allocation['case_configs'][task + '/' + str(seed)])


def run_case(repo, allocation_path, expected_sha, task, seed):
    repo = Path(repo).resolve(); allocation = load(allocation_path)
    require(digest(allocation_path) == expected_sha and digest(__file__) == allocation['runner_sha256'],
            'committed Gray Go allocation or runner changed')
    require(source_digest() == allocation['learner_source_sha256'], 'wrong frozen learner source')
    require(task in TASKS and type(seed) is int and seed in allocation['seeds'], 'case not selected')
    root = (repo / allocation['case_roots'][task + '/' + str(seed)]).resolve()
    require(root.is_relative_to((repo / 'runs').resolve()), 'case must stay inside project runs')
    stopped = []
    for name in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(name, lambda s, f: stopped.append('budget_limited' if s == signal.SIGUSR1 else 'interrupted'))
    with fresh_case_directory(root):
        admitted = time.monotonic()
        paths = {k: repo / allocation[k] for k in ('cpu_price_receipt', 'source_registry', 'own_source_plan', 'source_admission')}
        require(all(digest(p) == allocation[k + '_sha256'] for k, p in paths.items()), 'price or source registration changed')
        proof = read_qualified(paths['cpu_price_receipt']); config = case_settings(allocation, task, seed, proof, runtime_identity())
        require(proof['source_registry_sha256'] == digest(paths['source_registry'])
                and proof['own_source_plan_sha256'] == digest(paths['own_source_plan'])
                and proof['source_admission_sha256'] == digest(paths['source_admission']), 'price source bindings differ')
        admission = load(paths['source_admission'])
        cpu_cache_admission.check_case(dict(runtime_identity(), source_sha256=source_digest(), config=config.plan()), admission)
        require(not stopped, 'stopped before original initialization')
        root.mkdir()
        write_report(root / 'admission.json', {'version': 1, 'allocation_sha256': expected_sha,
            'source_sha256': source_digest(), 'task': task, 'seed': seed,
            'input_receipts_sha256': {k: digest(p) for k, p in paths.items()},
            'measured_preinitialization_seconds': time.monotonic() - admitted,
            'fresh_prices_and_source_inputs_verified': True, 'a2_complete': False})
        tick = time.monotonic(); trainer = NFSPTaskTrainer(config, root); setup = time.monotonic() - tick
        require({k: trainer.manifest[k] for k in IDENTITY_KEYS} == proof['runtime_identity']
                and trainer.manifest['config'] == config.plan(), 'emitted training manifest differs')
        cpu_cache_admission.check_case(trainer.manifest, admission)
        initialized = retain_initialization(trainer, root)
        write_report(root / 'case-state.json', {'status': 'running', 'task': task, 'seed': seed,
            'config': config.plan(), 'source_sha256': source_digest(), 'original_pid': os.getpid(),
            'constructor_seconds': setup, 'retained_initial_checkpoint_manifest_sha256':
            initialized['retained_initial_checkpoint_manifest_sha256']})
        result = trainer.run(stop_requested=lambda: stopped[0] if stopped else False)
        write_report(root / 'process-portion.json', {'constructor_seconds': setup,
            'loop_seconds': result['elapsed_seconds'], 'known_constructor_and_loop_seconds': setup + result['elapsed_seconds'],
            'scope': 'Known constructor and complete loop; outer process cap includes admission and cleanup.'})
        write_report(root / 'case-state.json', {'status': result['status'], 'task': task, 'seed': seed,
            'records': result['records'], 'source_sha256': source_digest(), 'original_pid': os.getpid(), 'a2_complete': False})
        print(json.dumps({'status': result['status'], 'task': task, 'seed': seed, 'records': result['records']}), flush=True)


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    q = sub.add_parser('qualify'); q.add_argument('--repo', required=True); q.add_argument('--specification', required=True)
    q.add_argument('--output', required=True)
    r = sub.add_parser('run'); r.add_argument('--repo', required=True); r.add_argument('--allocation', required=True)
    r.add_argument('--allocation-sha', required=True); r.add_argument('--task', choices=TASKS, required=True)
    r.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'qualify':
        require(not Path(args.output).exists(), 'preserve original CPU Gray Go price qualification')
        proof = qualify(args.repo, args.specification); write_report(args.output, proof)
        print(json.dumps({'status': proof['status'], 'caps': {t: p['cpu_process_ceiling_seconds'] for t, p in proof['prices'].items()}}))
    else:
        run_case(args.repo, args.allocation, args.allocation_sha, args.task, args.seed)


if __name__ == '__main__':
    main()

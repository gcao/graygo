"""Run only the two missing original full R1 delayed neural comparisons."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from graygo.manifest import source_digest, write_report
from graygo.storage import digest as _digest
from graygo.task_training import NFSPTaskConfig, NFSPTaskTrainer
from tools.delayed_r1_admission import read_qualified, check_case
from tools.nfsp_delayed_cpu_prices import runtime_identity, IDENTITY_KEYS
from tools.nfsp_resources import require
from tools.run_nfsp_cpu_delayed_case import fresh_case_directory, retain_initialization


def digest(path):
    return _digest(Path(path))


def load(path):
    return json.loads(Path(path).read_text())


def case_settings(allocation, seed, proof, identity):
    require(type(seed) is int and allocation['seeds'] == [1, 2] and seed in allocation['seeds'],
            'only original remaining seeds one and two are selected; preserve seed zero')
    require(allocation['cohort_role'] == 'one-step-reuse-comparison'
            and allocation['primary_A2_slots_fillable'] is False and allocation['task'] == 'delayed'
            and allocation['learner'] == 'neural' and allocation['q_reuse'] == 1 and allocation['target'] == 'one-step',
            'R1 comparisons cannot replace primary R4 cases')
    require(set(allocation['case_configs']) == set(allocation['case_roots']) == {'1', '2'}, 'both original comparisons required')
    roots = [Path(p) for p in allocation['case_roots'].values()]
    require(len(set(roots)) == 2 and all(not p.is_absolute() and len(p.parts) > 1
            and p.parts[0] == 'runs' and '..' not in p.parts for p in roots)
            and all(a not in b.parents for a in roots for b in roots if a != b), 'distinct original roots inside runs required')
    require(allocation['additional_control_reservation_seconds'] == '0' and allocation['new_accelerator_jobs'] == 0,
            'CPU neural comparisons cannot spend controls or GPU time')
    cap = allocation['per_case_process_ceiling_seconds']
    require(type(cap) is int and str(cap) == proof['cpu_prices']['prices']['C1']['cpu_process_ceiling_seconds']
            and allocation['aggregate_neural_cpu_reservation_seconds'] == str(2 * cap), 'complete qualified C1 caps must be reserved')
    require(allocation['learner_source_sha256'] == proof['candidate_source_sha256'], 'allocated R1 source is not admitted')
    for original_seed in allocation['seeds']:
        config = NFSPTaskConfig(**allocation['case_configs'][str(original_seed)])
        require(config.seed == original_seed, 'requested seed differs from its original comparison config')
        check_case(dict(identity, source_sha256=allocation['learner_source_sha256'], config=config.plan()), proof)
    return NFSPTaskConfig(**allocation['case_configs'][str(seed)])


def run_case(repo, allocation_path, expected_sha, seed):
    repo = Path(repo).resolve(); allocation = load(allocation_path)
    require(digest(allocation_path) == expected_sha and digest(__file__) == allocation['runner_sha256'], 'committed allocation or runner changed')
    require(source_digest() == allocation['learner_source_sha256'], 'wrong original learner package')
    require(type(seed) is int and seed in allocation['seeds'], 'comparison seed not selected')
    root = (repo / allocation['case_roots'][str(seed)]).resolve()
    require(root.is_relative_to((repo / 'runs').resolve()), 'comparison must remain inside project runs')
    stopped = []
    for name in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(name, lambda s, f: stopped.append('budget_limited' if s == signal.SIGUSR1 else 'interrupted'))
    with fresh_case_directory(root):
        admitted = time.monotonic(); receipt_path = repo / allocation['comparison_admission']
        require(digest(receipt_path) == allocation['comparison_admission_sha256'], 'comparison source/price receipt changed')
        proof = read_qualified(receipt_path); config = case_settings(allocation, seed, proof, runtime_identity())
        for key in ('source_registry', 'own_source_plan'):
            require(digest(repo / allocation[key]) == allocation[key + '_sha256'] == proof[key + '_sha256'], 'original primary source plan/registry changed')
        require(not stopped, 'stopped before original comparison initialization')
        root.mkdir()
        write_report(root / 'admission.json', {'version': 1, 'allocation_sha256': expected_sha,
            'comparison_admission_sha256': digest(receipt_path), 'source_sha256': source_digest(), 'seed': seed,
            'cohort_role': 'one-step-reuse-comparison', 'primary_A2_slots_fillable': False,
            'measured_preinitialization_seconds': time.monotonic() - admitted, 'fresh_source_prices_verified': True})
        tick = time.monotonic(); trainer = NFSPTaskTrainer(config, root); setup = time.monotonic() - tick
        check_case(trainer.manifest, proof)
        initialized = retain_initialization(trainer, root)
        write_report(root / 'case-state.json', {'status': 'running', 'seed': seed, 'config': config.plan(),
            'source_sha256': source_digest(), 'original_pid': os.getpid(), 'constructor_seconds': setup,
            'cohort_role': 'one-step-reuse-comparison', 'primary_A2_slots_fillable': False,
            'retained_initial_checkpoint_manifest_sha256': initialized['retained_initial_checkpoint_manifest_sha256']})
        result = trainer.run(stop_requested=lambda: stopped[0] if stopped else False)
        write_report(root / 'process-portion.json', {'constructor_seconds': setup,
            'loop_seconds': result['elapsed_seconds'], 'known_constructor_and_loop_seconds': setup + result['elapsed_seconds'],
            'scope': 'Known constructor and complete loop; whole-process cap includes admission and cleanup.'})
        write_report(root / 'case-state.json', {'status': result['status'], 'seed': seed, 'records': result['records'],
            'source_sha256': source_digest(), 'original_pid': os.getpid(), 'cohort_role': 'one-step-reuse-comparison',
            'primary_A2_slots_fillable': False, 'a2_complete': False})
        print(json.dumps({'status': result['status'], 'seed': seed, 'records': result['records'], 'primary_A2_slots_fillable': False}), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--repo', required=True); parser.add_argument('--allocation', required=True)
    parser.add_argument('--allocation-sha', required=True); parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args(); run_case(args.repo, args.allocation, args.allocation_sha, args.seed)


if __name__ == '__main__':
    main()

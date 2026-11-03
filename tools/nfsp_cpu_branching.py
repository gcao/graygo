"""Price and run the three original CPU branching neural cases.

The unselected control proposal is reused only as an immutable audited price
corpus. It provides no authority to increase the table-control budget.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from graygo.manifest import source_digest, write_report
from graygo.storage import digest as _digest
from graygo.task_training import NFSPTaskConfig, NFSPTaskTrainer
from tools.a2_source_cohort import read_registry, check_selected_cpu_case
from tools.nfsp_control_budget import build_proposal
from tools.nfsp_delayed_cpu_prices import runtime_identity
from tools.nfsp_resources import canonical, require
from tools.run_nfsp_cpu_delayed_case import fresh_case_directory, retain_initialization

VERSION = 'graygo/branching-cpu-runtime-price/1'


def digest(path):
    return _digest(Path(path))


def load(path):
    return json.loads(Path(path).read_text())


def qualify(repo, proposal_path, evidence_path):
    started = time.monotonic()
    proposal = load(proposal_path)
    fresh = build_proposal(repo, evidence_path)
    require(canonical(proposal) == canonical(fresh), 'fresh audited price corpus differs')
    require(proposal['selected'] is False and proposal['owner_approved'] is False,
            'preserve the original unselected control proposal')
    matches = [p for p in proposal['audited_cpu_prices']
               if p['task'] == 'branching' and p['learner'] == 'neural']
    require(len(matches) == 1, 'exactly one audited branching neural price required')
    price = matches[0]
    require(price['q_reuse'] == 4 and price['device'] == 'cpu' and price['device_hours'] == '0'
            and price['runtime_margin'] == '2' and price['formal_record_cap'] == 15000000
            and price['original_seed_scope'] == [0, 1, 2], 'full primary CPU price required')
    return {'version': VERSION, 'status': 'qualified_cpu_branching_neural_price',
            'price': price, 'runtime_identity': proposal['runtime_identity'],
            'verifier_sha256': digest(__file__),
            'proposal_sha256': digest(proposal_path), 'evidence_sha256': digest(evidence_path),
            'input_files_sha256': proposal['evidence_files_sha256'],
            'evidence': {'repo': str(Path(repo).resolve()),
                         'proposal': str(Path(proposal_path).resolve()),
                         'specification': str(Path(evidence_path).resolve())},
            'control_scope_approved': False, 'a2_complete': False, 'execution_authorized': False,
            'qualification_seconds': time.monotonic() - started,
            'scope': 'Direct source-bound CPU branching neural price only; complete pre/post '
                     'profiles, original twofold margin and original seeds. No table-control '
                     'budget increase, GPU/native/9x9 price or project entry.'}


def read_qualified(path):
    proof = load(path)
    require(proof['version'] == VERSION and proof['verifier_sha256'] == digest(__file__),
            'use the receipt-bound branching price verifier')
    evidence = proof['evidence']
    fresh = qualify(evidence['repo'], evidence['proposal'], evidence['specification'])
    left, right = dict(proof), dict(fresh)
    left.pop('qualification_seconds'); right.pop('qualification_seconds')
    require(left == right, 'branching price inputs or live runtime changed')
    return proof


def case_settings(allocation, seed, proof, identity):
    require(proof.get('version') == VERSION
            and proof.get('status') == 'qualified_cpu_branching_neural_price'
            and proof.get('execution_authorized') is False, 'qualified branching price required')
    require(type(seed) is int and allocation['seeds'] == [0, 1, 2] and seed in allocation['seeds'],
            'all three original seeds must be selected')
    require(allocation['task'] == 'branching' and allocation['learner'] == 'neural'
            and allocation['q_reuse'] == 4 and allocation['target'] == 'one-step',
            'only the primary branching neural recipe is selected')
    require(set(allocation['case_configs']) == set(allocation['case_roots']) == {'0', '1', '2'},
            'complete fixed case inventory required')
    roots = [Path(p) for p in allocation['case_roots'].values()]
    require(len(set(roots)) == 3 and all(not p.is_absolute() and len(p.parts) > 1
            and p.parts[0] == 'runs' and '..' not in p.parts for p in roots)
            and all(a not in b.parents for a in roots for b in roots if a != b),
            'distinct original case roots inside runs required')
    require(allocation['additional_control_reservation_seconds'] == '0'
            and allocation['new_accelerator_jobs'] == 0, 'CPU neural cases cannot spend controls or GPUs')
    price = proof['price']
    cap = allocation['per_case_process_ceiling_seconds']
    require(type(cap) is int and str(cap) == price['cpu_process_ceiling_seconds']
            and allocation['aggregate_neural_cpu_reservation_seconds'] == str(3 * cap),
            'reserve each complete original process cap')
    require(allocation['learner_source_sha256'] == price['source_sha256']
            and identity == proof['runtime_identity'], 'priced source/host/runtime required')
    # Validate the entire selected cohort, not just the seed being launched.
    for original_seed in allocation['seeds']:
        config = NFSPTaskConfig(**allocation['case_configs'][str(original_seed)])
        expected = NFSPTaskConfig(task='branching', learner='neural', seed=original_seed,
                                  q_reuse=4, time_cap_seconds=cap,
                                  budget_profile_sha256=price['raw_budget_sha256'])
        require(config.formal_protocol() and config.plan() == expected.plan(),
                'full original branching config or funded ceiling differs')
    return NFSPTaskConfig(**allocation['case_configs'][str(seed)])


def run_case(repo, allocation_path, expected_sha, seed):
    repo = Path(repo).resolve(); allocation = load(allocation_path)
    require(digest(allocation_path) == expected_sha and digest(__file__) == allocation['runner_sha256'],
            'committed branching allocation or runner changed')
    require(source_digest() == allocation['learner_source_sha256'], 'wrong frozen learner source')
    require(seed in allocation['seeds'], 'seed not selected')
    stopped = []
    for name in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(name, lambda s, f: stopped.append('budget_limited' if s == signal.SIGUSR1 else 'interrupted'))
    with fresh_case_directory(repo / allocation['case_roots'][str(seed)]) as root:
        admitted = time.monotonic()
        paths = {key: repo / allocation[key] for key in ('cpu_price_receipt', 'source_registry', 'own_source_plan')}
        require(all(digest(p) == allocation[k + '_sha256'] for k, p in paths.items()),
                'price or original source registration changed')
        proof = read_qualified(paths['cpu_price_receipt'])
        config = case_settings(allocation, seed, proof, runtime_identity())
        _, _, plans, edges = read_registry(paths['source_registry'], repo=repo, current=True)
        own = [p for p in plans.values() if p.learner_source_sha256 == source_digest()]
        require(len(own) == 1 and own[0].to_dict() == load(paths['own_source_plan']), 'own-source plan changed')
        bridge = next(e['proof'] for e in edges if e['proof'].get('candidate_source_sha256') == source_digest()
                      and e['proof'].get('source_compatibility_qualified'))
        preview = dict(runtime_identity(), source_sha256=source_digest(), config=config.plan())
        check_selected_cpu_case(preview, bridge)
        require(not stopped, 'stopped before original initialization')
        root.mkdir()
        write_report(root / 'admission.json', {'version': 1, 'allocation_sha256': expected_sha,
            'source_sha256': source_digest(), 'seed': seed,
            'price_receipt_sha256': digest(paths['cpu_price_receipt']),
            'source_registry_sha256': digest(paths['source_registry']),
            'own_source_plan_sha256': digest(paths['own_source_plan']),
            'measured_preinitialization_seconds': time.monotonic() - admitted,
            'fresh_prices_and_source_inputs_verified': True, 'a2_complete': False})
        tick = time.monotonic(); trainer = NFSPTaskTrainer(config, root); setup = time.monotonic() - tick
        require({k: trainer.manifest[k] for k in proof['runtime_identity']} == proof['runtime_identity'],
                'emitted manifest runtime differs from price')
        require(trainer.manifest['config'] == config.plan(), 'emitted training manifest differs')
        check_selected_cpu_case(trainer.manifest, bridge)
        initialized = retain_initialization(trainer, root)
        write_report(root / 'case-state.json', {'status': 'running', 'seed': seed, 'config': config.plan(),
            'source_sha256': source_digest(), 'original_pid': os.getpid(), 'constructor_seconds': setup,
            'retained_initial_checkpoint_manifest_sha256': initialized['retained_initial_checkpoint_manifest_sha256']})
        result = trainer.run(stop_requested=lambda: stopped[0] if stopped else False)
        write_report(root / 'process-portion.json', {'constructor_seconds': setup,
            'loop_seconds': result['elapsed_seconds'], 'known_constructor_and_loop_seconds': setup + result['elapsed_seconds'],
            'scope': 'Known constructor and complete loop; outer process budget also includes admission and cleanup.'})
        write_report(root / 'case-state.json', {'status': result['status'], 'seed': seed,
            'records': result['records'], 'source_sha256': source_digest(), 'original_pid': os.getpid(), 'a2_complete': False})
        print(json.dumps({'status': result['status'], 'seed': seed, 'records': result['records']}), flush=True)


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    q = sub.add_parser('qualify'); q.add_argument('--repo', required=True); q.add_argument('--proposal', required=True)
    q.add_argument('--evidence', required=True); q.add_argument('--output', required=True)
    r = sub.add_parser('run'); r.add_argument('--repo', required=True); r.add_argument('--allocation', required=True)
    r.add_argument('--allocation-sha', required=True); r.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'qualify':
        require(not Path(args.output).exists(), 'preserve original price qualification')
        proof = qualify(args.repo, args.proposal, args.evidence); write_report(args.output, proof)
        print(json.dumps({'status': proof['status'], 'cpu_process_ceiling_seconds': proof['price']['cpu_process_ceiling_seconds']}))
    else:
        run_case(args.repo, args.allocation, args.allocation_sha, args.seed)


if __name__ == '__main__':
    main()

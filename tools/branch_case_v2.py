"""Source-bound v2 branching case collection, never a v1 case conversion."""
from contextlib import ExitStack
import json
from pathlib import Path
import runpy

import torch
from graygo.manifest import source_digest, write_report
from graygo.storage import DEFAULT_RESERVE
from graygo.task_training import NFSPTaskConfig
from tools.a2_cohort import input_snapshot, read_lock, digest, canonical, require

VERSION = 'graygo/nfsp-branching-case/2'
AUDITOR = Path(__file__).with_name('audit_nfsp_bounded_v2.py')


def collect(root, output, *, expected_source_sha256, expected_auditor_sha256,
            allow_diagnostic=False, budget_profile=None, reserve_bytes=None):
    root = Path(root).resolve(); output = Path(output).resolve()
    require(not output.is_relative_to(root), 'case output must remain outside original inputs')
    require(not output.exists(), 'preserve the original case output')
    require(source_digest() == expected_source_sha256 and digest(AUDITOR) == expected_auditor_sha256,
            'exact new source and independent v2 auditor required')
    with ExitStack() as locks:
        read_lock(locks, root)
        before = input_snapshot(root)
        manifest = json.loads((root / 'run.json').read_text())
        result = json.loads((root / 'result.json').read_text())
        config = NFSPTaskConfig(**manifest['config'])
        require(config.task == 'branching' and config.seed in (0, 1, 2) and config.q_target == 'one-step',
                'original branching seed and independently audited one-step target required')
        require(manifest['source_sha256'] == result['source_sha256'] == expected_source_sha256
                and manifest.get('evaluation_protocol_version') == result.get('evaluation_protocol_version')
                    == 'graygo/nfsp-task-evaluation/2', 'explicit source-bound version-two evidence required')
        require(result['status'] == 'complete' and result['records'] == config.records,
                'a complete original configured run is required')
        primary = config.formal_protocol() and config.q_reuse == 4
        require(primary or allow_diagnostic, 'full fixed original primary recipe required')
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            audited = runpy.run_path(str(AUDITOR))['audit'](root, require_formal=primary,
                                                          budget_profile=budget_profile)
        require(audited['verified'] and audited['audit_semantics'] == 'graygo/independent-bounded-task-audit/2',
                'independent complete-strategy version-two audit required')
        if primary:
            require([r['records'] for r in audited['scheduled_evaluations']] ==
                    list(range(500000, 15000001, 500000)), 'all thirty fixed original scheduled reads required')
        require(input_snapshot(root) == before, 'original inputs changed during read-only collection')
        body = {'version': VERSION, 'case': ['branching', config.learner, config.seed], 'root': str(root),
                'records': config.records, 'primary_recipe': bool(primary),
                'evaluation_protocol_version': manifest['evaluation_protocol_version'],
                'grading_protocol_version': 'graygo/nfsp-branching-grading/2',
                'fresh_audit_verified': True, 'independent_audit': audited, 'input_snapshot': before,
                'provenance': {'learner_source_sha256': expected_source_sha256,
                               'auditor_sha256': expected_auditor_sha256,
                               'collector_sha256': digest(Path(__file__)),
                               'training_manifest_sha256': digest(root / 'run.json'),
                               'result_sha256': digest(root / 'result.json')},
                'fixed_cohort_requirements': {'seeds': [0, 1, 2], 'learners': ['neural', 'tabular'],
                    'records_per_seed': 15000000, 'final_evaluations': 5, 'evaluation_interval': 500000,
                    'average_gap': .05, 'native_response_gap': .05, 'supported_value_error': .10},
                'source_admission_qualified': False, 'a2_complete': False, 'execution_authorized': False,
                'scope': 'Version-two case evidence only. Original v1 cases/plans/grades remain unchanged; source admission, full controls, cohort completeness and entry scope are separate.'}
        wrapper = {'body': body, 'sha256': canonical(body)}
        write_report(output, wrapper, reserve_bytes=DEFAULT_RESERVE if reserve_bytes is None else reserve_bytes)
        return wrapper

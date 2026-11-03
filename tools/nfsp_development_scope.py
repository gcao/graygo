"""Immutable scope selection and cumulative development-stage mechanics.

These rules compose with the frozen BoardExecutor and DevelopmentCohort. They
do not replace fresh A2, pricing, source, fixture or resource-entry validation.
The CLI is read-only and cannot launch training.
"""
import argparse
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from graygo.board_execution import EXECUTION_VERSION
from graygo.cohort_evidence import DevelopmentCohort
from graygo.pilot_gates import DEVELOPMENT_SEEDS
from tools.nfsp_resources import NFSPResourcePlan, canonical, fingerprint, number, require

VERSION = 'graygo/nfsp-development-scope/1'
MODES = ('stop', 'limited-pilot', 'limited-screening', 'full-study')


@dataclass(frozen=True)
class ScopeSelection:
    resource_plan_sha256: str
    mode: str
    project_device_hour_ceiling: Decimal
    uniform_board_comparison_selected: bool
    owner_decision_sha256: str

    def __post_init__(self):
        require(fingerprint(self.resource_plan_sha256) and fingerprint(self.owner_decision_sha256),
                'exact immutable plan and explicit owner-decision identities required')
        require(self.mode in MODES and type(self.uniform_board_comparison_selected) is bool, 'explicit supported scope required')
        object.__setattr__(self, 'project_device_hour_ceiling', number(self.project_device_hour_ceiling))
        require(self.mode != 'full-study' or self.uniform_board_comparison_selected,
                'full study must include its declared matched uniform comparison')

    @classmethod
    def from_decision(cls, plan, decision):
        require(isinstance(plan, NFSPResourcePlan), 'typed source-bound numeric resource plan required')
        require(isinstance(decision, dict) and decision.get('version') == VERSION
                and decision.get('owner_approved') is True, 'explicit owner decision required; elapsed time is not approval')
        require(set(decision) == {'version', 'resource_plan_sha256', 'owner_approved', 'mode',
                'project_device_hour_ceiling', 'uniform_board_comparison_selected',
                'decision_record', 'decision_record_sha256'}, 'owner decision field inventory differs')
        require(decision.get('resource_plan_sha256') == canonical(plan.to_dict()), 'owner decision belongs to a different immutable plan')
        require(isinstance(decision.get('decision_record'), str) and decision['decision_record'].strip()
                and fingerprint(decision.get('decision_record_sha256')), 'record the actual owner answer and its evidence binding')
        require(hashlib.sha256(decision['decision_record'].encode()).hexdigest() == decision['decision_record_sha256'],
                'owner answer text differs from its immutable evidence fingerprint')
        value = cls(decision['resource_plan_sha256'], decision['mode'], decision['project_device_hour_ceiling'],
                    decision['uniform_board_comparison_selected'], canonical(decision))
        require(plan.project_device_hour_ceiling is None or value.project_device_hour_ceiling == plan.project_device_hour_ceiling,
                'an already filled project ceiling cannot silently change')
        return value

    def to_dict(self):
        return {'version': VERSION, 'resource_plan_sha256': self.resource_plan_sha256, 'mode': self.mode,
                'project_device_hour_ceiling': str(self.project_device_hour_ceiling),
                'uniform_board_comparison_selected': self.uniform_board_comparison_selected,
                'owner_decision_sha256': self.owner_decision_sha256}


def scope_review(plan, selection=None):
    require(isinstance(plan, NFSPResourcePlan), 'typed numeric resource plan required')
    numeric = plan.review(); terms = plan.term_device_hours
    require(selection is None or isinstance(selection, ScopeSelection)
            and selection.resource_plan_sha256 == canonical(plan.to_dict()), 'selection is bound to another plan')
    missing = set(numeric['missing_required_terms']) | set(numeric['unreferenced_numeric_terms'])
    prefix_ready = not (missing - {'B_U'})
    uniform_priced = terms['B_U'] is not None and terms['B_U'] > 0
    full_ready = not missing and uniform_priced
    components_ok = not numeric['component_limit_violations']
    costs = {'stop': Decimal(0), 'limited-pilot': None, 'limited-screening': None,
             'full-study': number(numeric['full_study_device_hours']) if full_ready else None}
    prior = None
    uniform_selected = selection is not None and selection.uniform_board_comparison_selected
    if prefix_ready and (not uniform_selected or uniform_priced):
        # N_A2 is a ceiling, not actual consumption. Retain it conservatively;
        # actual GPU accounting and CPU allocations remain separate validators.
        prior = terms['P'] + terms['U'] + number(numeric['N_A2']) + plan.other_unreused_device_hours + plan.corrective_device_hours
        uniform = terms['B_U'] if uniform_selected else Decimal(0)
        costs['limited-pilot'] = prior + 3 * terms['H_P'] + uniform
        # H_D includes H_P. Never charge or grant a fresh pilot on continuation.
        costs['limited-screening'] = prior + 3 * terms['H_D'] + uniform
    chosen = None if selection is None else costs[selection.mode]
    costed = selection is not None and (selection.mode == 'stop' or components_ok
             and (full_ready if selection.mode == 'full-study' else prefix_ready and (not uniform_selected or uniform_priced)))
    within = costed and chosen is not None and chosen <= selection.project_device_hour_ceiling
    return {'version': VERSION, 'resource_plan_sha256': canonical(plan.to_dict()), 'numeric_plan': numeric,
            'selection': None if selection is None else selection.to_dict(),
            'scope_ceiling_device_hours': {key: None if value is None else str(value) for key, value in costs.items()},
            'selected_scope_costed_and_within_ceiling': within,
            'prefix_price_gaps': sorted(missing - {'B_U'}),
            'full_study_price_gaps': sorted(missing | (set() if uniform_priced else {'B_U: selected single-GPU board comparison needs a positive independently qualified price'})),
            'known_full_study_numeric_subtotal_device_hours': numeric['known_term_device_ceiling_subtotal'],
            'status': 'owner_decision_missing' if selection is None else 'stop' if selection.mode == 'stop' else
                      'scope_numerically_affordable' if within else 'scope_uncosted_or_over_ceiling',
            'execution_authorized': False, 'a2_complete': False, 'runtime_prices_independently_qualified': False,
            'scope': 'Immutable owner selection and numeric scope arithmetic only. Filled future full-study '
                     'prices do not select that study. Fresh A2/source/runtime-price/fixture/control/accounting '
                     'validation and registered three-seed lifecycle are still required before training.'}


def plan_development_stage(plan, selection, stage, seed, *, cohort_root=None):
    require(isinstance(selection, ScopeSelection), 'explicit source-bound owner scope selection required')
    reviewed = scope_review(plan, selection)
    require(selection.mode != 'stop' and reviewed['selected_scope_costed_and_within_ceiling'], 'selected development scope is unavailable')
    require(plan.selected_target == 'one-step', 'the declared board executor needs a separately validated runtime for another target')
    require(plan.term_device_hours['H_P'] > 0 and plan.term_device_hours['H_D'] > 0,
            'development stages need positive single-GPU elapsed caps; CPU execution needs separately priced process ceilings')
    require(type(seed) is int and seed in DEVELOPMENT_SEEDS, 'all development work uses original seeds 100/101/102')
    require(stage in ('initialize', 'pilot-pre', 'pilot-post', 'screening'), 'confirmation requires its independent cohort and controller')
    cohort = None
    if stage != 'initialize':
        require(cohort_root is not None, 'register all three actual initialized development runs before training')
        cohort = DevelopmentCohort.load(cohort_root)
        require(cohort.body['source_sha256'] == plan.learner_source_sha256
                and set(cohort.body['training_runs']) == {'100', '101', '102'}
                and cohort.body['shared_config_without_seed']['q_reuse'] == plan.selected_reuse
                and str(cohort.body['shared_config_without_seed']['device']).split(':')[0] == 'cuda',
                'original complete source/reuse-bound development cohort required')
    if stage == 'screening':
        require(selection.mode in ('limited-screening', 'full-study'), 'a limited pilot cannot automatically become screening')
        report = cohort.report('pilot')['body']
        require(report['status'] == 'pass' and report['pilot']['status'] == 'pass',
                'all three original seeds must pass both fixed pilot windows before screening')
    endpoint = {'initialize': 0, 'pilot-pre': 5000000, 'pilot-post': 8000000, 'screening': 20000000}[stage]
    term = 'H_D' if stage == 'screening' else 'H_P'
    return {'version': VERSION, 'stage': stage, 'seed': seed, 'endpoint_records': endpoint,
            'unchanged_development_record_horizon': 20000000,
            'cumulative_stage_time_ceiling_seconds': str(3600 * plan.term_device_hours[term]),
            'unchanged_execution_contract_time_ceiling_seconds': str(3600 * plan.term_device_hours['H_D']),
            'scope_selection': selection.to_dict(), 'cohort_registry_sha256': None if cohort is None else cohort.sha256,
            'fixed_opening_evidence_required_before_post_collection': stage == 'pilot-post',
            'fresh_full_entry_validation_required': True, 'execution_authorized': False,
            'ceiling_interpretation': 'elapsed seconds for one independently priced physical GPU; CPU ceilings stay separate',
            'scope': 'Stage planning only. Keep the original 20M development clock, epsilon/opening history '
                     'and H_D execution contract across pilot and screening. Enforce the smaller cumulative '
                     'H_P externally during pilot; BoardMilestoneJournal still enforces fixed opening evidence.'}


def run_with_cumulative_stage_cap(executor, until_records, stage_time_ceiling_seconds, *, stop_requested=lambda: False):
    """Clamp an already authorized executor without changing its frozen contract.

    The caller owns entry verification and the stage selection. This mechanism
    never issues authority, initializes a run or resets a cumulative clock.
    """
    cap = number(stage_time_ceiling_seconds)
    require(cap > 0 and cap <= number(executor.plan.time_ceiling_seconds), 'positive cumulative stage cap within the unchanged executor contract required')
    require(executor.active and executor.contract['version'] == EXECUTION_VERSION, 'hold the supported original executor context')
    cause = None

    def stop():
        nonlocal cause
        if executor._spent() >= float(cap):
            cause = 'stage_budget_limited'; return True
        if stop_requested():
            cause = 'interrupted'; return True
        return False

    raw = executor.run(until_records, stop_requested=stop)
    # Preserve the underlying clock/report status exactly. The separate stage
    # report explains why its generic external-stop boundary was requested.
    status = cause if raw['status'] == 'interrupted' and cause else raw['status']
    return {'version': VERSION, 'status': status, 'executor_report': deepcopy(raw),
            'cumulative_stage_time_ceiling_seconds': str(cap),
            'observed_cumulative_execution_seconds': executor._spent(),
            'unchanged_execution_contract_sha256': raw['contract_sha256'],
            'execution_authority_granted': False,
            'budget_boundary': raw['budget_boundary']}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True); parser.add_argument('--decision')
    parser.add_argument('--output'); args = parser.parse_args()
    plan_path = Path(args.plan).resolve(); plan = NFSPResourcePlan.from_dict(json.loads(plan_path.read_text()))
    decision_path = None if args.decision is None else Path(args.decision).resolve()
    selection = None if decision_path is None else ScopeSelection.from_decision(plan, json.loads(decision_path.read_text()))
    result = scope_review(plan, selection)
    if args.output:
        output = Path(args.output).resolve()
        require(not output.exists() and output not in (plan_path, decision_path), 'new read-only review output outside inputs required')
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

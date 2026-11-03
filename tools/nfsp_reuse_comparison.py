"""Assess full delayed R1/R4 comparisons without filling primary A2 slots."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
from graygo.task_training import NFSPTaskConfig
from tools.a2_cohort import (NFSPA2Plan,canonical,digest,require,fingerprint,slots,thresholds,read_lock,input_snapshot,CASE_VERSION,RECORDS,INTERVAL)
from tools.delayed_r1_admission import read_qualified,check_case

VERSION = "graygo/delayed-reuse-comparison/1"

def checked_comparison(plan,wrapper):
    require(isinstance(wrapper,dict) and set(wrapper)=={'body','sha256'} and canonical(wrapper['body'])==wrapper['sha256'],'A2 case checksum differs')
    body=wrapper['body'];require(body.get('version')==CASE_VERSION and body.get('plan_sha256')==plan.sha256,'A2 case plan/semantics differs')
    try:key=tuple(body['case'])
    except (KeyError,TypeError) as exc:raise ValueError('missing A2 case identity') from exc
    require(key in slots(),'unknown A2 family/learner/original seed')
    require(type(key[2]) is int,'A2 seed must be an original integer identifier')
    proof=body['provenance']
    require(proof['learner_source_sha256']==plan.learner_source_sha256 and proof['collector_sha256']==plan.collector_sha256,'A2 case learner/collector source differs')
    require(fingerprint(proof['input_snapshot_sha256']) and canonical(body['input_snapshot'])==proof['input_snapshot_sha256'],'case input snapshot binding differs')
    require(isinstance(body['root'],str) and body['root'] and fingerprint(proof['training_manifest_sha256'])
            and fingerprint(proof['final_checkpoint_manifest_sha256']),'original run/checkpoint identities required')
    require(type(body.get('primary_recipe')) is bool and type(body.get('fresh_audit_verified')) is bool,'explicit case recipe/audit flags required')
    n=body['records'];require(type(n) is int and n>=0 and n%2==0,'aggregate paired A2 record count required')
    require(key[0]=='delayed' and key[1]=='neural','only delayed neural comparison cases are in scope')
    require(body['configured_records']==RECORDS and body['evaluation_interval']==INTERVAL and n<=RECORDS,'unapproved A2 endpoint/clock or extension')
    if body['status']!='complete' or n!=RECORDS:
        return key,body,{'status':'budget_limited' if body['status']=='budget_limited' else 'incomplete_run','numeric_passed':False}
    require(body['fresh_audit_verified'],'completed primary case requires a fresh independent audit snapshot')
    kind='matrix' if key[0] in ('asymmetric','rps') else ('delayed' if key[0]=='delayed' else 'bounded')
    require(proof['auditor_sha256']==dict(plan.auditors)[kind],'case independent auditor differs from fixed plan')
    if kind=='bounded':require(proof.get('shared_helper_sha256')==dict(plan.auditors)['delayed'],'bounded audit helper differs')
    rows=body['evaluations'];require([row['records'] for row in rows]==list(range(INTERVAL,RECORDS+1,INTERVAL)),'complete scheduled A2 evaluation inventory required')
    for row in rows:
        require(type(row['average_ready']) is bool,'explicit deployed-average publication state required')
        require(row['records']>5000000 or not row['average_ready'],'primary deployed average cannot be initialized before its fixed opening')
        for field in ('max_average_gap','max_greedy_gap','max_early_error'):
            value=row.get(field)
            if field=='max_early_error' and key[0] in ('asymmetric','rps'):require(value is None,'one-step matrices do not supply delayed-value evidence');continue
            require(type(value) in (int,float) and math.isfinite(value) and 0<=value<=2.00001,'finite complete original-game gap/error required')
    final=rows[-5:];requirements=thresholds(key[0]);violations=[]
    for row in final:
        if not row['average_ready']:violations.append({'records':row['records'],'metric':'average_not_initialized'})
        for metric,limit in requirements.items():
            if row[metric]>limit:violations.append({'records':row['records'],'metric':metric,'value':row[metric],'threshold':limit})
    maxima={metric:max(row[metric] for row in final) for metric in requirements}
    credit=None if key[0] not in ('delayed','branching') else all(row['max_greedy_gap']<=.05 and row['max_early_error']<=.10 for row in final)
    ready=all(row['average_ready'] for row in final);avg=max(row['max_average_gap'] for row in final)
    band=('uninitialized' if not ready else ('at_most_0.10' if avg<=.10 else ('averaging_lag_0.10_to_0.25' if avg<=.25 else 'above_0.25'))) if key[0]=='delayed' else ('short_game_pass' if ready and avg<=.05 else 'short_game_miss')
    return key,body,{'status':'numeric_pass' if not violations else 'numeric_miss','numeric_passed':not violations,
        'final_records':[row['records'] for row in final],'final_five_maxima':maxima,'q_credit_passed':credit,
        'average_band':band,'violations':violations,'case_sha256':wrapper['sha256']}


def load(path):
    return json.loads(Path(path).read_text())


def assess(repo, allocation_path, new_cases, *, current=True, output=None):
    allocation_path = Path(allocation_path).resolve(); allocation = load(allocation_path)
    require(allocation['cohort_role'] == 'one-step-reuse-comparison'
            and allocation['primary_A2_slots_fillable'] is False and allocation['seeds'] == [1, 2],
            'fixed separate original R1 comparison allocation required')
    repo = Path(repo).resolve()
    require(set(allocation['fixed_original_reference_cases']) == {'R1/0', 'R4/0', 'R4/1', 'R4/2'},
            'preserve original R1 seed zero and all three original R4 references')
    require(set(new_cases) <= {'1', '2'}, 'only the two fixed remaining R1 outputs are admissible')
    proof_path = repo / allocation['comparison_admission']
    require(digest(proof_path) == allocation['comparison_admission_sha256'], 'comparison receipt changed')
    proof = read_qualified(proof_path) if current else load(proof_path)
    cases = {}; provenance = {}; inputs = {str(allocation_path): digest(allocation_path), str(proof_path): digest(proof_path)}
    original_primary_flags = {}; common = None
    selections = []
    for label, reference in allocation['fixed_original_reference_cases'].items():
        path = repo / reference['case']; plan_path = repo / reference['original_plan']
        require(digest(path) == reference['case_file_sha256'] and digest(plan_path) == reference['original_plan_file_sha256'],
                'fixed original comparison reference changed')
        selections.append((label, path, plan_path, reference))
    own_plan = repo / allocation['own_source_plan']
    require(digest(own_plan) == allocation['own_source_plan_sha256'], 'original candidate primary plan changed')
    for seed, path in new_cases.items():
        selections.append(('R1/' + seed, Path(path).resolve(), own_plan, None))
    roots = []; manifests = set()
    with ExitStack() as stack:
        for label, path, plan_path, reference in selections:
            wrapper = load(path); body = wrapper['body']; plan = NFSPA2Plan.from_dict(load(plan_path))
            key, _, score = checked_comparison(plan, wrapper)
            seed = int(label.split('/')[1]); reuse = int(label.split('/')[0][1:]); root = Path(body['root']).resolve()
            require(key == ('delayed', 'neural', seed) and label not in cases and root not in roots,
                    'comparison case/root aliases or original seed changed')
            require(body['provenance']['training_manifest_sha256'] not in manifests, 'comparison runs cannot share a manifest')
            roots.append(root); manifests.add(body['provenance']['training_manifest_sha256'])
            if current:
                read_lock(stack, root)
                require(input_snapshot(root) == body['input_snapshot'], 'comparison case snapshot is stale')
                manifest = load(root / 'run.json')
            else:
                manifest = body['input_snapshot'].get('manifest')
                if manifest is None:
                    # Immutable case snapshots bind this original manifest. Reading its bytes
                    # is allowed in archived mode, but never claims live snapshot verification.
                    manifest = load(root / 'run.json')
                require(digest(root / 'run.json') == body['provenance']['training_manifest_sha256'], 'original comparison manifest changed')
            config = NFSPTaskConfig(**manifest['config'])
            require(manifest['source_sha256'] == plan.learner_source_sha256 and config.formal_protocol()
                    and config.task == 'delayed' and config.learner == 'neural' and config.seed == seed
                    and config.q_reuse == reuse and config.device == 'cpu' and config.storage == 'features'
                    and config.native_library is None, 'full original CPU comparison recipe differs')
            if reference is not None:
                require(str(root) == reference['original_root'] and manifest['config'] == reference['config']
                        and manifest['source_sha256'] == reference['source_sha256']
                        and digest(root / 'run.json') == reference['training_manifest_sha256'], 'original comparison identity changed')
            else:
                require(str(root) == str((repo / allocation['case_roots'][str(seed)]).resolve())
                        and manifest['config'] == allocation['case_configs'][str(seed)] and body['primary_recipe'] is False,
                        'new R1 comparison was relabeled as a primary R4 case')
                check_case(manifest, proof)
            settings = {k: v for k, v in manifest['config'].items() if k not in ('seed', 'q_reuse', 'time_cap_seconds', 'budget_profile_sha256')}
            if common is None: common = settings
            require(settings == common, 'one-step comparisons changed settings beyond seed/reuse/original funding')
            cases[label] = score; original_primary_flags[label] = body['primary_recipe']
            provenance[label] = {'original_root': str(root), 'original_plan_sha256': body['plan_sha256'],
                'original_source_sha256': plan.learner_source_sha256, 'case_body_sha256': wrapper['sha256'],
                'original_training_manifest_sha256': body['provenance']['training_manifest_sha256']}
            inputs[str(path)] = digest(path); inputs[str(plan_path)] = digest(plan_path)
        require(all(digest(path) == expected for path, expected in inputs.items()), 'comparison inputs changed during assessment')
    groups = {}
    for reuse in (1, 4):
        labels = ['R' + str(reuse) + '/' + str(seed) for seed in (0, 1, 2)]
        complete = all(label in cases and cases[label]['status'] in ('numeric_pass', 'numeric_miss') for label in labels)
        passed = complete and all(cases[label]['q_credit_passed'] is True for label in labels)
        groups['R' + str(reuse)] = {'complete': complete, 'credit_status': 'incomplete' if not complete else ('pass' if passed else 'miss'),
            'all_original_seeds_q_credit_passed': passed, 'missing_cases': [label for label in labels if label not in cases],
            'credit_misses': [label for label in labels if label in cases and cases[label].get('q_credit_passed') is False],
            'average_bands': {label: cases[label].get('average_band') for label in labels if label in cases}}
    result = {'version': VERSION, 'groups': groups, 'cases': cases, 'case_provenance': provenance,
        'original_primary_recipe_flags': original_primary_flags, 'input_files_sha256': inputs,
        'current_input_snapshots_verified': current, 'current_source_prices_verified': current,
        'analysis_mode': 'current_inputs' if current else 'archived_case_snapshots_with_bound_original_manifests',
        'both_one_step_credit_cohorts_miss': all(g['complete'] and g['credit_status'] == 'miss' for g in groups.values()),
        'primary_A2_slots_filled': 0, 'reuse_selected': False, 'Retrace_selected': False,
        'marginal_extension_selected': False, 'pilot_authorized': False, 'a2_complete': False,
        'scope': 'Numerical full-budget original R1/R4 delayed comparison only. Exact fixed five-read gates '
                 'and original primary flags remain intact. Q/greedy credit is separate from averaging bands. '
                 'Meaningful-branch/averaging/cohort/control/fixture checks, conditional-extension rules and '
                 'separate measured resource/route decisions remain required; no automatic target or pilot selection.'}
    if output is not None:
        output = Path(output).resolve()
        require(not output.exists() and output not in [Path(p) for p in inputs]
                and all(not output.is_relative_to(root) for root in roots), 'new comparison output outside original inputs required')
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--repo', required=True); parser.add_argument('--allocation', required=True)
    parser.add_argument('--case', action='append', default=[], help='original seed=case wrapper path')
    parser.add_argument('--archived', action='store_true'); parser.add_argument('--output', required=True)
    args = parser.parse_args(); cases = {}
    for item in args.case:
        seed, path = item.split('=', 1); require(seed not in cases, 'duplicate original R1 seed')
        cases[seed] = path
    result = assess(args.repo, args.allocation, cases, current=not args.archived, output=args.output)
    print(json.dumps({'groups': result['groups'], 'primary_A2_slots_filled': 0, 'a2_complete': False}))


if __name__ == '__main__':
    main()

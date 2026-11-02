"""Source-bound NFSP A2 case snapshots and fixed finite-cohort assessment.

This reports numerical requirements. Project entry additionally needs original
cohort/trajectory, control interpretation, 9x9 smoke and resource verification.
"""
import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import runpy
import time

PLAN_VERSION='graygo/nfsp-a2-cohort-plan/1'
CASE_VERSION='graygo/nfsp-a2-case/1'
ASSESS_VERSION='graygo/nfsp-a2-assessment/1'
FAMILIES=('asymmetric','rps','gray-pure','gray-mixed','delayed','branching')
SEEDS=(0,1,2)
LEARNERS=('neural','tabular')
RECORDS=15000000
INTERVAL=500000
AUDIT_FILES={'matrix':'audit_nfsp_matrix.py','delayed':'audit_nfsp_delayed.py','bounded':'audit_nfsp_bounded.py'}


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def fingerprint(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def require(condition,message):
    if not condition:raise ValueError(message)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def slots():return tuple((family,learner,seed) for family in FAMILIES for learner in LEARNERS for seed in SEEDS)
def label(key):return '/'.join(map(str,key))


@dataclass(frozen=True)
class NFSPA2Plan:
    name:str
    learner_source_sha256:str
    collector_sha256:str
    auditors:tuple
    multiround_reuse:int=1
    extension_budget_sha256:str|None=None

    def __post_init__(self):
        auditors=tuple(sorted(self.auditors.items())) if isinstance(self.auditors,dict) else tuple(tuple(x) for x in self.auditors)
        object.__setattr__(self,'auditors',auditors)
        require(isinstance(self.name,str) and bool(self.name),'named fixed A2 candidate required')
        require(fingerprint(self.learner_source_sha256) and fingerprint(self.collector_sha256),'exact learner/collector source fingerprints required')
        require(all(len(x)==2 and fingerprint(x[1]) for x in auditors) and tuple(x[0] for x in auditors)==tuple(sorted(AUDIT_FILES)),
                'exact matrix/delayed/bounded independent auditor fingerprints required')
        require(type(self.multiround_reuse) is int and self.multiround_reuse in (1,4),'selected one-step multiround reuse must be one or four')
        require(self.extension_budget_sha256 is None or fingerprint(self.extension_budget_sha256),'invalid predeclared extension-budget reference')

    def to_dict(self):
        return {'version':PLAN_VERSION,'name':self.name,'learner_source_sha256':self.learner_source_sha256,
            'collector_sha256':self.collector_sha256,'auditors':[list(x) for x in self.auditors],
            'multiround_reuse':self.multiround_reuse,'matrix_reuse':1,'target':'one-step',
            'families':list(FAMILIES),'seeds':list(SEEDS),'learners':list(LEARNERS),
            'records':RECORDS,'evaluation_interval':INTERVAL,'final_evaluations':5,
            'reservoir_open':5000000,'handoff_end':7000000,'epsilon_decay':7500000,'target_every':12800,
            'thresholds':{'short_average_gap':.05,'delayed_average_gap':.10,'greedy_credit_gap':.05,'supported_value_error':.10},
            'extension':{'maximum_additional_records':2000000,'endpoint':17000000,'paired_seeds':list(SEEDS),
                         'maximum_failing_families':1,'maximum_failing_seeds':1,'maximum_relative_miss':1.2,
                         'last_evaluation_must_pass':True,'budget_reference_sha256':self.extension_budget_sha256},
            'scope':'fixed numerical case requirements; this plan allocates no case/control/extension/board work and grants no A2 exit'}

    @property
    def sha256(self):return canonical(self.to_dict())

    @classmethod
    def from_dict(cls,value):
        require(isinstance(value,dict) and value.get('version')==PLAN_VERSION,'unknown A2 cohort plan')
        try:plan=cls(value['name'],value['learner_source_sha256'],value['collector_sha256'],value['auditors'],
                     value['multiround_reuse'],value['extension']['budget_reference_sha256'])
        except (KeyError,TypeError) as exc:raise ValueError('malformed A2 cohort plan') from exc
        require(plan.to_dict()==value,'fixed A2 families/seeds/settings/thresholds/extension rules changed')
        return plan


def thresholds(family):
    result={'max_average_gap':.10 if family=='delayed' else .05}
    if family in ('delayed','branching'):result.update(max_greedy_gap=.05,max_early_error=.10)
    return result


def checked_case(plan,wrapper):
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
    if not body['primary_recipe']:return key,body,{'status':'diagnostic_only','numeric_passed':False}
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


def assess(plan,wrappers):
    present={};bodies={};roots=set();manifests=set()
    for wrapper in wrappers:
        key,body,summary=checked_case(plan,wrapper);require(key not in present,'duplicate A2 case cannot replace an original seed')
        require(body['root'] not in roots and body['provenance']['training_manifest_sha256'] not in manifests,'different original A2 cases cannot alias one run')
        roots.add(body['root']);manifests.add(body['provenance']['training_manifest_sha256'])
        present[key]=summary;bodies[key]=body
    missing=[label(key) for key in slots() if key not in present]
    groups={}
    for learner in LEARNERS:
        keys=[key for key in slots() if key[1]==learner]
        complete=all(key in present and present[key]['status'] in ('numeric_pass','numeric_miss') for key in keys)
        passed=complete and all(present[key]['numeric_passed'] for key in keys)
        groups[learner]={'complete':complete,'status':'incomplete' if not complete else ('numeric_pass' if passed else 'numeric_miss'),
            'numeric_passed':passed,'missing_cases':[label(key) for key in keys if key not in present],
            'case_misses':[label(key) for key in keys if key in present and present[key]['status']=='numeric_miss']}
    extension={'numeric_eligible':False,'selected':False,'resource_authorization_verified':False,
        'budget_reference_sha256':plan.extension_budget_sha256,'reason':'complete original neural cohort required'}
    if groups['neural']['complete']:
        failures=[key for key in present if key[1]=='neural' and not present[key]['numeric_passed']]
        extension['reason']='exactly one original neural family/seed must miss'
        if len(failures)==1:
            key=failures[0];final=bodies[key]['evaluations'][-5:];limits=thresholds(key[0])
            last=all(final[-1][metric]<=limit for metric,limit in limits.items()) and final[-1]['average_ready']
            marginal=all(row['average_ready'] and all(row[metric]<=1.2*limit for metric,limit in limits.items()) for row in final[:-1])
            extension['reason']='last evaluation must pass and every earlier miss must be within twenty percent'
            if last and marginal:
                extension.update(numeric_eligible=True,reason='numerically marginal; budget/entry authority remains separate',family=key[0],failing_seed=key[2],
                    paired_seeds=list(SEEDS),maximum_additional_records_per_seed=2000000,required_endpoint=17000000,
                    original_case_sha256=[present[(key[0],'neural',seed)]['case_sha256'] for seed in SEEDS])
    delayed=[present.get(('delayed','neural',seed)) for seed in SEEDS]
    worst_delayed=None if not all(item and 'final_five_maxima' in item for item in delayed) else max(item['final_five_maxima']['max_average_gap'] for item in delayed)
    neural_band='incomplete' if not groups['neural']['complete'] else ('all_neural_numeric_gates_pass' if groups['neural']['numeric_passed'] else 'not_ready')
    if groups['neural']['complete'] and worst_delayed is not None and .10<worst_delayed<=.25:
        other_ok=all(summary['numeric_passed'] for key,summary in present.items() if key[1]=='neural' and key[0]!='delayed')
        if other_ok and all(item['q_credit_passed'] for item in delayed):neural_band='averaging_lag'
    return {'version':ASSESS_VERSION,'plan_sha256':plan.sha256,'learner_source_sha256':plan.learner_source_sha256,
        'required_cases':len(slots()),'present_cases':len(present),'missing_cases':missing,
        'cases':{label(key):value for key,value in sorted(present.items())},'neural':groups['neural'],'controls':groups['tabular'],
        'neural_deployment_band':neural_band,'worst_delayed_average_gap':worst_delayed,
        'control_interpretation_required':bool(groups['tabular']['case_misses']),'marginal_extension':extension,
        'a2_complete':False,'pilot_authorized':False,'extension_authorized':False,
        'independent_training_trajectory_verified':False,'original_cohort_registration_verified':False,'resource_scope_verified':False,
        'scope':'numerical fixed-cohort assessment of source-bound audited case snapshots; control outcomes remain separate; original registration/trajectory, comparator/fixture checks, 9x9 smoke and filled resource/entry decision remain required'}


def input_snapshot(root):
    files={};directories=[]
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(),'A2 input snapshots cannot contain symbolic links')
        relative=str(path.relative_to(root))
        if path.is_file():files[relative]=digest(path)
        elif path.is_dir():directories.append(relative)
    return {'files':files,'directories':directories}


def read_lock(stack,root):
    import fcntl
    file=stack.enter_context((root/'.trainer.lock').open('rb'))
    fcntl.flock(file,fcntl.LOCK_SH|fcntl.LOCK_NB)
    return file


def normalize_rows(family,result,audited=None):
    rows=[]
    for e in result['evaluations']:
        roots=e['roots'] if family not in ('asymmetric','rps') else [e]
        require(bool(roots) and all(len(root['average_gaps'])==len(root['greedy_q_gaps'])==2 for root in roots),'complete both-role A2 metrics required')
        rows.append({'records':e['records'],'average_ready':e['average_ready'],
            'max_average_gap':max(max(r['average_gaps']) for r in roots),'max_greedy_gap':max(max(r['greedy_q_gaps']) for r in roots),
            'max_early_error':None if family in ('asymmetric','rps') else max(r['max_early_error'] for r in roots)})
    if audited is None:return rows
    verified=audited['scheduled_evaluations'];require(len(verified)==len(rows),'fresh audit scheduled inventory differs')
    final=[]
    for actual,oracle in zip(rows,verified):
        expected=({'records':oracle['records'],'average_ready':oracle['average_ready'],
                   'max_average_gap':max(oracle['average_gaps']),'max_greedy_gap':max(oracle['greedy_q_gaps']),'max_early_error':None}
                  if family in ('asymmetric','rps') else {field:oracle[field] for field in actual})
        require(actual['records']==expected['records'] and actual['average_ready']==expected['average_ready'],'fresh audit evaluation identity/readiness differs')
        for metric in ('max_average_gap','max_greedy_gap','max_early_error'):
            require(actual[metric] is expected[metric] if actual[metric] is None or expected[metric] is None
                    else abs(actual[metric]-expected[metric])<=1e-12,'fresh audit metrics differ from stored result')
        final.append(expected)
    return final


def audit_case(plan,root,output,*,allow_diagnostic=False,budget_profile=None,reserve_bytes=None):
    """Freshly execute the pinned independent audit inside one immutable read."""
    from dataclasses import asdict
    from graygo.manifest import source_digest,write_report
    from graygo.storage import CheckpointStore,DEFAULT_RESERVE
    started=time.monotonic();root=Path(root).resolve();output=Path(output).resolve();tool=Path(__file__).resolve()
    require(not output.is_relative_to(root),'save A2 case output outside original training inputs')
    require(not output.exists(),'A2 case snapshot exists; preserve it and choose a new artifact')
    require(source_digest()==plan.learner_source_sha256,'run case collection with the exact frozen learner package on PYTHONPATH')
    require(digest(tool)==plan.collector_sha256,'case collector differs from the fixed plan; use its frozen tool')
    with ExitStack() as stack:
        read_lock(stack,root)
        require(all((root/name).is_dir() for name in ('resume','weights','staging')),'complete original checkpoint directories required; audit never creates them')
        before=input_snapshot(root);manifest=json.loads((root/'run.json').read_text());result=json.loads((root/'result.json').read_text())
        config=manifest['config'];key=(config['task'],config['learner'],config['seed']);require(key in slots(),'original A2 family/learner/seed required')
        require(manifest['source_sha256']==plan.learner_source_sha256 and result['source_sha256']==manifest['source_sha256'],'case learner source differs')
        kind='matrix' if key[0] in ('asymmetric','rps') else ('delayed' if key[0]=='delayed' else 'bounded')
        if kind=='matrix':
            from graygo.nfsp_training import NFSPMatrixConfig
            parsed=NFSPMatrixConfig(**config);reuse=1
        else:
            from graygo.task_training import NFSPTaskConfig
            parsed=NFSPTaskConfig(**config);reuse=plan.multiround_reuse
        primary=bool(parsed.formal_protocol() and parsed.q_reuse==reuse)
        require(primary or allow_diagnostic,'case does not match the full fixed primary task recipe')
        n=result['records'];require(type(n) is int and 0<=n<=parsed.records and n%2==0,'case aggregate record count differs')
        seal=CheckpointStore.verify(root/'resume'/f'{n:012d}')
        require(seal['step']==n and seal['kind']=='resume' and seal['metadata']['source_sha256']==plan.learner_source_sha256,'case final checkpoint binding differs')
        auditor=tool.with_name(AUDIT_FILES[kind]);require(digest(auditor)==dict(plan.auditors)[kind],'pinned independent audit implementation differs')
        helper=tool.with_name(AUDIT_FILES['delayed'])
        if kind=='bounded':require(digest(helper)==dict(plan.auditors)['delayed'],'pinned bounded audit helper differs')
        budget=None
        if config.get('budget_profile_sha256'):
            budget=Path(budget_profile).resolve() if budget_profile else root.parent/(key[1]+'-budget.json')
            require(digest(budget)==config['budget_profile_sha256'],'case funding-file fingerprint differs')
        elif budget_profile is not None:raise ValueError('case has no external funding-file binding')
        complete=result['status']=='complete' and n==parsed.records
        audited=None
        if complete:
            function=runpy.run_path(str(auditor))['audit'];kwargs={'require_formal':primary}
            if kind!='matrix' and budget is not None:kwargs['budget_profile']=budget
            import torch
            threads=torch.get_num_threads()
            try:
                with torch.random.fork_rng(devices=[]),torch.device('cpu'):audited=function(root,**kwargs)
            finally:torch.set_num_threads(threads)
            require(audited['verified'],'independent case audit did not verify')
        rows=normalize_rows(key[0],result,audited)
        require(input_snapshot(root)==before,'original A2 case inputs changed during the fresh audit')
        if budget is not None:require(digest(budget)==config['budget_profile_sha256'],'case funding file changed during audit')
        body={'version':CASE_VERSION,'plan_sha256':plan.sha256,'case':list(key),'root':str(root),
            'status':result['status'],'records':n,'configured_records':parsed.records,'evaluation_interval':parsed.eval_every,
            'primary_recipe':primary,'fresh_audit_verified':audited is not None,'evaluations':rows,
            'provenance':{'learner_source_sha256':plan.learner_source_sha256,'collector_sha256':digest(tool),
                'auditor_sha256':digest(auditor),'shared_helper_sha256':digest(helper) if kind=='bounded' else None,
                'training_manifest_sha256':before['files']['run.json'],
                'final_checkpoint_manifest_sha256':digest(root/'resume'/f'{n:012d}'/'manifest.json'),
                'budget_sha256':None if budget is None else digest(budget),'input_snapshot_sha256':canonical(before)},
            'input_snapshot':before,'independent_audit':audited,'measured_collection_seconds':time.monotonic()-started,
            'scope':'fresh independent finite-case audit and unchanged input snapshot; no cohort registration, resource authority or A2-exit grant'}
        wrapper={'body':body,'sha256':canonical(body)}
        write_report(output,wrapper,reserve_bytes=DEFAULT_RESERVE if reserve_bytes is None else reserve_bytes)
        return wrapper


def assess_files(plan,paths,*,output=None,reserve_bytes=None,verify_current_inputs=True):
    from graygo.manifest import write_report
    from graygo.storage import DEFAULT_RESERVE
    paths=[Path(path).resolve() for path in paths];wrappers=[json.loads(path.read_text()) for path in paths]
    for wrapper in wrappers:checked_case(plan,wrapper)
    roots=[Path(wrapper['body']['root']).resolve() for wrapper in wrappers]
    if output is not None:
        output=Path(output).resolve()
        require(output not in paths and not any(output.is_relative_to(root) for root in roots),'save A2 assessment outside case artifacts and training inputs')
    with ExitStack() as stack:
        if verify_current_inputs:
            for root in sorted(set(roots)):read_lock(stack,root)
            for wrapper,root in zip(wrappers,roots):require(input_snapshot(root)==wrapper['body']['input_snapshot'],'case snapshot is stale; repeat its explicit read-only audit')
        result=assess(plan,wrappers)
        result['current_input_snapshots_verified']=verify_current_inputs
        result['analysis_mode']='current_inputs' if verify_current_inputs else 'archived_snapshots'
        result['assessor_sha256']=digest(Path(__file__))
        if output is not None:write_report(output,result,reserve_bytes=DEFAULT_RESERVE if reserve_bytes is None else reserve_bytes)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan');p.add_argument('--name',required=True);p.add_argument('--learner-source-sha',required=True)
    p.add_argument('--reuse',type=int,choices=(1,4),required=True);p.add_argument('--extension-budget-sha');p.add_argument('--output',required=True)
    p=sub.add_parser('audit-case');p.add_argument('--plan',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--allow-diagnostic',action='store_true');p.add_argument('--budget-profile');p.add_argument('--time-cap-seconds',type=int,required=True)
    p=sub.add_parser('assess');p.add_argument('--plan',required=True);p.add_argument('--case',action='append',default=[]);p.add_argument('--output')
    p.add_argument('--archived',action='store_true')
    args=parser.parse_args(argv)
    try:
        from graygo.manifest import write_report
        if args.command=='plan':
            path=Path(args.output).resolve();require(not path.exists(),'A2 plan is immutable; retain its original file')
            tool=Path(__file__).resolve();plan=NFSPA2Plan(args.name,args.learner_source_sha,digest(tool),
                {kind:digest(tool.with_name(name)) for kind,name in AUDIT_FILES.items()},args.reuse,args.extension_budget_sha)
            write_report(path,plan.to_dict());result={'plan_sha256':plan.sha256,'allocated_training_records':0,'a2_complete':False}
        else:
            plan=NFSPA2Plan.from_dict(json.loads(Path(args.plan).read_text()))
            if getattr(args,'output',None) is not None:require(Path(args.output).resolve()!=Path(args.plan).resolve(),'output cannot overwrite the immutable A2 plan')
            if args.command=='audit-case':
                require(type(args.time_cap_seconds) is int and 0<args.time_cap_seconds<=14400,'explicit audit watchdog at most four hours required')
                import signal
                def stop(signum,frame):raise TimeoutError('original A2 case audit watchdog exhausted; no success snapshot written')
                signal.signal(signal.SIGALRM,stop);signal.alarm(args.time_cap_seconds)
                case=audit_case(plan,args.run_dir,args.output,allow_diagnostic=args.allow_diagnostic,budget_profile=args.budget_profile)
                signal.alarm(0);result={'case_sha256':case['sha256'],'case':case['body']['case'],'fresh_audit_verified':case['body']['fresh_audit_verified'],'a2_complete':False}
            else:result=assess_files(plan,args.case,output=args.output,verify_current_inputs=not args.archived)
        print(json.dumps(result,indent=2,sort_keys=True));return 0
    except (ValueError,RuntimeError,OSError) as exc:parser.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())

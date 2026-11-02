"""Exact NFSP route arithmetic and non-overlapping resource measurements.

Device-hour ceilings, CPU process-wall measurements, and physical accelerator
allocation intervals are separate quantities. Nothing here launches a job.
"""
import argparse
from dataclasses import dataclass
from decimal import Decimal,InvalidOperation
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

VERSION='graygo/nfsp-resource-ledger/1'
TERMS=('P','U','C1','C4','CT','B_0','B_U','H_P','H_D','H_N','D')
CPU_CATEGORIES=('control','resolver_evaluator','probe','neural','verification','development')
CPU_LIMITS={'control':Decimal(43200),'resolver_evaluator':Decimal(14400),'probe':Decimal(28800)}


def require(condition,message):
    if not condition:raise ValueError(message)
def fingerprint(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def number(value,*,nullable=False):
    if value is None and nullable:return None
    require(type(value) in (str,int,float,Decimal),'explicit numeric quantity required; booleans are not amounts')
    try:d=Decimal(str(value))
    except InvalidOperation as exc:raise ValueError('invalid resource number') from exc
    require(d.is_finite() and d>=0,'finite nonnegative resource quantity required')
    return d
def text_number(value):return None if value is None else str(value)


@dataclass(frozen=True)
class ExtensionCeiling:
    protocol_sha256:str
    family:str
    per_seed_device_hours:Decimal|None
    evidence_sha256:str|None=None

    def __post_init__(self):
        require(fingerprint(self.protocol_sha256) and self.family in ('asymmetric','rps','gray-pure','gray-mixed','delayed','branching'),'one named protocol/family extension required')
        object.__setattr__(self,'per_seed_device_hours',number(self.per_seed_device_hours,nullable=True))
        require(self.evidence_sha256 is None or fingerprint(self.evidence_sha256),'invalid extension reference')
    def to_dict(self):return {'protocol_sha256':self.protocol_sha256,'family':self.family,'per_seed_device_hours':text_number(self.per_seed_device_hours),'evidence_sha256':self.evidence_sha256}


@dataclass(frozen=True)
class NFSPResourcePlan:
    name:str
    learner_source_sha256:str
    selected_reuse:int
    selected_target:str
    term_device_hours:dict
    evidence_sha256:dict
    r4_selected:bool=False
    retrace_selected:bool=False
    zero_opening_selected:bool=False
    diagnostic_selected:bool=False
    extensions:tuple=()
    corrective_device_hours:Decimal=Decimal(0)
    promotion_rounds:int=0
    other_unreused_device_hours:Decimal=Decimal(0)
    project_device_hour_ceiling:Decimal|None=None

    def __post_init__(self):
        require(isinstance(self.name,str) and self.name and fingerprint(self.learner_source_sha256),'named selected learner source required')
        require(type(self.selected_reuse) is int and self.selected_reuse in (1,4) and self.selected_target in ('one-step','retrace'),'explicit selected target/reuse required')
        for flag in (self.r4_selected,self.retrace_selected,self.zero_opening_selected,self.diagnostic_selected):require(type(flag) is bool,'explicit optional-work selections required')
        require(self.selected_reuse!=4 or self.r4_selected,'selected reuse four must retain its comparison cost')
        require(self.selected_target!='retrace' or self.retrace_selected,'selected Retrace must retain its target-comparison cost')
        require(isinstance(self.term_device_hours,dict) and set(self.term_device_hours)==set(TERMS),'complete device-hour symbol inventory required; unknown is null')
        values={key:number(value,nullable=True) for key,value in self.term_device_hours.items()};object.__setattr__(self,'term_device_hours',MappingProxyType(values))
        require(isinstance(self.evidence_sha256,dict) and set(self.evidence_sha256)<=set(TERMS) and all(fingerprint(v) for v in self.evidence_sha256.values()),'cost references must be exact known-symbol fingerprints')
        object.__setattr__(self,'evidence_sha256',MappingProxyType(dict(self.evidence_sha256)))
        ext=tuple(self.extensions);require(all(isinstance(x,ExtensionCeiling) for x in ext),'typed extension ceilings required')
        require(len({x.protocol_sha256 for x in ext})==len(ext),'a candidate protocol cannot receive repeated/different-family extensions')
        object.__setattr__(self,'extensions',ext)
        for name in ('corrective_device_hours','other_unreused_device_hours'):
            object.__setattr__(self,name,number(getattr(self,name)))
        require(self.corrective_device_hours<=24,'corrective device reserve cannot exceed twenty-four hours')
        require(type(self.promotion_rounds) is int and self.promotion_rounds>=0,'nonnegative selected promotion count required')
        object.__setattr__(self,'project_device_hour_ceiling',number(self.project_device_hour_ceiling,nullable=True))
        if self.diagnostic_selected and values['D'] is not None:require(values['D']<=12,'uncorrected diagnostic ceiling exceeds twelve device-hours')
        if values['H_P'] is not None and values['H_D'] is not None:require(values['H_D']>=values['H_P'],'H_D is cumulative and must include H_P')

    def to_dict(self):
        return {'version':VERSION,'name':self.name,'learner_source_sha256':self.learner_source_sha256,
            'selected_reuse':self.selected_reuse,'selected_target':self.selected_target,
            'term_device_hours':{key:text_number(value) for key,value in self.term_device_hours.items()},'evidence_sha256':dict(self.evidence_sha256),
            'r4_selected':self.r4_selected,'retrace_selected':self.retrace_selected,'zero_opening_selected':self.zero_opening_selected,
            'diagnostic_selected':self.diagnostic_selected,'extensions':[x.to_dict() for x in self.extensions],
            'corrective_device_hours':str(self.corrective_device_hours),'promotion_rounds':self.promotion_rounds,
            'other_unreused_device_hours':str(self.other_unreused_device_hours),'project_device_hour_ceiling':text_number(self.project_device_hour_ceiling),
            'fixed':{'short_a2_device_hours':96,'a2_seeds':3,'development_seeds':3,'confirmation_runs':12,
                     'scientific_and_probe_inference_device_hours':16,'per_promotion_device_hours':8,'initial_pilot_max_device_hours_per_seed':24},
            'scope':'numeric device-ceiling arithmetic only; references are not independent pricing qualification or execution authority'}

    @classmethod
    def from_dict(cls,value):
        require(isinstance(value,dict) and value.get('version')==VERSION,'unknown resource plan')
        try:
            plan=cls(value['name'],value['learner_source_sha256'],value['selected_reuse'],value['selected_target'],value['term_device_hours'],value['evidence_sha256'],
                value['r4_selected'],value['retrace_selected'],value['zero_opening_selected'],value['diagnostic_selected'],
                tuple(ExtensionCeiling(**x) for x in value['extensions']),value['corrective_device_hours'],value['promotion_rounds'],
                value['other_unreused_device_hours'],value['project_device_hour_ceiling'])
        except (KeyError,TypeError) as exc:raise ValueError('malformed resource plan') from exc
        require(plan.to_dict()==value,'fixed ledger coefficients, units or optional selections changed')
        return plan

    def review(self):
        t=self.term_device_hours;required=['P','U','C1','B_U','H_P','H_D','H_N']
        if self.r4_selected:required.append('C4')
        if self.retrace_selected:required.append('CT')
        if self.zero_opening_selected:required.append('B_0')
        if self.diagnostic_selected:required.append('D')
        missing=[key for key in required if t[key] is None]
        a2={'primary_short':Decimal(96),'three_C1':None if t['C1'] is None else 3*t['C1']}
        for selected,key,multiplier in ((self.r4_selected,'C4',3),(self.retrace_selected,'CT',3),(self.zero_opening_selected,'B_0',1),(self.diagnostic_selected,'D',1)):
            a2[key]=None if selected and t[key] is None else (multiplier*t[key] if selected else Decimal(0))
        for extension in self.extensions:
            key='extension:'+extension.protocol_sha256
            a2[key]=None if extension.per_seed_device_hours is None else 3*extension.per_seed_device_hours
            if extension.per_seed_device_hours is None:missing.append(key)
        n_a2=None if any(value is None for value in a2.values()) else sum(a2.values(),Decimal(0))
        parts={'P_actual':t['P'],'U_actual_unreused':t['U'],'N_A2':n_a2,'three_H_D':None if t['H_D'] is None else 3*t['H_D'],
            'twelve_H_N':None if t['H_N'] is None else 12*t['H_N'],'scientific_probe_inference':Decimal(16),'B_U':t['B_U'],
            'corrective_reserve':self.corrective_device_hours,'promotions':8*Decimal(self.promotion_rounds),'other_unreused':self.other_unreused_device_hours}
        total=None if missing or any(value is None for value in parts.values()) else sum(parts.values(),Decimal(0))
        known=sum((value for value in parts.values() if value is not None),Decimal(0))
        if n_a2 is None:known+=sum((value for value in a2.values() if value is not None),Decimal(0))
        # H_P is a required input and a prefix of H_D, never another 3*H_P.
        unreferenced=[key for key in required if t[key] is not None and key not in self.evidence_sha256]
        unreferenced.extend('extension:'+x.protocol_sha256 for x in self.extensions if x.per_seed_device_hours is not None and x.evidence_sha256 is None)
        violations=[]
        if t['P'] is not None and t['P']>72:violations.append('PPO actual use exceeds its planned seventy-two-hour component; retain actual use')
        if t['H_P'] is not None and t['H_P']>24:violations.append('profiled H_P exceeds the initial twenty-four-device-hour per-seed pilot limit')
        if total is None:status='uncosted'
        elif self.project_device_hour_ceiling is None:status='no_project_device_ceiling'
        else:status='within_device_ceiling' if total<=self.project_device_hour_ceiling and not violations else 'over_device_ceiling_or_component_limit'
        return {'version':VERSION,'plan_sha256':canonical(self.to_dict()),'learner_source_sha256':self.learner_source_sha256,
            'selected_reuse':self.selected_reuse,'selected_target':self.selected_target,'unit':'device_hours',
            'N_A2':text_number(n_a2),'a2_components':{k:text_number(v) for k,v in a2.items()},
            'full_study_components':{k:text_number(v) for k,v in parts.items()},'full_study_device_hours':text_number(total),
            'known_term_device_ceiling_subtotal':str(known),'missing_required_terms':missing,'unreferenced_numeric_terms':unreferenced,
            'per_seed_H_P':text_number(t['H_P']),'per_seed_H_D_includes_H_P':text_number(t['H_D']),
            'project_device_hour_ceiling':text_number(self.project_device_hour_ceiling),'numerical_cost_status':status,'component_limit_violations':violations,
            'continuous_device_days_not_wall_time_promise':None if total is None else str(total/24),
            'runtime_prices_independently_qualified':False,'execution_authorized':False,
            'scope':'formula and reference inventory, not a measured complete-NFSP price; CPU seconds are separate; no PPO-price substitution, scope selection or launch'}


@dataclass(frozen=True)
class Usage:
    identity:str
    category:str
    unit:str
    seconds:Decimal
    covered_scopes:tuple
    evidence_sha256:str
    state:str='actual'
    device_ids:tuple=()
    interval_start:Decimal|None=None
    interval_end:Decimal|None=None

    def __post_init__(self):
        require(fingerprint(self.identity) and fingerprint(self.evidence_sha256),'exact measurement/reference identities required')
        require(self.category in CPU_CATEGORIES+('accelerator',) and self.state in ('actual','reserved'),'explicit resource category/state required')
        require(self.unit in ('cpu_process_seconds','gpu_equipped_process_seconds'),'CPU and accelerator process units must be explicit')
        require((self.unit=='cpu_process_seconds')==(self.category!='accelerator'),'CPU and accelerator categories cannot change units')
        object.__setattr__(self,'seconds',number(self.seconds));scopes=tuple(self.covered_scopes);devices=tuple(self.device_ids)
        require(bool(scopes) and len(set(scopes))==len(scopes) and all(fingerprint(x) for x in scopes),'unique leaf coverage identities required')
        require(len(set(devices))==len(devices) and all(isinstance(x,str) and x for x in devices),'unique physical device identities required')
        require(self.unit!='cpu_process_seconds' or not devices,'CPU measurements cannot be GPU reservations')
        object.__setattr__(self,'covered_scopes',scopes);object.__setattr__(self,'device_ids',devices)
        start=number(self.interval_start,nullable=True);end=number(self.interval_end,nullable=True)
        require((start is None)==(end is None),'both accelerator interval endpoints required')
        if start is not None:
            require(self.unit=='gpu_equipped_process_seconds' and devices and end>=start,'physical allocation intervals require actual device identities')
            require(abs((end-start)-self.seconds)<=Decimal('.000001'),'measurement duration differs from its allocation interval')
        object.__setattr__(self,'interval_start',start);object.__setattr__(self,'interval_end',end)

    def to_dict(self):
        return {'identity':self.identity,'category':self.category,'unit':self.unit,'seconds':str(self.seconds),
            'covered_scopes':list(self.covered_scopes),'evidence_sha256':self.evidence_sha256,'state':self.state,
            'device_ids':list(self.device_ids),'interval_start':text_number(self.interval_start),'interval_end':text_number(self.interval_end)}


def measured_usage(items):
    identities=set();covered=set();cpu={key:{'actual':Decimal(0),'reserved':Decimal(0)} for key in CPU_CATEGORIES}
    gpu_actual=Decimal(0);gpu_reserved=Decimal(0);intervals={};unknown=[]
    for item in items:
        require(item.identity not in identities,'duplicate resource measurement');identities.add(item.identity)
        # Driver/phase/leaf overlap remains overlap even across actual/reserved
        # states or CPU categories; a settled actual replaces its reservation.
        for scope in item.covered_scopes:
            key=(item.unit,scope);require(key not in covered,'overlapping driver/phase/leaf scopes would double-count resources');covered.add(key)
        if item.unit=='cpu_process_seconds':cpu[item.category][item.state]+=item.seconds;continue
        if item.state=='reserved':gpu_reserved+=item.seconds;continue
        gpu_actual+=item.seconds
        if item.interval_start is None:unknown.append(item.identity)
        else:
            for device in item.device_ids:intervals.setdefault(device,[]).append((item.interval_start,item.interval_end))
    physical={}
    for device,spans in intervals.items():
        total=Decimal(0);start=end=None
        for a,b in sorted(spans):
            if start is None:start,end=a,b
            elif a<=end:end=max(end,b)
            else:total+=end-start;start,end=a,b
        if start is not None:total+=end-start
        physical[device]=total
    categories={}
    for category,amounts in cpu.items():
        committed=amounts['actual']+amounts['reserved'];ceiling=CPU_LIMITS.get(category)
        categories[category]={'actual_process_seconds':str(amounts['actual']),'reserved_process_seconds':str(amounts['reserved']),
            'committed_process_seconds':str(committed),'ceiling_seconds':text_number(ceiling),
            'within_ceiling':None if ceiling is None else committed<=ceiling,'unallocated_seconds':None if ceiling is None else str(ceiling-committed)}
    return {'version':VERSION,'measurements':len(items),'cpu_categories':categories,
        'gpu_equipped_actual_process_seconds':str(gpu_actual),'gpu_equipped_reserved_process_seconds':str(gpu_reserved),
        'known_physical_gpu_allocation_seconds_by_device':{k:str(v) for k,v in sorted(physical.items())},
        'physical_gpu_allocation_device_hours':None if unknown else str(sum(physical.values(),Decimal(0))/3600),
        'gpu_measurements_without_physical_intervals':unknown,'project_consumption_complete':False,
        'scope':'declared non-overlapping measurement scopes only; GPU-equipped process elapsed sums are not kernel-active time or physical hours without intervals; CPU process-wall seconds sum across workers'}


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--usage');parser.add_argument('--output')
    args=parser.parse_args(argv)
    try:
        plan_path=Path(args.plan).resolve();plan=NFSPResourcePlan.from_dict(json.loads(plan_path.read_text()));report={'plan':plan.review()}
        inputs=[plan_path]
        if args.usage:
            path=Path(args.usage).resolve();inputs.append(path);data=json.loads(path.read_text())
            require(isinstance(data,dict) and set(data)=={'version','measurements'} and data['version']==VERSION,'unknown usage inventory')
            items=[Usage(**x) for x in data['measurements']];report['recorded_usage']=measured_usage(items)
        report['source_sha256']=sha(Path(__file__));report['input_files_sha256']={str(path):sha(path) for path in inputs}
        if args.output:
            output=Path(args.output).resolve();require(output not in inputs,'ledger output cannot overwrite its inputs')
            from graygo.manifest import write_report
            write_report(output,report)
        print(json.dumps(report,indent=2,sort_keys=True));return 0
    except (ValueError,TypeError,KeyError,OSError) as exc:parser.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())

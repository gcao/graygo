"""Preregistered NFSP development/confirmation cohort decisions (§6.3).

All thresholds are utility units. Largest completed planned looks are used;
there is no winning-seed/window selection, retrospective opening or Q/average
substitution. Missing evidence remains incomplete, never a pass.
"""
from dataclasses import dataclass
import math
from .confidence import ConfidenceFamily,canonical_sha,validate_estimate

DEVELOPMENT_SEEDS=(100,101,102)
CONFIRMATION_SEEDS=(200,201,202)
PRE=(4_500_000,4_750_000,5_000_000)
POST=(6_000_000,7_000_000,8_000_000)
OPPONENTS=('random','tactical')


def _execution_settings(value):
    from .evaluation import neural_inference_identity,pair_execution_identity
    if value is None:return {'inference_batch_size':1,'pair_batch_size':1,'inference_device':'cpu','cuda_graphs':False}
    fields={'pair_batch_size','neural_inference'}
    if not isinstance(value,dict) or set(value) not in (fields,fields|{'cuda_graphs','inference_device'}):raise ValueError('invalid preregistered evaluation execution')
    pair_execution_identity(value['pair_batch_size']);inference=value['neural_inference']
    if inference is None:batch=1;device='cpu'
    else:
        batch=inference['batch_size'];device=inference['device']
        if inference!=neural_inference_identity(batch,device):raise ValueError('unknown fixed neural inference convention')
    graphs=value.get('cuda_graphs',False)
    if 'cuda_graphs' in value:
        if graphs is not True:raise ValueError('nondefault graph execution must be explicitly selected')
        device=value['inference_device']
    return {'inference_batch_size':batch,'pair_batch_size':value['pair_batch_size'],'inference_device':device,'cuda_graphs':graphs}


def subject_label(seed,record,representation):return f'seed{seed}/{representation}/{record}'
def endpoint(seed,record,representation,opponent):return subject_label(seed,record,representation)+'/'+opponent


@dataclass(frozen=True)
class PilotProtocol:
    name: str
    random_sha256: str
    tactical_sha256: str
    training_runs: tuple
    include_average_diagnostics: bool=True
    alpha: float=.05
    looks: tuple=(500,2000)
    inference_batch_size: int=1
    pair_batch_size: int=1
    inference_device: str='cpu'
    cuda_graphs: bool=False

    def __post_init__(self):
        from .evaluation import neural_inference_identity,pair_execution_identity
        inference=neural_inference_identity(self.inference_batch_size,self.inference_device);pair_execution_identity(self.pair_batch_size)
        if type(self.cuda_graphs) is not bool:raise ValueError('explicit cohort CUDA graph selection required')
        device='cuda:0' if self.inference_device=='cuda' else self.inference_device
        if self.cuda_graphs and not (isinstance(device,str) and device.startswith('cuda:') and device[5:].isdigit()):raise ValueError('graph cohort requires a declared CUDA device')
        object.__setattr__(self,'inference_device',device if self.cuda_graphs else ('cpu' if inference is None else inference['device']))
        object.__setattr__(self,'looks',tuple(self.looks))
        registry=tuple(sorted(self.training_runs.items() if isinstance(self.training_runs,dict) else self.training_runs))
        object.__setattr__(self,'training_runs',registry)
        if tuple(seed for seed,_ in registry)!=DEVELOPMENT_SEEDS:raise ValueError('all three development run fingerprints required')
        for _,fingerprint in registry:
            if not isinstance(fingerprint,str) or len(fingerprint)!=64 or any(c not in '0123456789abcdef' for c in fingerprint):raise ValueError('invalid training run fingerprint')
        if not self.name or self.looks!=(500,2000) or self.alpha!=.05 or type(self.include_average_diagnostics) is not bool:raise ValueError('fixed pilot look plan required')
        for fingerprint in (self.random_sha256,self.tactical_sha256):
            if not isinstance(fingerprint,str) or len(fingerprint)!=64 or any(c not in '0123456789abcdef' for c in fingerprint):raise ValueError('frozen baseline SHA-256 required')
        self.family # validate alpha and complete endpoint allocation now

    @property
    def family(self):
        keys=[endpoint(s,n,'greedy_q',o) for s in DEVELOPMENT_SEEDS for n in (*PRE,*POST) for o in OPPONENTS]
        if self.include_average_diagnostics:keys += [endpoint(s,n,'average',o) for s in DEVELOPMENT_SEEDS for n in POST for o in OPPONENTS]
        return ConfidenceFamily(self.name,tuple(keys),self.looks,self.alpha)
    @property
    def sha256(self):return canonical_sha(self.to_dict())
    def to_dict(self):
        result={'version':'graygo/nfsp-pilot-gates/1','name':self.name,'seeds':list(DEVELOPMENT_SEEDS),'confirmation_seeds':list(CONFIRMATION_SEEDS),
        'training_runs':{str(s):h for s,h in self.training_runs},'pre_records':list(PRE),'post_records':list(POST),'required_complete_snapshots':2,'baselines':{'random':self.random_sha256,'tactical':self.tactical_sha256},
        'confidence':self.family.to_dict(),'average_is_diagnostic':True,'confirmation_performance_filters':False,
        'thresholds':{'random_lower_strictly_above':0.,'tactical_lower_at_least':0.},'evaluation_rules':{'size':9,'max_rounds':200,'collision':'gray'},'opening_kind':'official-empty'}
        if self.inference_batch_size>1 or self.pair_batch_size>1 or self.cuda_graphs:result['evaluation_execution']=self.execution_settings()
        return result

    def execution_settings(self):
        from .evaluation import neural_inference_identity
        result={'pair_batch_size':self.pair_batch_size,'neural_inference':neural_inference_identity(self.inference_batch_size,self.inference_device)}
        if self.cuda_graphs:result.update(cuda_graphs=True,inference_device=self.inference_device)
        return result

    def _validate_execution(self,read):
        from .evaluation import pair_execution_identity
        if read.get('subject_inference')!=self.execution_settings()['neural_inference']:raise ValueError('cohort inference layout differs from preregistration')
        expected_graphs=None
        if self.cuda_graphs:
            from .frozen_forward import execution_identity
            expected_graphs=execution_identity(True)
        if read.get('subject_frozen_execution')!=expected_graphs:raise ValueError('cohort frozen execution differs from preregistration')
        expected=pair_execution_identity(self.pair_batch_size)
        if read.get('evaluation_execution')!=expected and (read.get('evaluation_execution') is not None or self.pair_batch_size>1 or self.inference_batch_size>1 or self.cuda_graphs):
            raise ValueError('cohort pair batching differs from preregistration')

    @classmethod
    def from_dict(cls,value):
        if value.get('version')!='graygo/nfsp-pilot-gates/1':raise ValueError('unknown pilot protocol version')
        confidence=value['confidence'];include=any('/average/' in key for key in confidence['endpoints'])
        p=cls(value['name'],value['baselines']['random'],value['baselines']['tactical'],{int(s):h for s,h in value['training_runs'].items()},include,confidence['alpha'],tuple(confidence['looks']),**_execution_settings(value.get('evaluation_execution')))
        if p.to_dict()!=value:raise ValueError('pilot protocol constants/coverage changed')
        return p

    def _reads(self,reads):
        family=self.family;indexed={}
        for read in reads:
            self._validate_execution(read)
            key=read['endpoint'];look=read['games']
            validate_estimate(read,family)
            if key not in family.endpoints or look not in family.looks or read['family_sha256']!=family.sha256 or read['method']!=family.method or read['delta']!=family.delta:
                raise ValueError('pilot receipt has a different confidence plan')
            if read.get('comparison')!='utility' or read['independent_units']!=look//2 or read['support']!=[-1.,1.]:raise ValueError('pilot must use paired utility evidence')
            if not all(math.isfinite(read[k]) for k in ('mean','lower','sample_variance')) or not -1<=read['lower']<=read['mean']<=1 or read['sample_variance']<0:raise ValueError('invalid pilot estimate')
            opponent=key.rsplit('/',1)[1]
            expected=self.random_sha256 if opponent=='random' else self.tactical_sha256
            if read['opponent_sha256']!=expected or read['subject_label']!=key.rsplit('/',1)[0]:raise ValueError('pilot baseline/subject binding differs')
            representation=key.split('/')[1]
            seed=int(key.split('/')[0][4:]);record=int(key.split('/')[2]);origin=read.get('subject_origin')
            if not isinstance(origin,dict) or origin.get('training_run_sha256')!=dict(self.training_runs)[seed] or origin.get('seed')!=seed or origin.get('records')!=record:
                raise ValueError('pilot checkpoint origin differs from the registered run/clock')
            if read.get('rules')!={'size':9,'max_rounds':200,'collision':'gray'} or read.get('opening_kind')!='official-empty' or origin.get('rules')!=read['rules']:raise ValueError('pilot requires official empty 9x9 base-rule evaluation')
            if read['representation']!=representation:raise ValueError('wrong deployed policy representation')
            if read.get('available_games_per_pairing',look)<look:raise ValueError('receipt exceeds available evidence')
            if not all(isinstance(read.get(k),str) and len(read[k])==64 for k in ('evidence_sha256','plan_sha256','subject_sha256')):raise ValueError('missing evaluation evidence fingerprints')
            token=(key,look)
            if token in indexed:raise ValueError('duplicate pilot receipt')
            indexed[token]=read
        return indexed

    def window(self,reads,seed,*,window):
        if seed not in DEVELOPMENT_SEEDS or window not in ('pre','post'):raise ValueError('fixed development seed/window required')
        indexed=self._reads(reads);snapshots=[]
        for record in PRE if window=='pre' else POST:
            selected={};missing=[]
            for opponent in OPPONENTS:
                key=endpoint(seed,record,'greedy_q',opponent)
                candidates=[v for (e,_),v in indexed.items() if e==key]
                if not candidates:missing.append(opponent);continue
                chosen=max(candidates,key=lambda v:v['games'])
                completed=max(n for n in self.looks if n<=chosen.get('available_games_per_pairing',chosen['games']))
                if chosen['games']!=completed:raise ValueError('a later completed planned look was hidden')
                selected[opponent]=chosen
            if missing:status='incomplete'
            else:
                if selected['random']['subject_sha256']!=selected['tactical']['subject_sha256']:raise ValueError('baseline reads use different checkpoint weights')
                clears=selected['random']['lower']>0 and selected['tactical']['lower']>=0
                if clears:status='pass'
                elif all(v['games']==self.looks[-1] for v in selected.values()):status='no_go'
                else:status='inconclusive'
            snapshots.append({'record':record,'status':status,'reads':selected,'missing':missing})
        passes=sum(s['status']=='pass' for s in snapshots)
        # All three preregistered snapshots must be recorded; two positive reads
        # are insufficient if the third was quietly omitted.
        observed=all(not s['missing'] for s in snapshots)
        complete=all(s['status'] in ('pass','no_go') for s in snapshots)
        decision='pass' if observed and passes>=2 else ('no_go' if complete else 'incomplete')
        return {'seed':seed,'window':window,'status':decision,'complete_snapshot_passes':passes,'snapshots':snapshots,'protocol_sha256':self.sha256}

    def cohort(self,reads):
        windows=[self.window(reads,seed,window=w) for seed in DEVELOPMENT_SEEDS for w in ('pre','post')]
        state='pass' if all(w['status']=='pass' for w in windows) else ('no_go' if any(w['status']=='no_go' for w in windows) else 'incomplete')
        indexed=self._reads(reads)
        required=[endpoint(s,n,'average',o) for s in DEVELOPMENT_SEEDS for n in POST for o in OPPONENTS]
        missing=[key for key in required if not any(e==key for e,_ in indexed)]
        if state=='pass' and missing:state='incomplete'
        return {'status':state,'windows':windows,'seeds':list(DEVELOPMENT_SEEDS),'protocol_sha256':self.sha256,'a2_complete':False,'deployed_strength_established':False,
                'average_diagnostic_endpoints_in_family':self.include_average_diagnostics,'missing_average_diagnostics':missing}

    def label_insertion(self,reads,seed,records,*,scope,opening_event=None):
        if type(records) is not int or records<0:raise ValueError('aggregate record clock required')
        if scope=='confirmation':
            if seed not in CONFIRMATION_SEEDS:raise ValueError('independent confirmation seed required')
            return {'allowed':records>=5_000_000,'reason':'fixed confirmation opening; performance is diagnostic','protocol_sha256':self.sha256}
        if scope!='development' or seed not in DEVELOPMENT_SEEDS:raise ValueError('declared scope/cohort required')
        if records<5_000_000:return {'allowed':False,'reason':'before fixed opening','protocol_sha256':self.sha256}
        decision=self.window(reads,seed,window='pre')
        allowed=decision['status']=='pass';reason=decision['status']
        if allowed and records>5_000_000:
            expected={'seed':seed,'records':5_000_000,'protocol_sha256':self.sha256,'status':'pass'}
            if not isinstance(opening_event,dict) or any(opening_event.get(k)!=v for k,v in expected.items()):allowed=False;reason='no recorded pass at the fixed opening boundary'
        event=None if not allowed else {'seed':seed,'records':5_000_000,'protocol_sha256':self.sha256,'status':'pass'}
        return {'allowed':allowed,'reason':reason,'protocol_sha256':self.sha256,'opening_event':event}


class ScreeningProtocol:
    def __init__(self,name,random_sha256,tactical_sha256,training_runs,*,alpha=.05,looks=(500,2000),inference_batch_size=1,pair_batch_size=1,inference_device='cpu',cuda_graphs=False):
        self.pilot=PilotProtocol(name,random_sha256,tactical_sha256,training_runs,False,alpha,looks,inference_batch_size,pair_batch_size,inference_device,cuda_graphs)
        self.family=ConfidenceFamily(name,tuple(endpoint(s,20_000_000,'average',o) for s in DEVELOPMENT_SEEDS for o in OPPONENTS),tuple(looks),alpha)
    def to_dict(self):
        result={'version':'graygo/nfsp-screening-gates/1','name':self.family.name,'confidence':self.family.to_dict(),
            'training_runs':{str(s):h for s,h in self.pilot.training_runs},'baselines':{'random':self.pilot.random_sha256,'tactical':self.pilot.tactical_sha256},
            'seeds':list(DEVELOPMENT_SEEDS),'records':20_000_000,'representation':'average',
            'thresholds':{'random_lower_strictly_above':0.,'tactical_lower_at_least':.1},'confirmation_seed_filtering':False}
        if self.pilot.inference_batch_size>1 or self.pilot.pair_batch_size>1 or self.pilot.cuda_graphs:result['evaluation_execution']=self.pilot.execution_settings()
        return result

    @classmethod
    def from_dict(cls,value):
        if value.get('version')!='graygo/nfsp-screening-gates/1':raise ValueError('unknown screening protocol version')
        c=value['confidence'];p=cls(value['name'],value['baselines']['random'],value['baselines']['tactical'],{int(s):h for s,h in value['training_runs'].items()},alpha=c['alpha'],looks=tuple(c['looks']),**_execution_settings(value.get('evaluation_execution')))
        if p.to_dict()!=value:raise ValueError('screening protocol constants/coverage changed')
        return p

    def decide(self,reads):
        indexed={}
        for read in reads:
            self.pilot._validate_execution(read)
            validate_estimate(read,self.family)
            key=read['endpoint']
            if key not in self.family.endpoints or read['games'] not in self.family.looks or read['family_sha256']!=self.family.sha256 or read['delta']!=self.family.delta or read['representation']!='average' or read.get('comparison')!='utility' or read['independent_units']!=read['games']//2 or read.get('support')!=[-1.,1.]:raise ValueError('screening estimate is from a different protocol or policy')
            seed=int(key.split('/')[0][4:]);origin=read.get('subject_origin')
            if not isinstance(origin,dict) or origin.get('training_run_sha256')!=dict(self.pilot.training_runs)[seed] or origin.get('seed')!=seed or origin.get('records')!=20_000_000 or read.get('rules')!={'size':9,'max_rounds':200,'collision':'gray'} or origin.get('rules')!=read['rules'] or read.get('opening_kind')!='official-empty':raise ValueError('screening checkpoint/start provenance differs')
            opponent=key.rsplit('/',1)[1];expected=self.pilot.random_sha256 if opponent=='random' else self.pilot.tactical_sha256
            if read['opponent_sha256']!=expected or read['subject_label']!=key.rsplit('/',1)[0]:raise ValueError('screening binding differs')
            if (key,read['games']) in indexed:raise ValueError('duplicate screening receipt')
            if not all(isinstance(read.get(k),str) and len(read[k])==64 for k in ('evidence_sha256','plan_sha256','subject_sha256')):raise ValueError('missing screening evidence fingerprints')
            indexed[key,read['games']]=read
        results=[]
        for seed in DEVELOPMENT_SEEDS:
            chosen={}
            for opponent in OPPONENTS:
                candidates=[v for (key,_),v in indexed.items() if key==endpoint(seed,20_000_000,'average',opponent)]
                if candidates:
                    read=max(candidates,key=lambda v:v['games'])
                    available=read.get('available_games_per_pairing',read['games'])
                    if read['games']!=max(n for n in self.family.looks if n<=available):raise ValueError('later screening look was omitted')
                    chosen[opponent]=read
            if len(chosen)<2:status='incomplete'
            else:
                if chosen['random']['subject_sha256']!=chosen['tactical']['subject_sha256']:raise ValueError('screening opponents use different average weights')
                if chosen['random']['lower']>0 and chosen['tactical']['lower']>=.1:status='pass'
                elif all(v['games']==self.family.looks[-1] for v in chosen.values()):status='no_go'
                else:status='incomplete'
            results.append({'seed':seed,'status':status,'reads':chosen})
        return {'status':'pass' if all(r['status']=='pass' for r in results) else ('no_go' if any(r['status']=='no_go' for r in results) else 'incomplete'),
                'seeds':results,'confidence_family_sha256':self.family.sha256,'confirmation_seed_filtering':False,'a2_complete':False}

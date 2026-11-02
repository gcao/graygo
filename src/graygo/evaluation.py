"""Frozen, paired-color evaluation with private actions and resumable evidence.

One unit exchanges the agents' colors on the same complete public root.
Candidate/incumbent use common seeds/openings against the same frozen opponent.
Repeated trajectories remain observations; only duplicate UNIT IDs are forbidden.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sqlite3
import time
import numpy as np
from .agents import uniform,tactical,BASELINE_VERSION
from .confidence import canonical_sha
from .encoding import ENCODING_VERSION,encode
from .engine import BLACK,WHITE,RULE_VERSION,Rules,State,resolve,score,transform
from .manifest import source_digest,write_report
from .storage import DEFAULT_RESERVE,DiskBudgetError

EVALUATION_VERSION='graygo/paired-evaluation/2'


def seed_for(seed,*parts):
    if type(seed) is not int or seed<0:raise ValueError('nonnegative integral evaluation seed required')
    return int.from_bytes(hashlib.sha256(json.dumps([seed,*parts],separators=(',',':')).encode()).digest()[:16],'little')


class BaselinePolicy:
    def __init__(self,name):
        if name not in ('random','tactical'):raise ValueError('unknown frozen baseline')
        self.name=name
        self.identity={'kind':name,'version':BASELINE_VERSION,'source_sha256':source_digest()}
        self.sha256=canonical_sha(self.identity)
    def begin_episode(self,seed):return _BaselineActor(self.name,seed)
    def assert_frozen(self):
        if canonical_sha(self.identity)!=self.sha256 or self.identity['kind']!=self.name or self.identity['version']!=BASELINE_VERSION or self.identity['source_sha256']!=source_digest():raise ValueError('frozen baseline identity changed')


class _BaselineActor:
    def __init__(self,name,seed):self.function=uniform if name=='random' else tactical;self.rng=random.Random(seed);self.component=None
    def action(self,state,player):return self.function(state,player,self.rng)


class FrozenNeuralPolicy:
    def __init__(self,model,*,representation='average',label='frozen-policy',origin=None):
        import torch
        if representation not in ('average','actor','greedy_q'):raise ValueError('explicit deployed representation required')
        self._model=deepcopy(model).eval().requires_grad_(False);self.representation=representation;self.label=label
        devices={x.device for x in list(self._model.parameters())+list(self._model.buffers())}
        if len(devices)!=1:raise ValueError('one model device required')
        self.device=next(iter(devices));self._weights=self._weight_sha()
        self.identity={'kind':representation,'model_class':type(model).__name__,'weights_sha256':self._weights,'encoding_version':ENCODING_VERSION,'source_sha256':source_digest(),'origin':None if origin is None else json.loads(json.dumps(origin))}
        self.sha256=canonical_sha(self.identity)
    @classmethod
    def from_archive(cls,archive,*,representation='average',device='cpu',label='checkpoint'):
        import torch
        from .models import QNet,PolicyNet,ModelConfig
        from .storage import CheckpointStore
        directory=Path(archive).resolve()
        if directory.is_file():directory=directory.parent
        if directory.parent.name!='weights' or representation not in ('average','greedy_q'):raise ValueError('NFSP public weights archive and explicit policy required')
        root=directory.parents[1];store=CheckpointStore(root);sealed=store.verify(directory)
        manifest=json.loads((root/'run.json').read_text());data=torch.load(directory/'models.pt',weights_only=True,map_location=device)
        if manifest.get('rules') is None or data['config']!=manifest['config'] or sealed['metadata']['source_sha256']!=manifest['source_sha256']:raise ValueError('archive provenance/configuration differs')
        config=data['config'];shape=ModelConfig(config.get('width',64),config.get('blocks',8),config.get('norm_groups',8))
        model=(QNet(shape) if representation=='greedy_q' else PolicyNet(shape)).to(device)
        model.load_state_dict(data['q' if representation=='greedy_q' else 'average'])
        origin={'training_run_sha256':hashlib.sha256((root/'run.json').read_bytes()).hexdigest(),'source_sha256':manifest['source_sha256'],
            'archive_sha256':hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),'records':sealed['step'],'seed':config['seed'],'rules':manifest['rules']}
        return cls(model,representation=representation,label=label,origin=origin)

    def _weight_sha(self):
        h=hashlib.sha256()
        for name,value in sorted(self._model.state_dict().items()):
            a=value.detach().cpu().contiguous().numpy();h.update(name.encode()+b'\0'+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
        return h.hexdigest()
    def assert_frozen(self):
        if canonical_sha(self.identity)!=self.sha256 or self.identity['kind']!=self.representation or self.identity['source_sha256']!=source_digest() or self._weight_sha()!=self._weights:raise ValueError('evaluation model changed after freezing')
    def _action_weights(self,state,player):
        import torch
        from .models import greedy_distribution,masked_log_probs
        p=self;o=encode(state,player)
        with torch.no_grad():
            x=torch.tensor(o.spatial[None],device=p.device,dtype=torch.float32);g=torch.tensor(o.global_features[None],device=p.device,dtype=torch.float32)
            legal=torch.tensor(o.legal[None],device=p.device)
            if p.representation=='greedy_q':
                q=p._model(x,g)
                if q.shape!=legal.shape or not bool(torch.isfinite(q[legal]).all()) or bool((q[legal].abs()>1+1e-6).any()):raise ValueError('invalid legal Q values')
                probs=greedy_distribution(q,legal)
            else:
                logits=p._model.policy_logits(x,g) if hasattr(p._model,'policy_logits') else p._model(x,g)
                if isinstance(logits,tuple):logits=logits[0]
                if logits.shape!=legal.shape or bool(torch.isnan(logits[legal]).any()) or bool(torch.isposinf(logits[legal]).any()):raise ValueError('invalid actor logits')
                probs=masked_log_probs(logits,legal).exp()
        return probs.cpu().double().numpy()[0]

    def initial_distribution(self,state,player,*,fresh_episode=False):
        # Same normalization as the private sampler, without drawing an action.
        p=self._action_weights(state,player);p/=p.sum();return p

    def begin_episode(self,seed):return _NeuralActor(self,seed)


class _NeuralActor:
    def __init__(self,policy,seed):self.policy=policy;self.rng=np.random.default_rng(seed);self.component=None
    def action(self,state,player):
        from .sampling import draw_actions
        action,_=draw_actions(self.policy._action_weights(state,player)[None],self.rng)
        return int(action[0])


class EpisodeMixture:
    def __init__(self,components,weights,*,label='episode-mixture'):
        self.components=tuple(components);self.weights=np.asarray(weights,dtype=np.float64).copy()
        if not self.components or self.weights.shape!=(len(self.components),) or not np.isfinite(self.weights).all() or (self.weights<0).any() or abs(self.weights.sum()-1)>1e-12:raise ValueError('normalized episode component weights required')
        self.weights/=self.weights.sum();self.weights.flags.writeable=False
        self.identity={'kind':'independent-episode-mixture','label':label,'components':[c.sha256 for c in self.components],'weights':self.weights.tolist()}
        self.sha256=canonical_sha(self.identity)
    def assert_frozen(self):
        if canonical_sha(self.identity)!=self.sha256 or self.identity['components']!=[c.sha256 for c in self.components] or self.identity['weights']!=self.weights.tolist():raise ValueError('episode mixture changed')
        for c in self.components:c.assert_frozen()
    def initial_distribution(self,state,player,*,fresh_episode=False):
        if (state.rounds_played or state.history) and not fresh_episode:raise ValueError('continuation mixture needs a public-prefix-conditioned marginal')
        if not all(hasattr(c,'initial_distribution') for c in self.components):raise ValueError('component lacks exact public initial marginals')
        values=[np.asarray(c.initial_distribution(state,player,fresh_episode=fresh_episode),dtype=np.float64) for c in self.components]
        result=sum(weight*value for weight,value in zip(self.weights,values));result/=result.sum();return result

    def begin_episode(self,seed):
        chosen=int(np.random.default_rng(seed_for(seed,'component')).choice(len(self.components),p=self.weights))
        actor=self.components[chosen].begin_episode(seed_for(seed,'actions'));actor.component=self.components[chosen].sha256
        return actor


class EmptyStarts:
    def __init__(self):self.identity={'kind':'official-empty','version':1};self.sha256=canonical_sha(self.identity)
    def sample(self,seed,rules):return State(rules)


class FixedStarts:
    """A frozen independently generated opening catalog, not a prefix restart."""
    def __init__(self,states,*,label):
        self.states=tuple(states)
        if not self.states or any(not isinstance(s,State) or s.done for s in self.states):raise ValueError('nonterminal complete public openings required')
        self.identity={'kind':'fixed-public-opening-catalog','label':label,'states':[s.to_dict() for s in self.states]}
        self.sha256=canonical_sha(self.identity)
    def sample(self,seed,rules):
        if any(s.rules!=rules for s in self.states):raise ValueError('opening catalog rules differ')
        return self.states[int(np.random.default_rng(seed).integers(len(self.states)))]


def play_frozen(subject,opponent,root,player,*,subject_seed,opponent_seed):
    if player not in (BLACK,WHITE) or root.done:raise ValueError('nonterminal root and player required')
    a=subject.begin_episode(subject_seed);b=opponent.begin_episode(opponent_seed)
    state=root;h=hashlib.sha256(root.canonical_json().encode());rounds=0
    while not state.done:
        # Both inputs are the same immutable public state; no commitment is an input.
        mine=a.action(state,player);theirs=b.action(state,3-player)
        black,white=(mine,theirs) if player==BLACK else (theirs,mine)
        if black not in state.legal_actions(BLACK) or white not in state.legal_actions(WHITE):raise ValueError('evaluation policy chose an illegal move')
        h.update(json.dumps([black,white],separators=(',',':')).encode());state=resolve(state,black,white).state;rounds+=1
    result=score(state.board,state.rules.size);black=float(result.utility_black)
    return {'subject_utility':black if player==BLACK else -black,'black_utility':black,'rounds':rounds,
            'termination':state.terminal_reason,'black_score_units':result.black_units,'white_score_units':result.white_units,
            'trajectory_sha256':h.hexdigest(),'subject_component':getattr(a,'component',None),'opponent_component':getattr(b,'component',None)}


def evaluate_pair(subject,opponent,root,*,unit_seed,incumbent=None):
    output={'root_sha256':hashlib.sha256(root.canonical_json().encode()).hexdigest(),'subject':[],'incumbent':[]}
    for color,player in enumerate((BLACK,WHITE)):
        # Exchange the agents' colors on the same complete opening. Swapping
        # both board and agents would preserve an asymmetric starting advantage.
        opening=root
        subject_seed=seed_for(unit_seed,'subject',color);opponent_seed=seed_for(unit_seed,'opponent',opponent.sha256,color)
        output['subject'].append(play_frozen(subject,opponent,opening,player,subject_seed=subject_seed,opponent_seed=opponent_seed))
        if incumbent is not None:output['incumbent'].append(play_frozen(incumbent,opponent,opening,player,subject_seed=subject_seed,opponent_seed=opponent_seed))
    output['utility']=sum(g['subject_utility'] for g in output['subject'])/2
    output['incumbent_utility']=None if incumbent is None else sum(g['subject_utility'] for g in output['incumbent'])/2
    output['improvement']=None if incumbent is None else output['utility']-output['incumbent_utility']
    return output


def validate_pair(row,rules,has_incumbent):
    if not isinstance(row.get('root_sha256'),str) or len(row['root_sha256'])!=64:raise ValueError('invalid opening fingerprint')
    for key,expected in (('subject',2),('incumbent',2 if has_incumbent else 0)):
        games=row.get(key)
        if not isinstance(games,list) or len(games)!=expected:raise ValueError('paired game evidence is incomplete')
        for color,g in enumerate(games):
            if g['subject_utility'] not in (-1.,0.,1.) or g['black_utility'] not in (-1.,0.,1.) or g['subject_utility']!=g['black_utility']*(1 if color==0 else -1):raise ValueError('paired utility sign differs')
            if type(g['rounds']) is not int or not 1<=g['rounds']<=rules.max_rounds or g['termination'] not in ('mutual_pass','round_limit'):raise ValueError('invalid game termination')
            b,w=g['black_score_units'],g['white_score_units']
            if type(b) is not int or type(w) is not int or b<0 or w<0 or b+w!=4*rules.points or g['black_utility']!=int(b>w)-int(b<w):raise ValueError('score/utility accounting differs')
            if not isinstance(g['trajectory_sha256'],str) or len(g['trajectory_sha256'])!=64:raise ValueError('missing trajectory fingerprint')
    mean=sum(g['subject_utility'] for g in row['subject'])/2
    incumbent=None if not has_incumbent else sum(g['subject_utility'] for g in row['incumbent'])/2
    if row['utility']!=mean or row['incumbent_utility']!=incumbent or row['improvement']!=(None if incumbent is None else mean-incumbent):raise ValueError('paired aggregate differs from scored games')


@dataclass(frozen=True)
class EvaluationPlan:
    label: str
    subject_label: str
    endpoint: str
    rules: Rules=Rules()
    seed: int=0
    looks: tuple=(500,2000)
    confidence_sha256: str|None=None
    comparison: str='utility'
    decision_protocol_sha256: str|None=None
    calibration_sha256: str|None=None

    def __post_init__(self):
        object.__setattr__(self,'looks',tuple(self.looks))
        if not all(isinstance(v,str) and v for v in (self.label,self.subject_label,self.endpoint)) or type(self.seed) is not int or self.seed<0:raise ValueError('named fixed evaluation labels and seed required')
        if not isinstance(self.confidence_sha256,str) or len(self.confidence_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.confidence_sha256) or self.comparison not in ('utility','improvement'):raise ValueError('predeclared confidence-family fingerprint and endpoint type required')
        if (self.decision_protocol_sha256 is None)!=(self.calibration_sha256 is None):raise ValueError('protected protocol and pre-candidate calibration must be bound together')
        for value in (self.decision_protocol_sha256,self.calibration_sha256):
            if value is not None and (not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value)):raise ValueError('invalid protected protocol/calibration fingerprint')
        if not self.looks or tuple(sorted(set(self.looks)))!=self.looks or any(type(n) is not int or n<4 or n%2 for n in self.looks):raise ValueError('fixed increasing color-balanced looks required')


class EvaluationRun:
    def __init__(self,plan,root,subject,opponent,*,openings=None,incumbent=None,resume=False,reserve_bytes=DEFAULT_RESERVE):
        self.plan=plan;self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        if plan.comparison=='improvement' and incumbent is None:raise ValueError('matched improvement plan requires an incumbent')
        self.subject,self.opponent,self.incumbent=subject,opponent,incumbent;self.openings=openings or EmptyStarts();self.reserve=reserve_bytes
        for p in (subject,opponent,incumbent):
            if p is not None:p.assert_frozen()
        self.contract={'evaluation_version':EVALUATION_VERSION,'plan':json.loads(json.dumps(asdict(plan))),
            'subject':subject.identity,'subject_sha256':subject.sha256,'opponent':opponent.identity,'opponent_sha256':opponent.sha256,
            'incumbent':None if incumbent is None else incumbent.identity,'incumbent_sha256':None if incumbent is None else incumbent.sha256,
            'opening_distribution':self.openings.identity,'opening_sha256':self.openings.sha256,'rule_version':RULE_VERSION,'source_sha256':source_digest(),
            'unit':'agents exchange colors on the same complete opening; within-pair dependence retained; no trajectory deduplication','opening_semantics':'declared new episode roots; continuation-prefix mixtures require strategies.py'}
        self.sha256=canonical_sha(self.contract);path=self.root/'plan.json';self.database=self.root/'evidence.sqlite'
        with self._lock():
            if resume:
                if json.loads(path.read_text())!=self.contract:raise ValueError('evaluation plan/policy/source/openings differ on resume')
                if not self.database.is_file():raise ValueError('committed evaluation database is missing')
            else:
                if path.exists() or self.database.exists():raise FileExistsError('evaluation exists; use explicit resume')
                db=self._connect()
                with db:db.execute('CREATE TABLE units (idx INTEGER PRIMARY KEY, uid TEXT UNIQUE NOT NULL, payload TEXT NOT NULL, sha256 TEXT NOT NULL)')
                db.close();write_report(path,self.contract,reserve_bytes=reserve_bytes)
            self.records()

    def _connect(self):
        db=sqlite3.connect(self.database);db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA journal_mode=DELETE');return db
    def _lock(self):return _EvaluationLock(self.root/'.evaluation.lock')

    def records(self):
        db=self._connect()
        try:rows=list(db.execute('SELECT idx,uid,payload,sha256 FROM units ORDER BY idx'))
        finally:db.close()
        result=[]
        for index,(i,uid,payload,checksum) in enumerate(rows):
            expected=str(seed_for(self.plan.seed,'unit',i))
            if i!=index or uid!=expected or hashlib.sha256(payload.encode()).hexdigest()!=checksum:raise ValueError('evaluation evidence prefix/checksum differs')
            row=json.loads(payload)
            if row['index']!=i or row['uid']!=uid:raise ValueError('evaluation unit identity differs')
            validate_pair(row,self.plan.rules,self.incumbent is not None)
            opening=self.openings.sample(seed_for(int(uid),'opening'),self.plan.rules)
            if row['root_sha256']!=hashlib.sha256(opening.canonical_json().encode()).hexdigest():raise ValueError('recorded opening differs from frozen sampling plan')
            result.append(row)
        return result

    def run(self,games,*,time_cap_seconds,stop_requested=lambda:False):
        if games not in self.plan.looks or not math.isfinite(time_cap_seconds) or time_cap_seconds<=0:raise ValueError('planned look and explicit time ceiling required')
        with self._lock():
            for p in (self.subject,self.opponent,self.incumbent):
                if p is not None:p.assert_frozen()
            if self.contract['source_sha256']!=source_digest():raise ValueError('evaluation source changed')
            rows=self.records();target=games//2;started=time.monotonic();status='complete'
            if len(rows)>max(self.plan.looks)//2:raise ValueError('evaluation exceeds its declared maximum look')
            for index in range(len(rows),target):
                if stop_requested():status='interrupted';break
                if time.monotonic()-started>=time_cap_seconds:status='budget_limited';break
                if shutil.disk_usage(self.root).free-65536<self.reserve:raise DiskBudgetError('evaluation would violate disk reserve')
                unit=seed_for(self.plan.seed,'unit',index);opening=self.openings.sample(seed_for(unit,'opening'),self.plan.rules)
                result=evaluate_pair(self.subject,self.opponent,opening,unit_seed=unit,incumbent=self.incumbent)
                row={'index':index,'uid':str(unit),**result};payload=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)
                db=self._connect()
                try:
                    with db:db.execute('INSERT INTO units VALUES (?,?,?,?)',(index,str(unit),payload,hashlib.sha256(payload.encode()).hexdigest()))
                finally:db.close()
                rows.append(row)
            for p in (self.subject,self.opponent,self.incumbent):
                if p is not None:p.assert_frozen()
            if len(rows)<target and status=='complete':status='incomplete'
            result={'status':status,'committed_pairs':len(rows),'physical_games':len(rows)*(4 if self.incumbent is not None else 2),
                'requested_look_games_per_pairing':games,'plan_sha256':self.sha256,'elapsed_seconds':time.monotonic()-started,'scope':'frozen tested matchup; not full-game exploitability or training-seed certainty'}
            write_report(self.root/'progress.json',result,reserve_bytes=self.reserve);return result

    def estimate(self,family,games,*,comparison=None):
        if family.sha256!=self.plan.confidence_sha256 or family.looks!=self.plan.looks:raise ValueError('confidence/evaluation plans differ')
        comparison=self.plan.comparison if comparison is None else comparison
        if comparison!=self.plan.comparison:raise ValueError('endpoint type changed after evaluation planning')
        if comparison not in ('utility','improvement'):raise ValueError('unknown paired endpoint')
        available=self.records();rows=available[:games//2]
        if len(rows)!=games//2 or games not in self.plan.looks:raise ValueError('planned look has incomplete evidence')
        if comparison=='improvement' and self.incumbent is None:raise ValueError('improvement requires matched incumbent games')
        values=[r[comparison] for r in rows];bound=family.estimate(self.plan.endpoint,games,values,minimum=-2. if comparison=='improvement' else -1.,maximum=2. if comparison=='improvement' else 1.)
        hashes=[g['trajectory_sha256'] for r in rows for g in r['subject']]
        evidence=canonical_sha([canonical_sha(r) for r in rows])
        return dict(bound,comparison=comparison,plan_sha256=self.sha256,available_games_per_pairing=2*len(available),representation=self.subject.identity['kind'],evidence_sha256=evidence,
            subject_label=self.plan.subject_label,subject_sha256=self.subject.sha256,opponent_sha256=self.opponent.sha256,
            subject_origin=self.subject.identity.get('origin'),rules=self.contract['plan']['rules'],opening_kind=self.openings.identity['kind'],
            repeated_trajectory_fraction=1-len(set(hashes))/len(hashes),scope='fixed policies and opening distribution; color pairs are independent units')


class _EvaluationLock:
    def __init__(self,path):self.path=path
    def __enter__(self):
        self.file=self.path.open('a+b')
        try:fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:self.file.close();raise
        return self
    def __exit__(self,*args):fcntl.flock(self.file,fcntl.LOCK_UN);self.file.close()

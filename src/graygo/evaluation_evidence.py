"""Read-only scored evaluation snapshots and locks for formal gate readers."""
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
from .confidence import canonical_sha
from .engine import RULE_VERSION,Rules,State
from .evaluation import EVALUATION_VERSION,EvaluationPlan,EmptyStarts,FixedStarts,seed_for,validate_pair,pair_execution_identity


@contextmanager
def evaluation_read_lock(root):
    """Block collection without creating or modifying any input artifact."""
    with (Path(root)/'.evaluation.lock').open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def _component(identity,seed):
    if identity['kind']!='independent-episode-mixture':return None
    weights=np.asarray(identity['weights'],dtype=float);components=identity['components']
    if (weights.shape!=(len(components),) or not len(components) or not np.isfinite(weights).all()
            or (weights<0).any() or abs(weights.sum()-1)>1e-12):
        raise ValueError('invalid frozen episode-mixture identity')
    return components[int(np.random.default_rng(seed_for(seed,'component')).choice(len(components),p=weights))]


@dataclass(frozen=True)
class FrozenEvaluationEvidence:
    root:Path
    contract:dict
    rows:tuple
    sha256:str

    @classmethod
    def load(cls,root):
        root=Path(root).resolve();plan_path=root/'plan.json';database=root/'evidence.sqlite'
        if not database.is_file():raise ValueError('committed evaluation database is missing')
        original=plan_path.read_bytes();contract=json.loads(original)
        if contract.get('evaluation_version')!=EVALUATION_VERSION or contract.get('rule_version')!=RULE_VERSION:
            raise ValueError('unsupported evaluation/rule version')
        data=dict(contract['plan']);data['rules']=Rules(**data['rules']);plan=EvaluationPlan(**data)
        incumbent=contract['incumbent'];has_incumbent=incumbent is not None
        if 'pair_batch_size' in contract['plan'] or 'execution' in contract:
            if contract.get('execution')!=pair_execution_identity(plan.pair_batch_size,has_incumbent):raise ValueError('evaluation execution/budget boundary differs')
        if plan.comparison=='improvement' and not has_incumbent:raise ValueError('matched improvement evidence lacks its incumbent')
        for key in ('subject','opponent','incumbent'):
            identity=contract[key]
            if (None if identity is None else canonical_sha(identity))!=contract[key+'_sha256']:
                raise ValueError('frozen evaluation policy identity differs')
        opening=contract['opening_distribution']
        if canonical_sha(opening)!=contract['opening_sha256']:raise ValueError('frozen opening identity differs')
        if opening['kind']=='official-empty':
            sampler=EmptyStarts()
        elif opening['kind']=='fixed-public-opening-catalog':
            sampler=FixedStarts(tuple(State.from_dict(s) for s in opening['states']),label=opening['label'])
        else:raise ValueError('unsupported read-only opening distribution')
        if sampler.identity!=opening:raise ValueError('opening schema differs')
        db=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)
        try:
            db.execute('BEGIN')
            stored=list(db.execute('SELECT idx,uid,payload,sha256 FROM units ORDER BY idx'))
        finally:db.close()
        if len(stored)>plan.looks[-1]//2:raise ValueError('evaluation exceeds its maximum registered look')
        rows=[]
        for index,(i,uid,payload,checksum) in enumerate(stored):
            unit=seed_for(plan.seed,'unit',index)
            if i!=index or uid!=str(unit) or hashlib.sha256(payload.encode()).hexdigest()!=checksum:
                raise ValueError('evaluation evidence prefix/checksum differs')
            row=json.loads(payload)
            if row['index']!=index or row['uid']!=uid:raise ValueError('evaluation unit metadata differs')
            validate_pair(row,plan.rules,has_incumbent)
            state=sampler.sample(seed_for(unit,'opening'),plan.rules)
            if row['root_sha256']!=hashlib.sha256(state.canonical_json().encode()).hexdigest():
                raise ValueError('scored evaluation opening differs from its plan')
            for key,identity in (('subject',contract['subject']),('incumbent',incumbent)):
                for color,game in enumerate(row[key]):
                    if game['termination']=='round_limit' and game['rounds']!=plan.rules.max_rounds-state.rounds_played:
                        raise ValueError('round-limit game length differs from its complete opening')
                    if game['termination']=='mutual_pass' and game['rounds']<2-state.pass_streak:
                        raise ValueError('mutual-pass game ended before its required pass streak')
                    mine=seed_for(unit,'subject',color);theirs=seed_for(unit,'opponent',contract['opponent_sha256'],color)
                    if game['subject_component']!=_component(identity,mine) or game['opponent_component']!=_component(contract['opponent'],theirs):
                        raise ValueError('private episode component differs from its declared draw')
            rows.append(row)
        if plan_path.read_bytes()!=original:raise ValueError('evaluation plan changed while reading evidence')
        value={'contract':contract,'units':[canonical_sha(row) for row in rows]}
        return cls(root,contract,tuple(rows),canonical_sha(value))

    @property
    def plan(self):
        data=dict(self.contract['plan']);data['rules']=Rules(**data['rules']);return EvaluationPlan(**data)

    @property
    def available_games(self):return 2*len(self.rows)

    @property
    def largest_completed_look(self):
        return max((n for n in self.plan.looks if n<=self.available_games),default=None)

    def assert_current(self):
        value={'contract':self.contract,'units':[canonical_sha(row) for row in self.rows]}
        if canonical_sha(value)!=self.sha256:raise ValueError('evaluation snapshot was mutated in memory')
        if self.load(self.root).sha256!=self.sha256:raise ValueError('evaluation changed after its snapshot; read its latest completed look')

    def estimate(self,family,games,*,check_current=True):
        if check_current:self.assert_current()
        plan=self.plan
        if family.sha256!=plan.confidence_sha256 or family.looks!=plan.looks:raise ValueError('confidence/evaluation plan differs')
        if games not in plan.looks or games>self.available_games:raise ValueError('planned look has incomplete scored evidence')
        rows=self.rows[:games//2];comparison=plan.comparison
        bound=family.estimate(plan.endpoint,games,[row[comparison] for row in rows],
                              minimum=-2. if comparison=='improvement' else -1.,maximum=2. if comparison=='improvement' else 1.)
        hashes=[game['trajectory_sha256'] for row in rows for game in row['subject']]
        return dict(bound,comparison=comparison,plan_sha256=canonical_sha(self.contract),available_games_per_pairing=self.available_games,
            subject_inference=self.contract['subject'].get('inference'),subject_frozen_execution=self.contract['subject'].get('frozen_execution'),evaluation_execution=self.contract.get('execution'),
            representation=self.contract['subject']['kind'],evidence_sha256=canonical_sha([canonical_sha(row) for row in rows]),
            subject_label=plan.subject_label,subject_sha256=self.contract['subject_sha256'],opponent_sha256=self.contract['opponent_sha256'],
            subject_origin=self.contract['subject'].get('origin'),rules=self.contract['plan']['rules'],opening_kind=self.contract['opening_distribution']['kind'],
            repeated_trajectory_fraction=1-len(set(hashes))/len(hashes),scope='fixed policies and opening distribution; color pairs are independent units')

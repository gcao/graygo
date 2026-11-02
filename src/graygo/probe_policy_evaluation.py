"""Transactional frozen-policy marginals, exact probe grades and replay audit."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from .confidence import canonical_sha
from .curriculum import ExposureCatalog
from .engine import RULE_VERSION
from .evaluation import FrozenNeuralPolicy,_EvaluationLock
from .manifest import make_manifest,source_digest,write_report
from .pilot_gates import CONFIRMATION_SEEDS
from .probe_benchmark import ProbeBenchmark,assess_marginals
from .probe_policy_protocol import ProbePolicyProtocol,PROBE_POLICY_VERSION
from .science_evaluation import _connect,_read_connect,_meta_read,_meta_write,_runtime_identity,_archive_provenance
from .storage import DEFAULT_RESERVE,DiskBudgetError,digest

PROBE_RUN_VERSION='graygo/frozen-probe-inference/1'


def _freeze(benchmark,receipt,protocol):
    if (benchmark.sha256!=protocol.benchmark_sha256 or canonical_sha(receipt)!=protocol.freeze_sha256
            or receipt.get('benchmark_sha256')!=benchmark.sha256 or receipt.get('families')!=32
            or receipt.get('family_coverage_complete') is not True or receipt.get('generation_reproduced') is not True
            or receipt.get('candidate_policy_used') is not False or not benchmark.family_coverage_complete):
        raise ValueError('complete candidate-independent frozen benchmark/receipt required')


def _distribution(policy,state,role):
    full=np.asarray(policy.initial_distribution(state,role,fresh_episode=True),dtype=np.float64)
    if full.shape!=(82,) or not np.isfinite(full).all() or (full<0).any() or abs(full.sum()-1)>1e-7:
        raise ValueError('invalid frozen public action marginal')
    legal=state.legal_actions(role);illegal=np.ones(82,dtype=bool);illegal[list(legal)]=False
    if full[illegal].any():raise ValueError('frozen marginal assigns probability to an illegal action')
    return full[list(legal)].tolist()


class ProbePolicyEvidence:
    """Read-only finite-benchmark evidence; no model is needed to grade rows."""
    def __init__(self,root):
        self.root=Path(root).resolve();self.database=self.root/'evidence.sqlite'
        if not self.database.is_file():raise ValueError('probe inference database is missing')
        db=_read_connect(self.database)
        try:self.contract=_meta_read(db,'contract')
        finally:db.close()
        if json.loads((self.root/'plan.json').read_text())!=self.contract:raise ValueError('probe external/database contracts differ')
        if self.contract.get('version')!=PROBE_RUN_VERSION or self.contract.get('rule_version')!=RULE_VERSION:
            raise ValueError('unsupported probe inference semantics')
        self.protocol=ProbePolicyProtocol.from_dict(self.contract['protocol']);self.benchmark=ProbeBenchmark.from_dict(self.contract['benchmark'])
        _freeze(self.benchmark,self.contract['freeze_receipt'],self.protocol)
        if self.contract['catalog'] is not None:
            catalog=ExposureCatalog.from_dict(self.contract['catalog'])
            if catalog.sha256!=self.protocol.catalog_sha256:raise ValueError('probe frozen exposure catalog differs')
        self.sha256=canonical_sha(self.contract);self.reserve=self.contract['reserve_bytes']
        self.bindings={row['seed']:row for row in self.contract['policy_bindings']}
        if len(self.contract['policy_bindings'])!=3 or tuple(sorted(self.bindings))!=CONFIRMATION_SEEDS:raise ValueError('original three frozen probe policies required')
        for seed,sha in self.protocol.policy_ids:
            if self.bindings[seed]['sha256']!=sha or canonical_sha(self.bindings[seed]['identity'])!=sha:
                raise ValueError('probe policy identity differs from the frozen registry')

    def validate_family(self,row,index):
        family=self.benchmark.families[index]
        if row.get('index')!=index or row.get('family_id')!=family.identity or row.get('plan_sha256')!=self.sha256:
            raise ValueError('probe family prefix/identity differs')
        if row.get('root_ids')!=list(family.root_ids) or set(row.get('marginals',{}))!={str(s) for s in CONFIRMATION_SEEDS}:
            raise ValueError('complete original roots/three-seed marginals required')
        for seed in CONFIRMATION_SEEDS:
            pair=row['marginals'][str(seed)]
            if set(pair)!={'before','after'}:raise ValueError('both paired probe roots required')
            for name,root in (('before',family.before),('after',family.after)):
                if not isinstance(pair[name],list) or len(pair[name])!=2:raise ValueError('both probe roles required')
                for values,actions in zip(pair[name],(root.certificate.black_actions,root.certificate.white_actions)):
                    p=np.asarray(values,dtype=np.float64)
                    if p.shape!=(len(actions),) or not np.isfinite(p).all() or (p<0).any() or abs(p.sum()-1)>1e-7:
                        raise ValueError('complete legal normalized probe marginal required')

    def records(self):
        db=_read_connect(self.database)
        try:stored=list(db.execute('SELECT idx,family_id,payload,sha256 FROM families ORDER BY idx'))
        finally:db.close()
        if len(stored)>32:raise ValueError('probe inference exceeds the fixed catalog')
        rows=[]
        for index,(actual,identity,payload,checksum) in enumerate(stored):
            if actual!=index or identity!=self.benchmark.families[index].identity or hashlib.sha256(payload.encode()).hexdigest()!=checksum:
                raise ValueError('probe scored-prefix/checksum differs')
            row=json.loads(payload);self.validate_family(row,index);rows.append(row)
        return rows

    def estimate(self):
        rows=self.records();evidence_sha=canonical_sha([canonical_sha(row) for row in rows]);complete=len(rows)==32
        grade=None
        if complete:
            by_seed={seed:{row['family_id']:row['marginals'][str(seed)] for row in rows} for seed in CONFIRMATION_SEEDS}
            grade=assess_marginals(self.benchmark,by_seed)
        reproduced=False;audit_path=self.root/'inference-audit.json'
        if audit_path.exists():
            wrapper=json.loads(audit_path.read_text());body=wrapper['body']
            if wrapper['sha256']!=canonical_sha(body) or body['contract_sha256']!=self.sha256:
                raise ValueError('probe inference audit integrity/contract differs')
            if body['completed_families']==len(rows) and body['evidence_sha256']==evidence_sha:reproduced=body['verified'] is True
        clock=self.clock();spent=clock['observed_seconds']+clock['recovery_charge_seconds']
        budget_ok=clock['inflight_since'] is None and spent<=self.protocol.time_ceiling_seconds
        metadata=bool(self.protocol.purpose=='confirmation' and all(row['archive_reference'] is not None for row in self.bindings.values()))
        numerical=bool(grade and grade['all_seed_criterion_passed'])
        return {'version':PROBE_POLICY_VERSION,'contract_sha256':self.sha256,'protocol_sha256':self.protocol.sha256,
            'evidence_sha256':evidence_sha,'purpose':self.protocol.purpose,'completed_families':len(rows),'inference_complete':complete,
            'family_coverage_complete':True,'required_families':32,'required_passes_per_seed':24,'grade':grade,
            'numerical_behavior_pass':numerical,'archive_policy_binding_verified':metadata,'frozen_inference_reproduced':reproduced,
            'confirmation_behavioral_criterion_met':bool(numerical and metadata and reproduced and budget_ok),
            'clock':clock,'time_ceiling_seconds':self.protocol.time_ceiling_seconds,'within_inference_budget':budget_ok,
            'a2_complete':False,'training_scope_independently_verified':False,'training_trajectory_independently_verified':False,
            'matched_exposure_independently_verified':False,'gray_specialization_demonstrated':False,
            'scope':'exact gaps/collision marginals on a complete fixed benchmark; no binomial population confidence claim; full A2/training/exposure/performance prerequisites remain separate'}


    def clock(self):
        db=_read_connect(self.database)
        try:clock=_meta_read(db,'clock');count=db.execute('SELECT COUNT(*) FROM families').fetchone()[0]
        finally:db.close()
        if clock.get('version')!=1 or clock.get('plan_sha256')!=self.sha256 or clock.get('committed_families')!=count:
            raise ValueError('probe clock/prefix binding differs')
        if any(type(clock.get(k)) not in (int,float) or not math.isfinite(clock[k]) or clock[k]<0 for k in ('observed_seconds','recovery_charge_seconds')):
            raise ValueError('invalid cumulative probe clock')
        if clock['inflight_since'] is not None and (type(clock['inflight_since']) not in (int,float) or not math.isfinite(clock['inflight_since']) or clock['inflight_since']<=0):
            raise ValueError('invalid probe in-flight clock')
        if type(clock.get('reproduced_families')) is not int or clock['reproduced_families']<0:raise ValueError('invalid reproduced-family clock')
        return clock

class ProbePolicyRun(ProbePolicyEvidence):
    def __init__(self,protocol,root,benchmark,freeze,policies,*,catalog=None,archive_paths=None,resume=False,
                 reserve_bytes=DEFAULT_RESERVE,prior_setup_seconds=0.):
        initialization=time.monotonic()
        if type(prior_setup_seconds) not in (int,float) or not math.isfinite(prior_setup_seconds) or prior_setup_seconds<0:
            raise ValueError('finite measured probe setup time required')
        if type(reserve_bytes) is not int or reserve_bytes<0 or (protocol.purpose=='confirmation' and reserve_bytes<DEFAULT_RESERVE):
            raise ValueError('confirmation must preserve the twenty-GiB disk reserve')
        _freeze(benchmark,freeze,protocol)
        if tuple(sorted((seed,policy.sha256) for seed,policy in policies.items()))!=protocol.policy_ids:
            raise ValueError('actual frozen probe policies differ from protocol')
        for policy in policies.values():
            policy.assert_frozen()
            if policy.identity.get('kind')!=protocol.representation or not hasattr(policy,'initial_distribution'):
                raise ValueError('probe policy representation/marginal interface differs')
        references={}
        if protocol.purpose=='confirmation':
            if not isinstance(catalog,ExposureCatalog) or catalog.sha256!=protocol.catalog_sha256:
                raise ValueError('actual frozen confirmation exposure catalog required')
            if set(archive_paths or {})!=set(CONFIRMATION_SEEDS):raise ValueError('all three actual confirmation archive paths required')
            keys=tuple((protocol.curriculum,'gray',s) for s in CONFIRMATION_SEEDS)
            refs=_archive_provenance(protocol.science,{key:policies[key[2]] for key in keys},
                {key:(archive_paths or {})[key[2]] for key in keys},catalog,expected_keys=keys)
            references={seed:refs[protocol.curriculum,'gray',seed] for seed in CONFIRMATION_SEEDS}
        elif archive_paths:
            if set(archive_paths)!=set(CONFIRMATION_SEEDS):raise ValueError('all diagnostic archive paths required')
            for seed,path in archive_paths.items():
                path=Path(path).resolve();path=path.parent if path.is_file() else path
                expected=FrozenNeuralPolicy.from_archive(path,representation=protocol.representation)
                if expected.sha256!=policies[seed].sha256:raise ValueError('diagnostic policy differs from its actual archive')
                references[seed]={'archive':str(path),'archive_manifest_sha256':digest(path/'manifest.json'),
                                  'confirmation_metadata_verified':False,'scope_approval_independently_verified':False}
        runtime=_runtime_identity(policies)
        if protocol.purpose=='confirmation' and not runtime['deterministic_algorithms']:
            raise ValueError('confirmation requires pinned deterministic inference settings')
        root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
        contract={'version':PROBE_RUN_VERSION,'rule_version':RULE_VERSION,'source_sha256':source_digest(),
            'protocol':protocol.to_dict(),'benchmark':benchmark.to_dict(),'freeze_receipt':freeze,
            'catalog':None if catalog is None else catalog.to_dict(),'runtime':runtime,'reserve_bytes':reserve_bytes,
            'policy_bindings':[{'seed':s,'sha256':policies[s].sha256,'identity':policies[s].identity,
                               'archive_reference':references.get(s)} for s in CONFIRMATION_SEEDS],
            'root_semantics':'new probe episode at each complete legal root; setup trace proves reachability; role marginals and mixture draws independent',
            'clock_scope':'cumulative measured setup, collection, validation/commit and frozen-forward reproduction; standalone reports are separate CPU overhead'}
        sha=canonical_sha(contract);database=root/'evidence.sqlite';path=root/'plan.json'
        with _EvaluationLock(root/'.probe.lock'):
            if resume:
                if not path.is_file() or not database.is_file() or json.loads(path.read_text())!=contract:
                    raise ValueError('probe source/model/benchmark/ceiling/runtime changed on resume')
            else:
                if path.exists() or database.exists():raise FileExistsError('probe inference exists; inspect original handle and resume explicitly')
                if shutil.disk_usage(root).free-4*len(json.dumps(contract).encode())-524288<reserve_bytes:
                    raise DiskBudgetError('probe plan would violate disk reserve')
                db=_connect(database)
                try:
                    with db:
                        db.execute('CREATE TABLE meta (key TEXT PRIMARY KEY,payload TEXT NOT NULL,sha256 TEXT NOT NULL)')
                        db.execute('CREATE TABLE families (idx INTEGER PRIMARY KEY,family_id TEXT UNIQUE NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL)')
                        _meta_write(db,'contract',contract)
                        _meta_write(db,'clock',{'version':1,'plan_sha256':sha,'observed_seconds':0.,'recovery_charge_seconds':0.,
                                              'inflight_since':time.time()-prior_setup_seconds-(time.monotonic()-initialization),
                                              'committed_families':0,'reproduced_families':0})
                finally:db.close()
                write_report(path,contract,reserve_bytes=reserve_bytes)
                write_report(root/'run.json',make_manifest(seed=0,command={'probe_policy':protocol.to_dict()},rules={'size':9,'max_rounds':200,'collision':'gray'}),reserve_bytes=reserve_bytes)
        super().__init__(root);self.policies=policies
        with _EvaluationLock(root/'.probe.lock'):
            clock=self.clock();setup=prior_setup_seconds+time.monotonic()-initialization
            if resume:self._recover(clock,setup)
            else:clock['inflight_since']=None
            clock['observed_seconds']+=setup;self._save_clock(clock)

    def _save_clock(self,clock):
        if shutil.disk_usage(self.root).free-65536<self.reserve:raise DiskBudgetError('probe clock write would violate reserve')
        db=_connect(self.database)
        try:
            with db:_meta_write(db,'clock',clock)
        finally:db.close()

    def _recover(self,clock,setup=0.):
        if clock['inflight_since'] is not None:
            remaining=max(0.,self.protocol.time_ceiling_seconds-clock['observed_seconds']-clock['recovery_charge_seconds']-setup)
            gap=time.time()-clock['inflight_since']-setup
            clock['recovery_charge_seconds']+=min(remaining,gap) if gap>=0 else remaining
            clock['inflight_since']=None

    def _assert_frozen(self):
        if (canonical_sha(self.contract)!=self.sha256 or self.benchmark.sha256!=self.protocol.benchmark_sha256
                or self.contract['source_sha256']!=source_digest() or _runtime_identity(self.policies)!=self.contract['runtime']):
            raise ValueError('probe inference source/numerical runtime changed')
        for policy in self.policies.values():policy.assert_frozen()

    @contextmanager
    def _session(self):
        clock=self.clock();self._recover(clock);base=clock['observed_seconds'];started=time.monotonic()
        clock['inflight_since']=time.time();self._save_clock(clock)
        def remaining():return self.protocol.time_ceiling_seconds-base-clock['recovery_charge_seconds']-(time.monotonic()-started)
        try:
            self._assert_frozen();yield clock,started,base,remaining
        finally:
            clock['observed_seconds']=base+time.monotonic()-started;clock['inflight_since']=None;self._save_clock(clock)

    def _family(self,index):
        family=self.benchmark.families[index]
        return {'index':index,'family_id':family.identity,'plan_sha256':self.sha256,'root_ids':list(family.root_ids),
            'marginals':{str(seed):{name:[_distribution(self.policies[seed],root.state,role) for role in (1,2)]
                for name,root in (('before',family.before),('after',family.after))} for seed in CONFIRMATION_SEEDS}}

    def run(self,until_families=32,*,stop_requested=lambda:False):
        if type(until_families) is not int or not 0<=until_families<=32:raise ValueError('bounded complete-family endpoint required')
        with _EvaluationLock(self.root/'.probe.lock'),self._session() as (clock,started,base,remaining):
            rows=self.records()
            if len(rows)>until_families:raise ValueError('cannot hide already committed probe families')
            status='complete' if until_families==32 else 'segment_complete'
            while len(rows)<until_families:
                if stop_requested():status='interrupted';break
                if remaining()<=0:status='budget_limited';break
                if shutil.disk_usage(self.root).free-262144<self.reserve:raise DiskBudgetError('probe unit would violate reserve')
                index=len(rows);row=self._family(index);self._assert_frozen();self.validate_family(row,index)
                payload=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)
                if shutil.disk_usage(self.root).free-len(payload.encode())-262144<self.reserve:raise DiskBudgetError('probe commit would violate reserve')
                updated=dict(clock,observed_seconds=base+time.monotonic()-started,inflight_since=time.time(),committed_families=index+1)
                db=_connect(self.database)
                try:
                    with db:
                        db.execute('INSERT INTO families VALUES (?,?,?,?)',(index,row['family_id'],payload,hashlib.sha256(payload.encode()).hexdigest()))
                        _meta_write(db,'clock',updated)
                    clock.clear();clock.update(updated)
                finally:db.close()
                rows.append(row)
            self._assert_frozen()
        result={'status':status,'committed_families':len(rows),'requested_families':until_families,'clock':self.clock(),
                'purpose':self.protocol.purpose,'contract_sha256':self.sha256,'gray_specialization_demonstrated':False}
        write_report(self.root/'progress.json',result,reserve_bytes=self.reserve);return result

    def reproduce(self,*,stop_requested=lambda:False):
        with _EvaluationLock(self.root/'.probe.lock'),self._session() as (clock,started,base,remaining):
            rows=self.records();checked=0;status='complete'
            for index,row in enumerate(rows):
                if stop_requested():status='interrupted';break
                if remaining()<=0:status='budget_limited';break
                self._assert_frozen();actual=self._family(index)
                if actual!=row:raise ValueError('stored probe marginal differs from its actual frozen-model forward pass')
                checked+=1;clock['reproduced_families']+=1
                clock['observed_seconds']=base+time.monotonic()-started;clock['inflight_since']=time.time();self._save_clock(clock)
            body={'verified':status=='complete' and checked==len(rows),'status':status,'contract_sha256':self.sha256,
                  'evidence_sha256':canonical_sha([canonical_sha(row) for row in rows]),'completed_families':len(rows),'checked_families':checked,
                  'source_sha256':source_digest(),'archive_policy_binding_verified':self.protocol.purpose=='confirmation',
                  'scope':'exact marginal replay under pinned model/source/numerical runtime; training trajectory/scope and scientific performance remain separate'}
            write_report(self.root/'inference-audit.json',{'body':body,'sha256':canonical_sha(body)},reserve_bytes=self.reserve)
        return body

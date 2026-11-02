"""Matched four-arm evaluation with atomic 48-game blocks and sealed budget state."""
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import sqlite3
import time
import numpy as np
from .confidence import canonical_sha
from .curriculum import ExposureCatalog,ExposureStarts
from .engine import BLACK,WHITE,RULE_VERSION,Rules,State
from .evaluation import EVALUATION_VERSION,FrozenNeuralPolicy,evaluate_pair,validate_pair,seed_for,_EvaluationLock
from .manifest import _command,make_manifest,source_digest,write_report
from .pilot_gates import CONFIRMATION_SEEDS
from .science_protocol import ScienceProtocol,CURRICULA,CONTEXTS,RULES,policy_keys,GAMES_PER_BLOCK
from .storage import CheckpointStore,DEFAULT_RESERVE,DiskBudgetError,digest

SCIENCE_RUN_VERSION='graygo/matched-scientific-evaluation/1'


def _connect(path):
    db=sqlite3.connect(path);db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA journal_mode=DELETE');return db


def _read_connect(path):return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)


def _meta_write(db,key,value):
    payload=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    db.execute('INSERT OR REPLACE INTO meta VALUES (?,?,?)',(key,payload,hashlib.sha256(payload.encode()).hexdigest()))


def _meta_read(db,key):
    row=db.execute('SELECT payload,sha256 FROM meta WHERE key=?',(key,)).fetchone()
    if row is None or hashlib.sha256(row[0].encode()).hexdigest()!=row[1]:raise ValueError('scientific metadata/clock checksum differs')
    return json.loads(row[0])


def _component(identity,seed):
    if identity['kind']!='independent-episode-mixture':return None
    weights=np.asarray(identity['weights'],dtype=float);components=identity['components']
    if weights.shape!=(len(components),) or not np.isfinite(weights).all() or (weights<0).any() or abs(weights.sum()-1)>1e-12:raise ValueError('invalid declared episode mixture')
    return components[int(np.random.default_rng(seed_for(seed,'component')).choice(len(components),p=weights))]


def _runtime_identity(policies):
    import torch
    def devices(policy):
        if hasattr(policy,'components'):return {device for component in policy.components for device in devices(component)}
        return {str(getattr(policy,'device','cpu'))}
    device_names=sorted({device for policy in policies.values() for device in devices(policy)})
    gpu=[]
    for name in device_names:
        if name.startswith('cuda'):
            props=torch.cuda.get_device_properties(torch.device(name))
            gpu.append({'device':name,'name':props.name,'capability':[props.major,props.minor],'total_memory':props.total_memory})
    cpu=None
    if Path('/proc/cpuinfo').exists():
        cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),None)
    driver=_command(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader']) if gpu else None
    return {'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,'gpu_driver':driver,
            'devices':device_names,'gpus':gpu,'cpu':cpu,'torch_threads':torch.get_num_threads(),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
            'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cublas_workspace':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def _archive_provenance(protocol,policies,paths,catalog,*,expected_keys=None):
    keys=policy_keys() if expected_keys is None else tuple(expected_keys)
    if not keys or not set(keys)<=set(policy_keys()) or set(paths)!=set(keys) or set(policies)!=set(keys):
        raise ValueError('all declared confirmation checkpoint paths/policies required')
    if protocol.representation!='average':raise ValueError('confirmation archive loader currently supports the selected NFSP average representation')
    rows={};training_runs=set()
    for key in keys:
        curriculum,training_rule,seed=key;policy=policies[key]
        if policy.identity['kind']!='average':raise ValueError('confirmation requires its declared deployed average policy')
        path=Path(paths[key]).resolve()
        if path.is_file():path=path.parent
        seal=CheckpointStore.verify(path)
        if path.parent.name!='weights':raise ValueError('permanent public confirmation archive required')
        manifest_path=path.parents[1]/'run.json';manifest=json.loads(manifest_path.read_text());config=manifest['config']
        expected=FrozenNeuralPolicy.from_archive(path,representation='average',device=str(policy.device),inference_batch_size=policy.inference_batch_size,cuda_graphs=policy.cuda_graphs)
        if expected.sha256!=policy.sha256:raise ValueError('provided policy differs from the verified public checkpoint')
        origin=policy.identity.get('origin') or {}
        rules={'size':9,'max_rounds':200,'collision':training_rule}
        if origin.get('seed')!=seed or origin.get('records')!=protocol.training_records or origin.get('rules')!=rules or manifest.get('rules')!=rules:
            raise ValueError('confirmation checkpoint rule/seed/milestone provenance differs')
        if config.get('scope')!='confirmation' or config.get('records')!=protocol.training_records or config.get('seed')!=seed:
            raise ValueError('development/gate agents cannot be relabeled as independent confirmation runs')
        if (seal['step']!=protocol.training_records or (config.get('width'),config.get('blocks'),config.get('norm_groups'))!=(64,8,8)
                or (config.get('batch'),config.get('learning_start'),config.get('capacity'),config.get('target_every'))!=(512,4096,500000,12800)
                or (config.get('learning_rate'),config.get('max_grad_norm'))!=(3e-4,.5)
                or (config.get('reservoir_open'),config.get('handoff_end'))!=(5000000,7000000)
                or config.get('uniform_comparison') or not config.get('augmentation') or not config.get('deterministic')):
            raise ValueError('confirmation archive does not use the fixed full-board primary protocol')
        from .board_journal import BoardJournalPlan
        plan=config.get('journal_plan')
        try:journal=BoardJournalPlan(plan['scope'],plan['records'],plan['milestones'],plan['evaluation_seed'])
        except (TypeError,KeyError) as exc:raise ValueError('complete confirmation journal plan required') from exc
        if journal.to_dict()!=plan or journal.scope!='confirmation' or journal.records!=protocol.training_records:
            raise ValueError('confirmation journal plan differs from its declared training scope')
        starts=manifest.get('training_starts')
        expected_starts={'kind':'official-empty','version':1} if curriculum=='empty' else ExposureStarts(catalog).identity
        if starts!=expected_starts:raise ValueError('confirmation training curriculum/catalog differs')
        run_sha=digest(manifest_path)
        if (seal['metadata'].get('training_run_sha256')!=run_sha or seal['metadata'].get('journal_plan_sha256')!=canonical_sha(config.get('journal_plan'))):
            raise ValueError('confirmation archive seal belongs to another run/journal plan')
        if run_sha in training_runs:raise ValueError('independent arm/seed training runs cannot be aliases')
        training_runs.add(run_sha)
        rows[key]={'archive':str(path),'archive_manifest_sha256':digest(path/'manifest.json'),'training_run_sha256':run_sha,
                   'training_scope':config['scope'],'training_records':seal['step'],'training_starts':starts,
                   'scope_approval_independently_verified':False}
        del expected
    return rows


class ScientificEvidence:
    """Read-only scored evidence; no model process or RNG is needed for analysis."""
    def __init__(self,root):
        self.root=Path(root).resolve();self.database=self.root/'evidence.sqlite'
        if not self.database.is_file():raise ValueError('scientific evidence database is missing')
        db=_read_connect(self.database)
        try:self.contract=_meta_read(db,'contract')
        finally:db.close()
        if json.loads((self.root/'plan.json').read_text())!=self.contract:raise ValueError('scientific external/database plans differ')
        if self.contract.get('version')!=SCIENCE_RUN_VERSION or self.contract.get('evaluation_version')!=EVALUATION_VERSION or self.contract.get('rule_version')!=RULE_VERSION:
            raise ValueError('unsupported scientific evidence semantics')
        if 'pair_batching' in self.contract and type(self.contract['pair_batching']) is not bool:raise ValueError('invalid scientific pair batching contract')
        self.protocol=ScienceProtocol.from_dict(self.contract['protocol']);self.catalog=ExposureCatalog.from_dict(self.contract['catalog'])
        if self.catalog.sha256!=self.protocol.catalog_sha256:raise ValueError('scientific held-out catalog differs')
        self.sha256=canonical_sha(self.contract);self.reserve=self.contract['reserve_bytes']
        self.bindings={}
        for row in self.contract['policy_bindings']:
            key=(row['curriculum'],row['training_rule'],row['seed'])
            if key in self.bindings or canonical_sha(row['identity'])!=row['sha256']:raise ValueError('scientific policy identity changed')
            self.bindings[key]=row
        if set(self.bindings)!=set(policy_keys()) or tuple(sorted((*key,row['sha256']) for key,row in self.bindings.items()))!=self.protocol.policy_ids:
            raise ValueError('scientific arm/seed registry differs')
        self.starts=ExposureStarts(self.catalog,'heldout','gray-only')

    def opening(self,index,context,rule):
        rules=Rules(collision=rule)
        if context=='official-empty':return State(rules),None
        return self.starts.sample_with_family(seed_for(self.protocol.seed,'heldout-opening',index),rules)

    def unit_seed(self,index,training_seed):return seed_for(self.protocol.seed,'scientific-unit',index,training_seed)

    def validate_block(self,row,index):
        uid=str(seed_for(self.protocol.seed,'scientific-block',index))
        if row.get('index')!=index or row.get('uid')!=uid or row.get('plan_sha256')!=self.sha256 or set(row.get('cells',{}))!=set(CONTEXTS):raise ValueError('scientific block identity/context inventory differs')
        if set(row.get('openings',{}))!=set(CONTEXTS):raise ValueError('scientific opening inventory differs')
        for context in CONTEXTS:
            if set(row['cells'][context])!=set(CURRICULA) or set(row['openings'][context])!=set(RULES):raise ValueError('scientific curriculum/rule inventory differs')
            for rule in RULES:
                root,family=self.opening(index,context,rule)
                expected={'root_sha256':hashlib.sha256(root.canonical_json().encode()).hexdigest(),'family_id':family}
                if row['openings'][context][rule]!=expected:raise ValueError('scientific sampled opening/family differs')
            for curriculum in CURRICULA:
                cells=row['cells'][context][curriculum]
                if set(cells)!={str(seed) for seed in CONFIRMATION_SEEDS}:raise ValueError('scientific training seeds were omitted/replaced')
                for seed in CONFIRMATION_SEEDS:
                    if set(cells[str(seed)])!=set(RULES):raise ValueError('scientific rule comparison is incomplete')
                    unit=self.unit_seed(index,seed);subject=self.bindings[curriculum,'gray',seed];opponent=self.bindings[curriculum,'remove',seed]
                    for rule in RULES:
                        pair=cells[str(seed)][rule];validate_pair(pair,Rules(collision=rule),False)
                        if pair['root_sha256']!=row['openings'][context][rule]['root_sha256']:raise ValueError('scientific game uses another opening')
                        for color,game in enumerate(pair['subject']):
                            mine=seed_for(unit,'subject',color);theirs=seed_for(unit,'opponent',opponent['sha256'],color)
                            if game['subject_component']!=_component(subject['identity'],mine) or game['opponent_component']!=_component(opponent['identity'],theirs):
                                raise ValueError('scientific private episode component differs from its declared draw')

    def records(self):
        db=_read_connect(self.database)
        try:rows=list(db.execute('SELECT idx,uid,payload,sha256 FROM blocks ORDER BY idx'))
        finally:db.close()
        if len(rows)>self.protocol.block_looks[-1]:raise ValueError('scientific evidence exceeds its maximum allocation')
        result=[]
        for expected,(index,uid,payload,checksum) in enumerate(rows):
            if index!=expected or uid!=str(seed_for(self.protocol.seed,'scientific-block',index)) or hashlib.sha256(payload.encode()).hexdigest()!=checksum:
                raise ValueError('scientific evidence prefix/checksum differs')
            row=json.loads(payload);self.validate_block(row,index);result.append(row)
        return result

    def estimate(self,total_games):
        rows=self.records();values=np.empty((len(rows),2,2,3,2),dtype=np.float64)
        for block,row in enumerate(rows):
            for ci,context in enumerate(CONTEXTS):
                for ai,curriculum in enumerate(CURRICULA):
                    for si,seed in enumerate(CONFIRMATION_SEEDS):
                        for ri,rule in enumerate(RULES):values[block,ci,ai,si,ri]=row['cells'][context][curriculum][str(seed)][rule]['utility']
        result=self.protocol.summarize(values,total_games);n=total_games//GAMES_PER_BLOCK
        result.update({'plan_sha256':self.sha256,'evidence_sha256':canonical_sha([canonical_sha(row) for row in rows[:n]]),
            'checkpoint_metadata_checked':self.protocol.purpose=='confirmation','training_scope_approval_verified':False,
            'matched_training_exposure_verified':False,'independent_probe_coverage_verified':False})
        return result


class ScientificRun(ScientificEvidence):
    def __init__(self,protocol,root,policies,catalog,*,archive_paths=None,resume=False,reserve_bytes=DEFAULT_RESERVE,prior_setup_seconds=0.,pair_batching=False):
        initialization_started=time.monotonic()
        if not math.isfinite(prior_setup_seconds) or prior_setup_seconds<0:raise ValueError('finite observed setup time required')
        if set(policies)!=set(policy_keys()) or catalog.sha256!=protocol.catalog_sha256:raise ValueError('complete frozen policy/catalog registry required')
        if type(reserve_bytes) is not int or reserve_bytes<0:raise ValueError('nonnegative scientific disk reserve required')
        if type(pair_batching) is not bool:raise ValueError('explicit scientific pair batching selection required')
        if protocol.purpose=='confirmation' and reserve_bytes<DEFAULT_RESERVE:raise ValueError('confirmation cannot reduce the twenty-GiB reserve')
        ids=tuple(sorted((*key,policy.sha256) for key,policy in policies.items()))
        if ids!=protocol.policy_ids:raise ValueError('scientific policy registry differs from its preregistered protocol')
        for policy in policies.values():policy.assert_frozen()
        provenance=_archive_provenance(protocol,policies,archive_paths or {},catalog) if protocol.purpose=='confirmation' else {}
        runtime=_runtime_identity(policies)
        if protocol.purpose=='confirmation' and not runtime['deterministic_algorithms']:raise ValueError('confirmation requires declared deterministic inference settings')
        root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
        contract={'version':SCIENCE_RUN_VERSION,'protocol':protocol.to_dict(),'catalog':catalog.to_dict(),
            'pair_batching':pair_batching,
            'policy_bindings':[{'curriculum':c,'training_rule':r,'seed':s,'sha256':policies[c,r,s].sha256,'identity':policies[c,r,s].identity,
                               'archive_reference':provenance.get((c,r,s))} for c,r,s in policy_keys()],
            'rule_version':RULE_VERSION,'evaluation_version':EVALUATION_VERSION,'source_sha256':source_digest(),'runtime':runtime,'reserve_bytes':reserve_bytes,
            'pairing':'agents exchange colors on the same complete root; same private seeds across evaluation rules',
            'clock':'cumulative setup/collection/in-run validation and commit wall time; standalone reporting is separate CPU overhead; unclean in-flight gaps include offline time conservatively',
            'budget_boundary':'checked between atomic matched 48-game blocks; final complete block/IO may overrun'}
        database=root/'evidence.sqlite';path=root/'plan.json';sha=canonical_sha(contract)
        with _EvaluationLock(root/'.scientific.lock'):
            if resume:
                if not database.is_file() or not path.is_file() or json.loads(path.read_text())!=contract:raise ValueError('scientific source/model/catalog/protocol/runtime changed on resume')
            else:
                if database.exists() or path.exists():raise FileExistsError('scientific run exists; use explicit resume with its original handle')
                if shutil.disk_usage(root).free-524288<reserve_bytes:raise DiskBudgetError('scientific plan would violate disk reserve')
                db=_connect(database)
                try:
                    with db:
                        db.execute('CREATE TABLE meta (key TEXT PRIMARY KEY,payload TEXT NOT NULL,sha256 TEXT NOT NULL)')
                        db.execute('CREATE TABLE blocks (idx INTEGER PRIMARY KEY,uid TEXT UNIQUE NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL)')
                        _meta_write(db,'contract',contract)
                        _meta_write(db,'clock',{'version':1,'plan_sha256':sha,'observed_seconds':0.,'recovery_charge_seconds':0.,'inflight_since':None,'committed_blocks':0})
                finally:db.close()
                write_report(path,contract,reserve_bytes=reserve_bytes)
                write_report(root/'run.json',make_manifest(seed=protocol.seed,command={'scientific_protocol':protocol.to_dict()},rules={'size':9,'max_rounds':200,'collision':'gray/remove'}),reserve_bytes=reserve_bytes)
        super().__init__(root);self.policies=policies;self.pair_batching=pair_batching
        with _EvaluationLock(self.root/'.scientific.lock'):
            state=self.clock();setup=prior_setup_seconds+time.monotonic()-initialization_started
            if state['inflight_since'] is not None:
                gap=max(0.,time.time()-state['inflight_since']-setup)
                remaining=max(0.,protocol.time_ceiling_seconds-state['observed_seconds']-state['recovery_charge_seconds'])
                state['recovery_charge_seconds']+=min(remaining,gap);state['inflight_since']=None
            state['observed_seconds']+=setup;self._clock_save(state)

    def clock(self):
        db=_read_connect(self.database)
        try:state=_meta_read(db,'clock');count=db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0]
        finally:db.close()
        if state.get('version')!=1 or state.get('plan_sha256')!=self.sha256 or state.get('committed_blocks')!=count:raise ValueError('scientific clock/prefix binding differs')
        if any(not isinstance(state.get(k),(int,float)) or not math.isfinite(state[k]) or state[k]<0 for k in ('observed_seconds','recovery_charge_seconds')):raise ValueError('invalid scientific cumulative clock')
        if state['inflight_since'] is not None and (not isinstance(state['inflight_since'],(int,float)) or not math.isfinite(state['inflight_since']) or state['inflight_since']<=0):raise ValueError('invalid scientific in-flight clock')
        return state

    def _clock_save(self,state):
        if shutil.disk_usage(self.root).free-65536<self.reserve:raise DiskBudgetError('scientific clock write would violate disk reserve; last committed prefix retained')
        db=_connect(self.database)
        try:
            with db:_meta_write(db,'clock',state)
        finally:db.close()

    def _assert_frozen(self):
        if self.contract['source_sha256']!=source_digest():raise ValueError('scientific evaluator source changed')
        for policy in self.policies.values():policy.assert_frozen()

    def _block(self,index):
        row={'index':index,'uid':str(seed_for(self.protocol.seed,'scientific-block',index)),'plan_sha256':self.sha256,'openings':{},'cells':{}}
        requests=[];locations=[]
        for context in CONTEXTS:
            openings={rule:self.opening(index,context,rule) for rule in RULES}
            row['openings'][context]={rule:{'root_sha256':hashlib.sha256(root.canonical_json().encode()).hexdigest(),'family_id':family} for rule,(root,family) in openings.items()}
            row['cells'][context]={}
            for curriculum in CURRICULA:
                row['cells'][context][curriculum]={}
                for seed in CONFIRMATION_SEEDS:
                    subject=self.policies[curriculum,'gray',seed];opponent=self.policies[curriculum,'remove',seed];unit=self.unit_seed(index,seed)
                    row['cells'][context][curriculum][str(seed)]={}
                    for rule in RULES:
                        if self.pair_batching:
                            from .evaluation_batch import PairRequest
                            requests.append(PairRequest(subject,opponent,openings[rule][0],unit));locations.append((context,curriculum,str(seed),rule))
                        else:row['cells'][context][curriculum][str(seed)][rule]=evaluate_pair(subject,opponent,openings[rule][0],unit_seed=unit)
        if self.pair_batching:
            from .evaluation_batch import evaluate_pairs_batched
            for (context,curriculum,seed,rule),result in zip(locations,evaluate_pairs_batched(requests)):
                row['cells'][context][curriculum][seed][rule]=result
        return row

    def run(self,total_games,*,stop_requested=lambda:False):
        if type(total_games) is not int or total_games not in self.protocol.physical_looks:raise ValueError('predeclared total scientific game look required')
        target=total_games//GAMES_PER_BLOCK
        with _EvaluationLock(self.root/'.scientific.lock'):
            clock=self.clock();cap=self.protocol.time_ceiling_seconds
            if clock['inflight_since'] is not None:
                gap=max(0.,time.time()-clock['inflight_since']);remaining=max(0.,cap-clock['observed_seconds']-clock['recovery_charge_seconds'])
                clock['recovery_charge_seconds']+=min(remaining,gap);clock['inflight_since']=None;self._clock_save(clock)
            base=clock['observed_seconds'];started=time.monotonic();status='complete'
            clock['inflight_since']=time.time();self._clock_save(clock)
            try:
                self._assert_frozen();rows=self.records()
                completed=[n for n in self.protocol.block_looks if n<=len(rows)]
                if completed and target<max(completed):raise ValueError('a later completed scientific look cannot be hidden')
                while len(rows)<target:
                    if stop_requested():status='interrupted';break
                    if base+time.monotonic()-started+clock['recovery_charge_seconds']>=cap:status='budget_limited';break
                    if shutil.disk_usage(self.root).free-262144<self.reserve:raise DiskBudgetError('scientific block would violate disk reserve')
                    index=len(rows);row=self._block(index);self._assert_frozen();self.validate_block(row,index)
                    payload=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)
                    if shutil.disk_usage(self.root).free-len(payload.encode())-262144<self.reserve:raise DiskBudgetError('scientific evidence commit would violate disk reserve')
                    next_clock=dict(clock,observed_seconds=base+time.monotonic()-started,inflight_since=time.time(),committed_blocks=index+1)
                    db=_connect(self.database)
                    try:
                        with db:
                            db.execute('INSERT INTO blocks VALUES (?,?,?,?)',(index,row['uid'],payload,hashlib.sha256(payload.encode()).hexdigest()))
                            _meta_write(db,'clock',next_clock)
                        clock=next_clock
                    finally:db.close()
                    rows.append(row)
                self._assert_frozen()
            finally:
                clock['observed_seconds']=base+time.monotonic()-started;clock['inflight_since']=None;self._clock_save(clock)
            result={'status':status,'committed_blocks':clock['committed_blocks'],'physical_games':clock['committed_blocks']*GAMES_PER_BLOCK,
                'requested_total_games':total_games,'plan_sha256':self.sha256,'clock':clock,'purpose':self.protocol.purpose,
                'gray_specialization_demonstrated':False,'scope':'fixed scored scientific contrast; training scope/exposure and independent probe requirements are separate'}
            write_report(self.root/'progress.json',result,reserve_bytes=self.reserve);return result

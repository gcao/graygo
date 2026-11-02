"""Preregister three original board runs and verify real milestone evidence.

The numerical cohort gate is conditional on sealed archives and scored rows.
It does not issue A2, resource, training-trajectory or formal-scope approval.
"""
from contextlib import ExitStack,contextmanager
from copy import deepcopy
from dataclasses import fields
import fcntl
from importlib import metadata
import json
from pathlib import Path
import sys
from .board_journal import BoardJournalPlan,JOURNAL_VERSION
from .board_runtime import BoardRuntimeConfig
from .confidence import canonical_sha
from .engine import Rules
from .evaluation import BaselinePolicy,EmptyStarts,EvaluationPlan,FrozenNeuralPolicy,evaluation_contract,seed_for,neural_inference_identity
from .evaluation_evidence import FrozenEvaluationEvidence,evaluation_read_lock
from .manifest import source_digest,write_report
from .nfsp import NFSPCadence
from .pilot_gates import DEVELOPMENT_SEEDS,PRE,POST,OPPONENTS,PilotProtocol,ScreeningProtocol,endpoint,subject_label
from .storage import CheckpointStore,digest

COHORT_VERSION='graygo/development-cohort-evidence/1'


def _run(root,seed):
    root=Path(root).resolve();manifest=json.loads((root/'run.json').read_text());config=manifest['config']
    if manifest['source_sha256']!=source_digest() or config['scope']!='development' or config['records']!=20000000 or config['seed']!=seed:
        raise ValueError('original development seed/source/horizon required')
    versions={}
    for name in ('numpy','scipy','torch'):
        try:versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:versions[name]=None
    if manifest['dependencies']!=versions or manifest['python']!=sys.version:
        raise ValueError('use the original development numerical dependencies/Python runtime')
    runtime_fields={f.name for f in fields(BoardRuntimeConfig)}
    if set(config)!=runtime_fields|{'scope','records','journal_plan'}:raise ValueError('development runtime declaration differs')
    data={key:config[key] for key in runtime_fields};data['rules']=Rules(**data['rules']);runtime=BoardRuntimeConfig(**data)
    if (runtime.rules!=Rules() or manifest['rules']!=config['rules'] or manifest['training_starts']!=EmptyStarts().identity
            or (runtime.width,runtime.blocks,runtime.norm_groups)!=(64,8,8)
            or (runtime.batch,runtime.learning_start,runtime.capacity,runtime.target_every)!=(512,4096,500000,12800)
            or (runtime.learning_rate,runtime.max_grad_norm)!=(3e-4,.5)
            or (runtime.reservoir_open,runtime.handoff_end)!=(5000000,7000000)
            or not runtime.augmentation or not runtime.deterministic or runtime.uniform_comparison):
        raise ValueError('formal primary development configuration/start distribution differs')
    plan=dict(config['journal_plan'])
    if plan.pop('version')!=JOURNAL_VERSION:raise ValueError('unknown development journal version')
    journal=BoardJournalPlan(**plan)
    if journal.scope!='development' or 0 not in journal.milestones or journal.to_dict()!=config['journal_plan']:
        raise ValueError('initialized formal development archive plan required')
    return root,manifest,runtime,journal


def _archive(root,manifest,runtime,journal,record):
    path=root/'weights'/f'{record:012d}';seal=CheckpointStore.verify(path);meta=seal['metadata']
    if (seal['step']!=record or set(seal['files'])!={'models.pt'} or meta['source_sha256']!=manifest['source_sha256']
            or meta['training_run_sha256']!=digest(root/'run.json') or meta['journal_plan_sha256']!=journal.sha256):
        raise ValueError('milestone archive source/origin/clock differs')
    counters=NFSPCadence(**meta['counters']);counters.validate()
    expected=max(0,record//runtime.batch-runtime.learning_start//runtime.batch+1)*runtime.q_reuse
    if (counters.fresh_records!=record or counters.fresh_q!=record or counters.fresh_ordinary!=record or counters.q_updates!=expected
            or counters.q_loss_targets!=expected*runtime.batch or counters.target_refreshes!=record//runtime.target_every
            or counters.supervised_targets!=counters.supervised_updates*runtime.batch
            or counters.supervised_updates>record//runtime.batch
            or (record<=runtime.reservoir_open and counters.supervised_updates)):
        raise ValueError('milestone fresh/update/refresh/supervised accounting differs')
    initial=root/'weights'/'000000000000'
    if record and meta['native_schema']!=CheckpointStore.verify(initial)['metadata']['native_schema']:
        raise ValueError('native observation decoder changed between milestones')
    return path,digest(path/'manifest.json')


def milestone_plan(protocol,journal,seed,record,representation,opponent,*,label='cohort-milestone'):
    if seed not in DEVELOPMENT_SEEDS or opponent not in OPPONENTS:raise ValueError('fixed development seed/opponent required')
    if isinstance(protocol,PilotProtocol):
        valid=(record in PRE and representation=='greedy_q') or (record in POST and representation in ('greedy_q','average'))
        phase='pilot-evaluation'
    elif isinstance(protocol,ScreeningProtocol):
        valid=record==20000000 and representation=='average';phase='screening-evaluation'
    else:raise ValueError('fixed pilot or screening protocol required')
    if not valid:raise ValueError('unregistered cohort milestone/representation')
    owner=protocol if isinstance(protocol,PilotProtocol) else protocol.pilot
    return EvaluationPlan(label,subject_label(seed,record,representation),endpoint(seed,record,representation,opponent),
                          Rules(),seed_for(journal.evaluation_seed,phase,seed,record,representation,opponent),
                          protocol.family.looks,protocol.family.sha256,pair_batch_size=owner.pair_batch_size)


def _initial_weight_hashes(runtime):
    import torch
    from .models import QNet,PolicyNet
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):
        generator=torch.Generator(device='cpu').manual_seed(runtime.seed)
        torch.set_rng_state(generator.get_state())
        q=QNet(runtime.model_config());average=PolicyNet(runtime.model_config())
        return (FrozenNeuralPolicy(q,representation='greedy_q').identity['weights_sha256'],
                FrozenNeuralPolicy(average,representation='average').identity['weights_sha256'])


def read_milestone(training_root,protocol,seed,record,representation,opponent,directory,*,cache=None):
    """Read one original archive/score slot. Caller holds its shared input lock."""
    root,manifest,runtime,journal=_run(training_root,seed)
    registered=dict(protocol.training_runs if isinstance(protocol,PilotProtocol) else protocol.pilot.training_runs)
    if registered[seed]!=digest(root/'run.json'):raise ValueError('milestone is not the registered original run')
    archive,archive_sha=_archive(root,manifest,runtime,journal,record)
    if record not in journal.milestones:raise ValueError('undeclared current model cannot replace a milestone')
    evidence=FrozenEvaluationEvidence.load(directory)
    plan=milestone_plan(protocol,journal,seed,record,representation,opponent,label=evidence.plan.label)
    key=(str(root),record,representation,archive_sha)
    if cache is not None and key in cache:subject=cache[key]
    else:
        subject=FrozenNeuralPolicy.from_archive(archive,representation=representation,label=plan.subject_label)
        if cache is not None:cache[key]=subject
    baseline=BaselinePolicy(opponent)
    expected=evaluation_contract(plan,subject,baseline,EmptyStarts())
    owner=protocol if isinstance(protocol,PilotProtocol) else protocol.pilot
    if owner.inference_batch_size>1:
        # Verify the declared device's identity from archived weights on CPU;
        # this read-only score reader performs no GPU inference or sampling.
        expected['subject']=deepcopy(expected['subject'])
        expected['subject']['inference']=neural_inference_identity(owner.inference_batch_size,owner.inference_device)
        expected['subject_sha256']=canonical_sha(expected['subject'])
    if owner.cuda_graphs:
        from .frozen_forward import execution_identity
        expected['subject']=deepcopy(expected['subject']);expected['subject']['frozen_execution']=execution_identity(True)
        expected['subject_sha256']=canonical_sha(expected['subject'])
    if evidence.contract!=expected:raise ValueError('scored milestone uses another policy, plan, baseline, source or opening')
    baseline_sha=owner.random_sha256 if opponent=='random' else owner.tactical_sha256
    if baseline.sha256!=baseline_sha:raise ValueError('cohort baseline source/version differs')
    look=evidence.largest_completed_look
    receipt=None if look is None else evidence.estimate(protocol.family,look,check_current=False)
    return receipt,{'directory':str(Path(directory).resolve()),'archive':str(archive),'archive_manifest_sha256':archive_sha,
                    'snapshot_sha256':evidence.sha256,'available_games':evidence.available_games,'completed_look':look}


class DevelopmentCohort:
    def __init__(self,root,body):
        self.root=Path(root).resolve();self.body=body;self.sha256=canonical_sha(body)
        self.pilot=PilotProtocol.from_dict(body['pilot']);self.screening=ScreeningProtocol.from_dict(body['screening'])

    @classmethod
    def register(cls,root,name,training_roots,*,inference_batch_size=1,pair_batch_size=1,inference_device='cpu',cuda_graphs=False):
        if not isinstance(name,str) or not name or set(training_roots)!=set(DEVELOPMENT_SEEDS):
            raise ValueError('named original three-seed cohort required')
        root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
        with (root/'.registry.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (root/'cohort.json').exists():raise FileExistsError('cohort registry is immutable; use its original reader')
            entries={};common=None;run_ids=set();q_ids=set();average_ids=set();paths=set()
            for seed in DEVELOPMENT_SEEDS:
                directory,manifest,runtime,journal=_run(training_roots[seed],seed)
                if directory in paths:raise ValueError('cohort runs cannot alias the same root')
                paths.add(directory)
                config=dict(manifest['config']);config.pop('seed')
                if common is None:common=config
                elif common!=config:raise ValueError('cohort seeds must keep the same implementation/settings/clock')
                archives=sorted(p.name for p in (directory/'weights').glob('[0-9]*'))
                resumes=sorted(p.name for p in (directory/'resume').glob('[0-9]*'))
                if archives!=['000000000000'] or not resumes or int(resumes[-1])!=0 or (directory/'opening-decision.json').exists():
                    raise ValueError('register the complete cohort at initialization, before training or seed selection')
                CheckpointStore.verify(directory/'resume'/resumes[-1])
                archive,initial_sha=_archive(directory,manifest,runtime,journal,0)
                q=FrozenNeuralPolicy.from_archive(archive,representation='greedy_q')
                average=FrozenNeuralPolicy.from_archive(archive,representation='average')
                if (q.identity['weights_sha256'],average.identity['weights_sha256'])!=_initial_weight_hashes(runtime):
                    raise ValueError('record-zero models differ from their declared fresh seeded initialization')
                run_sha=digest(directory/'run.json')
                if run_sha in run_ids or q.identity['weights_sha256'] in q_ids or average.identity['weights_sha256'] in average_ids:
                    raise ValueError('independent seeded initializations and original run fingerprints required')
                run_ids.add(run_sha);q_ids.add(q.identity['weights_sha256']);average_ids.add(average.identity['weights_sha256'])
                entries[str(seed)]={'root':str(directory),'training_run_sha256':run_sha,'initial_archive_manifest_sha256':initial_sha,
                                    'initial_q_weights_sha256':q.identity['weights_sha256'],'initial_average_weights_sha256':average.identity['weights_sha256']}
            runs={int(seed):row['training_run_sha256'] for seed,row in entries.items()}
            random=BaselinePolicy('random');tactical=BaselinePolicy('tactical')
            execution=dict(inference_batch_size=inference_batch_size,pair_batch_size=pair_batch_size,inference_device=inference_device,cuda_graphs=cuda_graphs)
            pilot=PilotProtocol(name+'/pilot',random.sha256,tactical.sha256,runs,**execution)
            screening=ScreeningProtocol(name+'/screening',random.sha256,tactical.sha256,runs,**execution)
            body={'version':COHORT_VERSION,'name':name,'source_sha256':source_digest(),'training_runs':entries,
                  'shared_config_without_seed':common,'pilot':pilot.to_dict(),'screening':screening.to_dict(),
                  'authority':'registration and numerical evidence only; no A2/resource/training-scope grant'}
            write_report(root/'cohort.json',{'body':body,'sha256':canonical_sha(body)})
        return cls.load(root)

    @classmethod
    def load(cls,root):
        root=Path(root).resolve();wrapper=json.loads((root/'cohort.json').read_text());body=wrapper['body']
        if wrapper['sha256']!=canonical_sha(body) or body['version']!=COHORT_VERSION or body['source_sha256']!=source_digest():
            raise ValueError('cohort registry integrity/source differs; use its frozen source')
        cohort=cls(root,body)
        if set(body['training_runs'])!={str(s) for s in DEVELOPMENT_SEEDS}:raise ValueError('cohort seed set changed')
        expected={int(seed):entry['training_run_sha256'] for seed,entry in body['training_runs'].items()}
        if (dict(cohort.pilot.training_runs)!=expected or dict(cohort.screening.pilot.training_runs)!=expected
                or cohort.pilot.random_sha256!=BaselinePolicy('random').sha256
                or cohort.pilot.tactical_sha256!=BaselinePolicy('tactical').sha256
                or cohort.screening.pilot.random_sha256!=cohort.pilot.random_sha256
                or cohort.screening.pilot.tactical_sha256!=cohort.pilot.tactical_sha256
                or cohort.screening.pilot.execution_settings()!=cohort.pilot.execution_settings()
                or cohort.pilot.name!=body['name']+'/pilot' or cohort.screening.family.name!=body['name']+'/screening'
                or not cohort.pilot.include_average_diagnostics):
            raise ValueError('original cohort runs/baselines/mandatory average diagnostics differ')
        q_ids=set();average_ids=set()
        for seed in DEVELOPMENT_SEEDS:
            entry=body['training_runs'][str(seed)];directory,manifest,runtime,journal=_run(entry['root'],seed)
            config=dict(manifest['config']);config.pop('seed')
            if config!=body['shared_config_without_seed'] or digest(directory/'run.json')!=entry['training_run_sha256']:
                raise ValueError('registered initialized cohort run changed')
            archive,checksum=_archive(directory,manifest,runtime,journal,0)
            if checksum!=entry['initial_archive_manifest_sha256']:raise ValueError('original initialization archive changed')
            q=FrozenNeuralPolicy.from_archive(archive,representation='greedy_q').identity['weights_sha256']
            average=FrozenNeuralPolicy.from_archive(archive,representation='average').identity['weights_sha256']
            if ((q,average)!=_initial_weight_hashes(runtime) or q!=entry['initial_q_weights_sha256']
                    or average!=entry['initial_average_weights_sha256'] or q in q_ids or average in average_ids):
                raise ValueError('original independent seeded initialization differs')
            q_ids.add(q);average_ids.add(average)
        return cohort

    def location(self,seed,record,representation,opponent):
        return self.root/'evaluations'/str(seed)/f'{record:012d}'/representation/opponent

    def plan_for(self,seed,record,representation,opponent):
        if seed not in DEVELOPMENT_SEEDS:raise ValueError('fixed development seed required')
        directory,manifest,runtime,journal=_run(self.body['training_runs'][str(seed)]['root'],seed)
        protocol=self.screening if record==20000000 else self.pilot
        plan=milestone_plan(protocol,journal,seed,record,representation,opponent,label=self.body['name']+'/'+endpoint(seed,record,representation,opponent))
        return plan,protocol.family,directory/'weights'/f'{record:012d}',self.location(seed,record,representation,opponent)

    @contextmanager
    def snapshot(self,stage):
        if stage not in ('pilot','screening'):raise ValueError('fixed cohort stage required')
        current=self.load(self.root)
        if current.sha256!=self.sha256:raise ValueError('registered cohort changed after loading')
        slots=[('pilot',s,n,r,o) for s in DEVELOPMENT_SEEDS for n in (*PRE,*POST)
               for r in (('greedy_q',) if n in PRE else ('greedy_q','average')) for o in OPPONENTS]
        if stage=='screening':slots += [('screening',s,20000000,'average',o) for s in DEVELOPMENT_SEEDS for o in OPPONENTS]
        reads={'pilot':[],'screening':[]};missing=[];bindings=[];cache={}
        with ExitStack() as locks:
            for phase,seed,record,representation,opponent in slots:
                path=self.location(seed,record,representation,opponent)
                if path.exists():locks.enter_context(evaluation_read_lock(path))
            for phase,seed,record,representation,opponent in slots:
                path=self.location(seed,record,representation,opponent);key=endpoint(seed,record,representation,opponent)
                if not path.exists():missing.append({'endpoint':key,'reason':'missing scored evaluation'});continue
                protocol=self.pilot if phase=='pilot' else self.screening
                receipt,binding=read_milestone(self.body['training_runs'][str(seed)]['root'],protocol,seed,record,representation,opponent,path,cache=cache)
                bindings.append({'endpoint':key,**binding})
                if receipt is None:missing.append({'endpoint':key,'reason':'no completed registered look','available_games':binding['available_games']})
                else:reads[phase].append(receipt)
            pilot=self.pilot.cohort(reads['pilot']);screen=None if stage=='pilot' else self.screening.decide(reads['screening'])
            status=pilot['status'] if screen is None else ('pass' if pilot['status']==screen['status']=='pass' else
                    'no_go' if 'no_go' in (pilot['status'],screen['status']) else 'incomplete')
            body={'version':COHORT_VERSION,'stage':stage,'status':status,'registry_sha256':self.sha256,'source_sha256':source_digest(),
                  'pilot':pilot,'screening':screen,'reads':reads,'missing':missing,'bindings':bindings,
                  'a2_complete':False,'formal_scope_approved':False,'training_trajectory_independently_verified':False,
                  'evaluation_trajectories_independently_replayed':False,'numerical_runtime_independently_verified':False,
                  'scope':'real archived policy/score bindings and numerical three-seed gates; independent training, A2/resource authority remain separate',
                  'input_locks':'all present evaluation inputs shared-locked across the complete snapshot; no input artifacts written'}
            if self.load(self.root).sha256!=self.sha256:raise ValueError('cohort registry changed while reading evidence')
            yield {'body':body,'sha256':canonical_sha(body)}

    def report(self,stage):
        with self.snapshot(stage) as report:return deepcopy(report)

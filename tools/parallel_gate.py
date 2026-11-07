"""Parallel independent color pairs; the original gate makes every decision."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import hashlib
import json
import math
import shutil
from multiprocessing import get_context
from pathlib import Path
import time

from graygo.confidence import canonical_sha
from graygo.engine import Rules
from graygo.evaluation import EvaluationPlan,EvaluationRun,EmptyStarts,evaluate_pair,seed_for,validate_pair
from graygo.manifest import write_report
from graygo.storage import DEFAULT_RESERVE,DiskBudgetError
from tools.champion_gate import IndependentGate,gate_root

_WORKER=None


def load_policy(descriptor):
    from graygo.evaluation import FrozenNeuralPolicy
    from tools.legacy_policy import LegacyPolicy
    from tools.policy_selection import apply_selection
    if descriptor['kind']=='archive':
        base=FrozenNeuralPolicy.from_archive(descriptor['archive'],label=descriptor['label'])
    elif descriptor['kind']=='legacy':
        from graygo.storage import digest
        if digest(Path(descriptor['checkpoint']))!=descriptor['checkpoint_sha256']:raise ValueError('legacy payload changed')
        base=LegacyPolicy(descriptor['repo'],descriptor['checkpoint'],label=descriptor['label'])
    else:raise ValueError('unsupported gate policy descriptor')
    result=apply_selection(base,descriptor.get('selection'),descriptor.get('bundle'))
    if result.sha256!=descriptor['policy_sha256']:raise ValueError('gate policy identity differs')
    return result


def _initialize_worker(candidate,champion,pool,rules):
    import torch
    global _WORKER
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    _WORKER=(load_policy(candidate),load_policy(champion),{key:load_policy(value) for key,value in pool.items()},Rules(**rules))


def _unit(job):
    label,index,unit=job;candidate,champion,pool,rules=_WORKER
    opening=EmptyStarts().sample(seed_for(unit,'opening'),rules)
    result=evaluate_pair(candidate,champion if label=='head' else pool[label],opening,unit_seed=unit,
                         incumbent=None if label=='head' else champion)
    return label,index,{'index':index,'uid':str(unit),**result}


class ParallelGate(IndependentGate):
    def __init__(self,plan,*,workers=8,reserve_bytes=DEFAULT_RESERVE,protected_authorizer=None):
        if type(workers) is not int or not 1<=workers<=8:raise ValueError('one to eight gate workers required')
        super().__init__(plan,reserve_bytes=reserve_bytes,protected_authorizer=protected_authorizer)
        self.workers=workers
        self.identity['parallel_execution']={'workers':workers,'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'unit':'independent color pairs; ordered committed prefixes; CPU batch-1 policy inference'}
        self.sha256=canonical_sha(self.identity)

    def evaluate_descriptors(self,candidate,champion,pool,root,*,round_index,time_cap_seconds=3600.,stop_requested=lambda:False):
        import numpy as np
        import torch
        if not math.isfinite(time_cap_seconds) or time_cap_seconds<=0:raise ValueError('positive gate session duration required')
        deadline=time.monotonic()+time_cap_seconds
        torch.set_num_threads(1)
        subject=load_policy(candidate);incumbent=load_policy(champion);opponents={key:load_policy(value) for key,value in pool.items()}
        # Create standard immutable plans/databases without collecting a game.
        initial=super().evaluate(subject,incumbent,opponents,root,round_index=round_index,time_cap_seconds=1e-12)
        if initial['status']=='complete':return initial
        actual=gate_root(root,self.plan);contract=json.loads((actual/'gate-plan.json').read_text());rules=Rules(**contract['rules']);family=self.plan.family(round_index,tuple(pool))
        runs={}
        for label,opponent in [('head',incumbent),*opponents.items()]:
            raw=json.loads((actual/label/'plan.json').read_text())['plan'];raw['rules']=Rules(**raw['rules'])
            plan=EvaluationPlan(**raw)
            runs[label]=EvaluationRun(plan,actual/label,subject,opponent,openings=EmptyStarts(),
                incumbent=None if label=='head' else incumbent,resume=True,reserve_bytes=self.reserve)
        available=max(len(run.records())*2 for run in runs.values())
        for games in self.plan.looks:
            if games<available:continue
            if stop_requested():return {'status':'interrupted','numeric_pass':False,'promotion_permitted':False}
            rows={key:run.records() for key,run in runs.items()};next_index={key:len(value) for key,value in rows.items()}
            jobs=[]
            for index in range(games//2):
                for label,run in runs.items():
                    if index>=next_index[label]:jobs.append((label,index,seed_for(run.plan.seed,'unit',index)))
            pending={label:{} for label in runs}
            with ProcessPoolExecutor(max_workers=self.workers,mp_context=get_context('spawn'),initializer=_initialize_worker,
                initargs=(candidate,champion,pool,asdict(rules))) as executor:
                futures=[executor.submit(_unit,job) for job in jobs]
                for future in as_completed(futures):
                    if stop_requested() or time.monotonic()>=deadline:
                        for waiting in futures:waiting.cancel()
                        return {'status':'interrupted' if stop_requested() else 'budget_limited','numeric_pass':False,'promotion_permitted':False}
                    try:label,index,row=future.result()
                    except BaseException:
                        for waiting in futures:waiting.cancel()
                        raise
                    validate_pair(row,rules,label!='head');pending[label][index]=row
                    run=runs[label]
                    while next_index[label] in pending[label]:
                        current=pending[label].pop(next_index[label]);payload=json.dumps(current,sort_keys=True,separators=(',',':'),allow_nan=False)
                        if shutil.disk_usage(actual).free-len(payload.encode())-65536<self.reserve:
                            raise DiskBudgetError('parallel gate commit would violate disk reserve')
                        with run._lock():
                            db=run._connect()
                            try:
                                with db:db.execute('INSERT INTO units VALUES (?,?,?,?)',(current['index'],current['uid'],payload,hashlib.sha256(payload.encode()).hexdigest()))
                            finally:db.close()
                        rows[label].append(current);next_index[label]+=1
                    write_report(actual/'parallel-progress.json',{'look':games,'games_per_pairing':{key:value*2 for key,value in next_index.items()},'promotion_permitted':False},reserve_bytes=self.reserve)
            decision=self.plan.numeric(round_index,
                [sum(float(np.sign(game['subject_utility'])) for game in row['subject'])/2 for row in rows['head']],
                {key:[row['utility'] for row in rows[key]] for key in pool},
                {key:[row['incumbent_utility'] for row in rows[key]] for key in pool},games)
            if decision['numeric_pass'] or decision['stop_for_futility'] or games==self.plan.looks[-1]:
                # All data are now present. The unchanged serial gate validates
                # evidence and produces the authoritative decision without play.
                before={key:len(value) for key,value in rows.items()}
                result=super().evaluate(subject,incumbent,opponents,root,round_index=round_index,time_cap_seconds=600)
                if before!={key:len(run.records()) for key,run in runs.items()}:raise ValueError('gate unexpectedly collected extra samples')
                return result
        raise RuntimeError('no registered gate look completed')

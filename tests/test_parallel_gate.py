import tempfile
from pathlib import Path
import unittest
import torch
from graygo.board_execution import BoardExecutor,BoardExecutionPlan
from graygo.board_journal import BoardJournalPlan
from graygo.board_runtime import BoardRuntimeConfig
from graygo.engine import Rules
from graygo.evaluation import FrozenNeuralPolicy
from tools.champion_gate import GatePlan,IndependentGate,gate_root
from tools.parallel_gate import ParallelGate,load_policy
from tools.policy_selection import SelectedAveragePolicy,selection_spec


class ParallelGateTests(unittest.TestCase):
    def test_parallel_units_reproduce_the_serial_evidence_and_decision(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config=BoardRuntimeConfig(rules=Rules(3,5),envs=2,capacity=64,batch=4,learning_start=4,target_every=8,width=8,blocks=1,norm_groups=2)
            with BoardExecutor(root/'model',config,BoardJournalPlan('diagnostic',8,(0,8)),BoardExecutionPlan(60),reserve_bytes=0):pass
            archive=root/'model/weights/000000000000';base=FrozenNeuralPolicy.from_archive(archive,label='reference')
            reference={'kind':'archive','archive':str(archive),'label':'reference','rules':{'size':3,'max_rounds':5,'collision':'gray'},
                'records':0,'policy_sha256':base.sha256,'weights_sha256':base.identity['weights_sha256']}
            selection=selection_spec();selected=SelectedAveragePolicy(base,selection)
            candidate=dict(reference,label='reference',selection=selection,policy_sha256=selected.sha256)
            plan=GatePlan(1,looks=(4,))
            serial=IndependentGate(plan,reserve_bytes=0).evaluate(load_policy(candidate),base,{'historical-0':base},root/'serial',round_index=0,time_cap_seconds=30)
            parallel=ParallelGate(plan,workers=2,reserve_bytes=0).evaluate_descriptors(candidate,reference,{'historical-0':reference},root/'parallel',round_index=0)
            self.assertEqual(serial,parallel)
            checks=[]
            def stop_after_a_committed_pair():
                checks.append(True)
                return len(checks)>=3
            runner=ParallelGate(plan,workers=2,reserve_bytes=0)
            interrupted=runner.evaluate_descriptors(candidate,reference,{'historical-0':reference},root/'resumed',
                round_index=0,stop_requested=stop_after_a_committed_pair)
            self.assertEqual(interrupted['status'],'interrupted')
            resumed=runner.evaluate_descriptors(candidate,reference,{'historical-0':reference},root/'resumed',round_index=0)
            self.assertEqual(serial,resumed)
            for label in ('head','historical-0'):
                import sqlite3
                values=[]
                for mode in ('serial','parallel','resumed'):
                    db=sqlite3.connect(gate_root(root/mode,plan)/label/'evidence.sqlite')
                    values.append(db.execute('SELECT idx,uid,payload,sha256 FROM units ORDER BY idx').fetchall());db.close()
                self.assertEqual(values[0],values[1]);self.assertEqual(values[0],values[2])
    def test_q_head_parallel_gate_reproduces_serial_statistics(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config=BoardRuntimeConfig(rules=Rules(3,5),envs=2,capacity=64,batch=4,learning_start=4,target_every=8,width=8,blocks=1,norm_groups=2)
            with BoardExecutor(root/'model',config,BoardJournalPlan('diagnostic',8,(0,8)),BoardExecutionPlan(60),reserve_bytes=0):pass
            archive=root/'model/weights/000000000000';base=FrozenNeuralPolicy.from_archive(archive,label='reference')
            reference={'kind':'archive','archive':str(archive),'label':'reference','rules':{'size':3,'max_rounds':5,'collision':'gray'},
                'records':0,'policy_sha256':base.sha256,'weights_sha256':base.identity['weights_sha256']}
            q=FrozenNeuralPolicy.from_archive(archive,representation='greedy_q',label='q-candidate')
            candidate=dict(reference,head='greedy_q',label='q-candidate',policy_sha256=q.sha256,weights_sha256=q.identity['weights_sha256'])
            plan=GatePlan(1,looks=(4,))
            serial=IndependentGate(plan,reserve_bytes=0).evaluate(q,base,{'historical-0':base},root/'serial',round_index=0,time_cap_seconds=30)
            parallel=ParallelGate(plan,workers=2,reserve_bytes=0).evaluate_descriptors(candidate,reference,{'historical-0':reference},root/'parallel',round_index=0)
            self.assertEqual(serial,parallel)



if __name__=='__main__':unittest.main()

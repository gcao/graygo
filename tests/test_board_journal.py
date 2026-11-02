from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from graygo.board_journal import BoardJournalPlan,BoardMilestoneJournal,MilestonePending,OpeningPending,OpeningNoGo
from graygo.board_runtime import BoardRuntimeConfig,BoardNFSPRuntime
from graygo.confidence import canonical_sha
from graygo.engine import Rules
from graygo.evaluation import FrozenNeuralPolicy
from graygo.storage import CheckpointStore,digest,CorruptCheckpoint


class BoardJournalTests(unittest.TestCase):
    def config(self):return BoardRuntimeConfig(rules=Rules(3,5),envs=4,seed=5,batch=8,learning_start=8,capacity=256,target_every=16,
        width=8,blocks=1,norm_groups=2,reservoir_open=64,handoff_end=96)
    def plan(self):return BoardJournalPlan('diagnostic',256,(0,32,48,64,96,128,256),83)
    def create(self,path,*,resume=False,device='cpu'):
        runtime=BoardNFSPRuntime(replace(self.config(),device=device));journal=BoardMilestoneJournal(path,runtime,self.plan(),resume=resume,reserve_bytes=0)
        return runtime,journal
    def drive(self,runtime,journal,endpoint,*,capture_end=True):
        while runtime.records<endpoint:
            journal.capture_if_due()
            count=min(2*runtime.config.envs,runtime.backend.counters.until_q_boundary(runtime.backend.config),endpoint-runtime.records,journal.next_boundary()-runtime.records)
            journal.before_advance(count);runtime.advance(count,opening_enabled=journal.label_insertion_allowed())
        if capture_end:journal.capture_if_due()

    def test_exact_milestones_public_archive_origin_and_rng_neutral_load(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime,journal=self.create(directory)
            with self.assertRaises(MilestonePending):journal.before_advance(2)
            path=journal.capture();self.assertEqual(set(CheckpointStore.verify(path)['files']),{'models.pt'})
            before=torch.get_rng_state().clone();policy=FrozenNeuralPolicy.from_archive(path,representation='average')
            torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)
            self.assertEqual(policy.identity['origin']['training_run_sha256'],digest(Path(directory)/'run.json'))
            self.assertEqual(policy.identity['origin']['records'],0)
            self.drive(runtime,journal,48)
            self.assertEqual(set(journal.archived),{'0','32','48'})
            self.drive(runtime,journal,54,capture_end=False)
            with self.assertRaises(ValueError):journal.capture()
            for point in (0,32,48):self.assertTrue((Path(directory)/'weights'/f'{point:012d}').is_dir())

    def test_opening_stops_collection_and_never_inserts_early_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime,journal=self.create(directory);self.drive(runtime,journal,62)
            with self.assertRaises(MilestonePending):journal.before_advance(4)
            journal.before_advance(2);runtime.advance(2,opening_enabled=journal.label_insertion_allowed());journal.capture()
            self.assertEqual(runtime.reservoir.seen,0)
            with self.assertRaises(OpeningPending):journal.before_advance(2)
            decision=journal.diagnostic_opening('pass');self.assertEqual(decision['records'],64)
            self.drive(runtime,journal,256)
            self.assertGreater(runtime.reservoir.seen,0)
            self.assertTrue((runtime.reservoir.arrays['record'][:runtime.reservoir.size]>=64).all())
            with self.assertRaises(MilestonePending):journal.before_advance(2)

    def check_opening_overlay(self,device):
        with tempfile.TemporaryDirectory() as directory:
            full,full_j=self.create(Path(directory)/'full',device=device);self.drive(full,full_j,64);full_j.checkpoint();full_j.diagnostic_opening('pass')
            # Reusing the exact pending checkpoint is safe: the immutable decision
            # is separately bound to these exact milestone weights and run.
            full_j.checkpoint();self.drive(full,full_j,256);full_j.checkpoint()
            part,part_j=self.create(Path(directory)/'part',device=device);self.drive(part,part_j,64);checkpoint=part_j.checkpoint()
            state=json.loads((checkpoint/'controller.json').read_text());self.assertIsNone(state['opening_sha256'])
            part_j.diagnostic_opening('pass')
            restored,journal=self.create(Path(directory)/'part',resume=True,device=device);journal.restore()
            self.assertTrue(journal.label_insertion_allowed());self.drive(restored,journal,256);journal.checkpoint()
            for name in ('q','target','average'):
                for key,value in getattr(full.backend,name).state_dict().items():torch.testing.assert_close(value,getattr(restored.backend,name).state_dict()[key],rtol=0,atol=0)
            for a,b in ((full.replay,restored.replay),(full.reservoir,restored.reservoir)):
                for key,value in a.arrays.items():np.testing.assert_array_equal(value,b.arrays[key])
            self.assertEqual(full.collector.metrics,restored.collector.metrics)
            for key,rng in full.rng.items():self.assertEqual(rng.bit_generator.state,restored.rng[key].bit_generator.state)
            full_j.finalize();journal.finalize()
            self.assertEqual(len(list((Path(directory)/'part/resume').glob('[0-9]*'))),1)
            self.assertEqual(len(list((Path(directory)/'part/weights').glob('[0-9]*'))),7)

    def test_post_decision_crash_restores_pending_cpu_checkpoint_overlay_exactly(self):self.check_opening_overlay('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_post_decision_crash_restores_pending_cuda_checkpoint_overlay_exactly(self):self.check_opening_overlay('cuda')

    def test_future_archive_is_verified_on_replay_not_used_to_skip_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime,journal=self.create(directory);self.drive(runtime,journal,32);journal.checkpoint()
            self.drive(runtime,journal,48)
            restored,resumed=self.create(directory,resume=True);resumed.restore()
            self.assertEqual(restored.records,32);self.assertNotIn('48',resumed.archived)
            self.drive(restored,resumed,48,capture_end=False)
            with torch.no_grad():next(restored.backend.q.parameters()).add_(.001)
            with self.assertRaisesRegex(ValueError,'replayed milestone weights'):resumed.capture()

    def test_no_go_is_immutable_and_later_opening_is_not_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime,journal=self.create(directory);self.drive(runtime,journal,32)
            with self.assertRaisesRegex(ValueError,'original fixed boundary'):journal.diagnostic_opening('pass')
            with self.assertRaisesRegex(ValueError,'active/budget-limited'):journal.finalize()
            self.drive(runtime,journal,64);journal.diagnostic_opening('no_go')
            with self.assertRaises(OpeningNoGo):journal.before_advance(2)
            with self.assertRaisesRegex(ValueError,'immutable'):journal.diagnostic_opening('pass')
            self.assertEqual(int(journal.finalize().name),64)

    def test_controller_and_decision_integrity_fail_before_resume_can_advance(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime,journal=self.create(directory);self.drive(runtime,journal,64);path=journal.checkpoint();journal.diagnostic_opening('pass')
            decision=Path(directory)/'opening-decision.json';data=json.loads(decision.read_text());data['body']['status']='no_go';decision.write_text(json.dumps(data))
            fresh,restored=self.create(directory,resume=True)
            with self.assertRaisesRegex(ValueError,'checksum'):restored.restore(path)
        with tempfile.TemporaryDirectory() as directory:
            runtime,journal=self.create(directory);path=journal.checkpoint();seal=CheckpointStore.verify(path)
            self.assertEqual(seal['metadata']['controller_sha256'],canonical_sha(json.loads((path/'controller.json').read_text())))
            (path/'controller.json').write_text('{}')
            fresh,restored=self.create(directory,resume=True)
            with self.assertRaises(CorruptCheckpoint):restored.restore(path)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_public_archive_loading_preserves_cpu_and_cuda_training_rngs(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime=BoardNFSPRuntime(replace(self.config(),device='cuda'))
            journal=BoardMilestoneJournal(directory,runtime,self.plan(),reserve_bytes=0)
            journal.capture();self.drive(runtime,journal,64);journal.checkpoint()
            cpu=torch.get_rng_state().clone();cuda=[x.clone() for x in torch.cuda.get_rng_state_all()]
            FrozenNeuralPolicy.from_archive(journal._archive_path(64),representation='greedy_q',device='cuda')
            torch.testing.assert_close(torch.get_rng_state(),cpu,rtol=0,atol=0)
            for a,b in zip(cuda,torch.cuda.get_rng_state_all()):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_diagnostic_plan_cannot_allocate_production_or_issue_formal_permission(self):
        with self.assertRaises(ValueError):BoardJournalPlan('diagnostic',100000,(64,100000))
        with self.assertRaises(ValueError):BoardJournalPlan('development',20000000,(5000000,20000000))
        with tempfile.TemporaryDirectory() as directory:
            runtime,journal=self.create(directory)
            self.assertFalse(journal.manifest['scope_approval_verified'])
            with self.assertRaises(ValueError):journal.record_pilot_opening(None,{})
            with self.assertRaises(ValueError):BoardMilestoneJournal(directory,runtime,replace(self.plan(),evaluation_seed=84),resume=True,reserve_bytes=0)

if __name__=='__main__':unittest.main()

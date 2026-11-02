from fractions import Fraction
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from graygo.buffers import ReplayBuffer
from graygo.confidence import canonical_sha
from graygo.storage import CheckpointStore,digest
from graygo.task_training import NFSPTaskConfig,NFSPTaskTrainer
from graygo.task_adapter import TaskAdapter

AUDITOR=Path(__file__).resolve().parents[1]/'tools/audit_nfsp_bounded.py'
MODULE=runpy.run_path(str(AUDITOR))
AUDIT=MODULE['audit']


class BoundedAuditTests(unittest.TestCase):
    def fixture(self,root,task='branching',learner='tabular',records=64,storage='features',capacity=64,reuse=1,uniform=False):
        config=NFSPTaskConfig(time_cap_seconds=120,task=task,learner=learner,seed=5,records=records,eval_every=16,
            batch=8,learning_start=8,capacity=capacity,target_every=16,reservoir_open=8,handoff_end=32,epsilon_decay=32,
            horizons=(3,4),hidden=8,width=8,blocks=1,norm_groups=2,purpose='profile',storage=storage,q_reuse=reuse,uniform_comparison=uniform)
        trainer=NFSPTaskTrainer(config,root,reserve_bytes=0);self.assertEqual(trainer.run(verbose=False)['status'],'complete');return trainer

    def snapshot(self,root):return {p.relative_to(root):digest(p) for p in Path(root).rglob('*') if p.is_file()}

    def reseal(self,root,mutate):
        path=CheckpointStore(root,reserve_bytes=0).latest();mutate(path)
        manifest=json.loads((path/'manifest.json').read_text())
        for name in manifest['files']:
            payload=path/name;manifest['files'][name]={'bytes':payload.stat().st_size,'sha256':digest(payload)}
        (path/'manifest.json').write_text(json.dumps(manifest,sort_keys=True));(path/'manifest.sha256').write_text(digest(path/'manifest.json')+'\n')
        CheckpointStore.verify(path)

    def state_change(self,root,mutate):
        def change(path):
            state=torch.load(path/'trainer.pt',weights_only=True,map_location='cpu');mutate(state);torch.save(state,path/'trainer.pt')
        self.reseal(root,change)

    def test_tabular_and_neural_all_bounded_families_audit_read_only_without_trainer_evaluator(self):
        for task in ('branching','gray-pure','gray-mixed'):
            for learner in ('tabular','neural'):
                with self.subTest(task=task,learner=learner),tempfile.TemporaryDirectory() as directory:
                    trainer=self.fixture(directory,task,learner);before=self.snapshot(directory)
                    with patch('graygo.task_evaluation.evaluate_task',side_effect=AssertionError('trainer evaluator must not run')),\
                         patch('graygo.task_training.evaluate_task',side_effect=AssertionError('trainer evaluator alias must not run')),\
                         patch('graygo.task_evaluation._native_q_values',side_effect=AssertionError('trainer native-value recursion must not run')):
                        result=AUDIT(directory,require_formal=False)
                    self.assertTrue(result['verified']);self.assertEqual(result['records'],64)
                    self.assertEqual(result['retained_executed_records_checked'],64)
                    self.assertEqual(result['retained_joint_transition_records_checked'],64)
                    self.assertGreater(result['retained_complete_action_paths_reconstructed'],0)
                    self.assertFalse(result['completed_original_log_hashes_reconstructed']);self.assertFalse(result['formal_case_passed']);self.assertFalse(result['a2_complete'])
                    self.assertEqual(result['q_eligible_records'],trainer.backend.counters.fresh_q)
                    certificate=result['exact_game_certificate'];self.assertEqual(len(certificate['states']),len(trainer.adapter.states))
                    self.assertEqual(canonical_sha(certificate),result['exact_game_certificate_sha256'])
                    if task=='branching':
                        self.assertEqual(result['rational_root_values'],['-1/7','1/7'])
                        first=next(row for row in certificate['states'] if row['state_index']==certificate['root_state_indices'][0])
                        self.assertEqual(first['continuation_matrix'],[['-1/7'],['-7/22']]);self.assertEqual(first['row_strategy'],['1','0'])
                    if task=='gray-pure':self.assertEqual(result['rational_root_values'],['0'])
                    if task=='gray-mixed':
                        self.assertEqual(result['rational_root_values'],['-1/4'])
                        self.assertTrue(any(sum(Fraction(x)>0 for x in row[side])>1 for row in certificate['states'] for side in ('row_strategy','column_strategy')))
                    self.assertEqual(before,self.snapshot(directory))
                    with self.assertRaisesRegex(ValueError,'formal primary'):AUDIT(directory)

    def test_rule_state_packets_are_decoded_against_actual_engine_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory,'gray-mixed','neural',storage='rule-state');before=self.snapshot(directory)
            result=AUDIT(directory,require_formal=False)
            self.assertTrue(result['verified']);self.assertEqual(result['retained_joint_transition_records_checked'],64)
            self.assertEqual(before,self.snapshot(directory))

    def test_active_meaningful_branch_continuation_and_final_five_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory,records=162);result=AUDIT(directory,require_formal=False)
            self.assertEqual(result['active_episode_steps'],1)
            self.assertEqual([row['records'] for row in result['scheduled_evaluations']],[*range(16,161,16),162])
            self.assertFalse(result['formal_protocol'])

    def test_component_posterior_uses_revealed_root_move(self):
        c=NFSPTaskConfig(60,task='branching',purpose='profile',records=64,hidden=8,reservoir_open=8,handoff_end=32)
        adapter=TaskAdapter(c);exact=MODULE['ExactGame'](adapter);target=np.zeros((len(adapter.states),2,2));average=np.zeros_like(target)
        for public,index in adapter.index.items():
            for role in (1,2):
                legal=adapter.game.actions(public,role);target[index,role-1,list(legal)]=1/len(legal);average[index,role-1,list(legal)]=1/len(legal)
            controller=public.controller
            if public.depth==0:target[index,controller-1]=(.9,.1);average[index,controller-1]=(.1,.9)
            else:target[index,controller-1]=(.8,.2);average[index,controller-1]=(.2,.8)
        roots=MODULE['independent_roots'](adapter,exact,np.zeros_like(target),average,target,target,32,True)
        root=adapter.roots[0];public=adapter.game.transition(root,0,0);index=adapter.index[public]
        expected=next(row['expected'] for row in roots[0]['early_values'] if row['state_index']==index and row['player']==2)
        # Prior (.1,.9), likelihood (.9,.1) -> posterior (.5,.5).
        # The two fixed-component white values are .05 and .20.
        self.assertAlmostEqual(expected,float(Fraction(1,8)),places=12)
        self.assertGreater(abs(expected-(.1*.05+.9*.20)),.01)

    def test_validly_resealed_wrong_native_value_fails_independent_math(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            def mutate(state):state['evaluations'][-1]['roots'][0]['root_action_values'][0]['expected']+=.25
            self.state_change(directory,mutate)
            with self.assertRaisesRegex(ValueError,'numerical result differs'):AUDIT(directory,require_formal=False)

    def test_resealed_legal_action_with_wrong_successor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            def mutate(path):
                replay=ReplayBuffer.load(path/'replay.npz');a=replay.arrays
                for slot in range(replay.size):
                    if int(a['step'][slot])==0 and int(a['stream'][slot])%2==0:
                        _,_,legal=replay.spec.unpack(a['spatial'][slot:slot+1],a['globals'][slot:slot+1],a['legal'][slot:slot+1])
                        if legal[0].sum()==2:a['action'][slot]=1-int(a['action'][slot]);break
                else:self.fail('missing branch-controlling root record')
                replay.save(path/'replay.npz')
            self.reseal(directory,mutate)
            with self.assertRaisesRegex(ValueError,'joint transition|forced root'):AUDIT(directory,require_formal=False)

    def test_resealed_rng_or_q_clock_drift_is_rejected(self):
        for variant in ('rng','clock'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as directory:
                self.fixture(directory)
                def mutate(state):
                    if variant=='rng':state['rng']['action_black']['state']['state']+=1
                    else:state['backend']['counters']['fresh_q']+=1
                self.state_change(directory,mutate)
                with self.assertRaisesRegex(ValueError,'private draw count|eligible Q'):AUDIT(directory,require_formal=False)

    def test_active_actor_rng_cannot_disagree_with_the_global_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory,records=162)
            def mutate(state):state['active']['episode']['actors']['1']['rng']['state']['state']+=1
            self.state_change(directory,mutate)
            with self.assertRaisesRegex(ValueError,'active actor RNG'):AUDIT(directory,require_formal=False)

    def test_active_original_action_probability_is_reconstructed_from_frozen_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory,records=162)
            def mutate(state):state['active']['episode']['log'][0]['probabilities'][0]+=1e-10
            self.state_change(directory,mutate)
            with self.assertRaisesRegex(ValueError,'active frozen action probability'):AUDIT(directory,require_formal=False)

    def test_missing_public_archive_and_mutated_codebook_are_rejected(self):
        for variant in ('archive','codebook'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as directory:
                self.fixture(directory);root=Path(directory)
                if variant=='archive':(root/'weights/000000000016').rename(root/'weights/missing-first-look')
                else:
                    manifest=json.loads((root/'run.json').read_text());manifest['adapter']['codebook_sha256']='a'*64
                    (root/'run.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'archive inventory|codebook'):AUDIT(directory,require_formal=False)

    def test_published_tabular_means_verified_when_tail_replaces_no_old_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory,records=160,capacity=256);result=AUDIT(directory,require_formal=False)
            self.assertTrue(result['published_tabular_mean_verified']);self.assertEqual(result['unresolved_published_labels'],0)
            self.assertGreater(result['published_tabular_exact_mean_states'],0)
            def mutate(state):
                table=state['backend']['average']['probabilities'];table[0]=torch.tensor([.123,.877],dtype=table.dtype)
            self.state_change(directory,mutate)
            with self.assertRaisesRegex(ValueError,'published exact tabular'):AUDIT(directory,require_formal=False)

    def test_concurrent_writer_prevents_any_audit_read(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            with (Path(directory)/'.trainer.lock').open('rb') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                try:
                    with self.assertRaises(BlockingIOError):AUDIT(directory,require_formal=False)
                finally:fcntl.flock(lock,fcntl.LOCK_UN)

    def test_installed_script_cli_emits_receipt_and_protects_training_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run';self.fixture(root);before=self.snapshot(root);output=Path(directory)/'audit.json'
            command=[sys.executable,str(AUDITOR),str(root),'--allow-diagnostic','--time-cap-seconds','60','--output',str(output)]
            result=subprocess.run(command,text=True,capture_output=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stderr);report=json.loads(output.read_text())
            self.assertTrue(report['verified']);self.assertEqual(report['audit_watchdog_seconds'],60)
            self.assertLess(report['measured_audit_entry_wall_seconds'],60);self.assertEqual(before,self.snapshot(root))
            command[-1]=str(root/'result.json');rejected=subprocess.run(command,text=True,capture_output=True,timeout=90)
            self.assertNotEqual(rejected.returncode,0);self.assertIn('outside',rejected.stderr);self.assertEqual(before,self.snapshot(root))

    def test_odd_capacity_retains_one_explicitly_unpaired_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory,capacity=63);result=AUDIT(directory,require_formal=False)
            self.assertEqual(result['retained_unpaired_transition_records_checked'],1)
            self.assertEqual(result['retained_joint_transition_records_checked'],62)

    def test_audit_preserves_caller_cpu_rng_and_thread_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory);before=torch.get_rng_state().clone();old=torch.get_num_threads()
            try:
                torch.set_num_threads(2);AUDIT(directory,require_formal=False)
                self.assertEqual(torch.get_num_threads(),2);self.assertTrue(torch.equal(before,torch.get_rng_state()))
            finally:torch.set_num_threads(old)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA default-device guard requires CUDA')
    def test_cpu_verification_ignores_ambient_cuda_and_preserves_both_rngs(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory);cpu=torch.get_rng_state().clone();cuda=[s.clone() for s in torch.cuda.get_rng_state_all()]
            with torch.device('cuda'):result=AUDIT(directory,require_formal=False)
            self.assertTrue(result['verified']);self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())))

    def test_mutated_dependency_version_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory);path=Path(directory)/'run.json';manifest=json.loads(path.read_text())
            manifest['dependencies']['torch']='0.0';path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'pinned dependency'):AUDIT(directory,require_formal=False)

    def test_overwritten_unpublished_label_keeps_its_mean_uncertainty_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory,records=190,capacity=8);result=AUDIT(directory,require_formal=False)
            self.assertGreater(result['unresolved_published_labels'],0)
            self.assertTrue(result['published_tabular_mean_interval_checked'])
            self.assertFalse(result['published_tabular_mean_verified'])
            self.assertEqual(len(result['published_tabular_uncertain_tokens']),result['published_tabular_uncertain_mean_states'])
            self.assertGreater(result['published_tabular_exact_mean_states'],0)

    def test_reuse_four_and_uniform_comparator_preserve_independent_clocks_and_values(self):
        for reuse,uniform in ((4,False),(1,True)):
            with self.subTest(reuse=reuse,uniform=uniform),tempfile.TemporaryDirectory() as directory:
                trainer=self.fixture(directory,records=162,reuse=reuse,uniform=uniform)
                result=AUDIT(directory,require_formal=False);self.assertTrue(result['verified'])
                self.assertEqual(result['q_updates'],trainer.backend.counters.q_updates)
                self.assertEqual(result['q_loss_targets'],result['q_updates']*trainer.config.batch)
                self.assertFalse(result['formal_case_passed'])


if __name__=='__main__':unittest.main()

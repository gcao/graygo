from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from graygo.task_training import NFSPTaskConfig,NFSPTaskTrainer
from tools.retrace_recording import BoundaryRecorder,file_sha
from tools.audit_retrace_boundaries import audit,equal


def config(task='delayed'):
    return NFSPTaskConfig(120,task=task,learner='neural',seed=13,records=128,eval_every=32,
        batch=4,learning_start=4,capacity=128,target_every=8,q_reuse=4,q_target='retrace',
        target_batch=4,reservoir_open=0,handoff_end=8,epsilon_decay=40,horizons=(5,),
        equivalent_actions=3,hidden=8,width=8,blocks=1,norm_groups=2,purpose='profile')


def snapshot(root):return {str(p.relative_to(root)):file_sha(p) for p in Path(root).rglob('*') if p.is_file()}


class RetraceBoundaryTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def recorded(self,root,task='delayed',records=None):
        cfg=config(task)
        if records is not None:cfg=replace(cfg,records=records)
        trainer=NFSPTaskTrainer(cfg,Path(root)/'trial',reserve_bytes=0)
        journal=Path(root)/'recording'
        with BoundaryRecorder(trainer.backend,journal,cfg,q_boundaries=(4,8),reserve_bytes=0):
            trainer.run(verbose=False)
        return trainer,journal

    def reseal(self,journal,mutate):
        header=json.loads((journal/'recording.json').read_text());row=header['boundaries'][0]
        path=journal/row['file'];item=torch.load(path,weights_only=True,map_location='cpu')
        mutate(item);torch.save(item,path)
        row['sha256']=file_sha(path);row['bytes']=path.stat().st_size
        header['recorded_bytes']=sum(r['bytes'] for r in header['boundaries'])
        (journal/'recording.json').write_text(json.dumps(header,indent=2)+'\n')

    def test_actual_q_supervised_and_refresh_boundaries_verify_without_production_helpers(self):
        with tempfile.TemporaryDirectory() as root:
            trainer,journal=self.recorded(root);before=snapshot(journal);rng=torch.get_rng_state().clone()
            with patch('graygo.sequence_targets.retrace_targets',side_effect=AssertionError('production targets forbidden')),                  patch('graygo.sequence_targets.SequenceBatch.from_replay',side_effect=AssertionError('production batch validation forbidden')),                  patch('graygo.nfsp.apply_q_update',side_effect=AssertionError('production update forbidden')),                  patch('graygo.buffers.ReplayBuffer.sample_sequences',side_effect=AssertionError('production sampling forbidden')),                  patch('graygo.buffers.ReservoirBuffer.sample',side_effect=AssertionError('production reservoir sampling forbidden')):
                result=audit(journal)
            self.assertTrue(result['verified']);self.assertEqual(before,snapshot(journal))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(any(r['target_refreshed'] for r in result['boundaries']))
            self.assertTrue(any(r['supervised_update'] for r in result['boundaries']))
            self.assertGreater(sum(r['context_records'] for r in result['boundaries']),0)
            self.assertFalse(result['production_target_sampler_cadence_or_update_called'])
            self.assertFalse(result['full_training_history_verified']);self.assertFalse(result['source_admission_qualified'])

    def test_observation_preserves_all_model_optimizer_rng_buffer_and_trajectory_state(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=config()
            plain=NFSPTaskTrainer(cfg,Path(root)/'plain',reserve_bytes=0);plain.run(verbose=False)
            observed,journal=self.recorded(Path(root)/'observed')
            def strip(value):
                if isinstance(value,dict):return {k:strip(v) for k,v in value.items() if k not in ('timings','elapsed','elapsed_seconds')}
                if isinstance(value,list):return [strip(v) for v in value]
                if isinstance(value,tuple):return tuple(strip(v) for v in value)
                return value
            equal(strip(plain._state()),strip(observed._state()))
            for name in plain.replay.arrays:np.testing.assert_array_equal(plain.replay.arrays[name],observed.replay.arrays[name])
            for name in plain.reservoir.arrays:np.testing.assert_array_equal(plain.reservoir.arrays[name],observed.reservoir.arrays[name])
            for name in plain.metrics:np.testing.assert_array_equal(plain.metrics[name],observed.metrics[name])
            self.assertEqual((Path(root)/'plain/episodes.bin').read_bytes(),(Path(root)/'observed/trial/episodes.bin').read_bytes())

    def test_resealed_target_gradient_optimizer_counter_and_sample_changes_are_rejected(self):
        mutations=[
            lambda x:x['q_steps'][0]['details']['raw_targets'].add_(.01),
            lambda x:x['q_steps'][0]['targets'].zero_(),
            lambda x:next(iter(x['q_steps'][0]['post']['q'].values())).add_(.001),
            lambda x:x['post']['counters'].__setitem__('q_loss_targets',x['post']['counters']['q_loss_targets']+4),
            lambda x:x['q_steps'][0]['data']['behavior_prob'].__setitem__((0,0),.77),
            lambda x:x['post']['target'][next(iter(x['post']['target']))].add_(.001)]
        with tempfile.TemporaryDirectory() as root:
            _,original=self.recorded(root)
            import shutil
            for index,mutate in enumerate(mutations):
                with self.subTest(mutation=index):
                    target=Path(root)/f'mutated-{index}';shutil.copytree(original,target)
                    self.reseal(target,mutate)
                    with self.assertRaises(ValueError):audit(target)

    def test_cnn_task_boundaries_have_independent_exact_optimizer_replay(self):
        with tempfile.TemporaryDirectory() as root:
            _,journal=self.recorded(root,'gray-mixed',records=128)
            result=audit(journal)
            self.assertTrue(result['all_optimizer_tensors_bit_exact'])
            self.assertTrue(all(r['all_sampler_inputs_and_rngs_exact'] for r in result['boundaries']))

    def test_incomplete_selection_is_not_promoted_and_context_failure_restores_backend(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=replace(config(),records=8)
            trainer=NFSPTaskTrainer(cfg,Path(root)/'trial',reserve_bytes=0)
            journal=Path(root)/'recording'
            original=trainer.backend.learn_after_fresh
            with BoundaryRecorder(trainer.backend,journal,cfg,q_boundaries=(4,8,16),reserve_bytes=0):
                trainer.run(verbose=False)
            self.assertNotIn('learn_after_fresh',trainer.backend.__dict__)
            with self.assertRaisesRegex(ValueError,'incomplete'):audit(journal)
            cfg=config();other=NFSPTaskTrainer(cfg,Path(root)/'failure-trial',reserve_bytes=0)
            import graygo.sequence_targets as target
            function=target.update_retrace_q
            with self.assertRaisesRegex(RuntimeError,'stop fixture'):
                with BoundaryRecorder(other.backend,Path(root)/'failed',cfg,q_boundaries=(4,8),reserve_bytes=0):
                    raise RuntimeError('stop fixture')
            self.assertIs(target.update_retrace_q,function)
            self.assertNotIn('learn_after_fresh',other.backend.__dict__)
            self.assertEqual(json.loads((Path(root)/'failed/recording.json').read_text())['status'],'failed')


if __name__=='__main__':unittest.main()

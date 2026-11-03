from pathlib import Path
import json
import tempfile
import unittest
from graygo.nfsp_training import NFSPMatrixConfig,NFSPMatrixTrainer
from tools.profile_terminal_batch import cases,compare_pair


class TerminalBatchProfileTests(unittest.TestCase):
    def test_fixed_work_inventory_includes_both_buffer_wrap_stress(self):
        rows=cases();self.assertEqual(len(rows),12)
        self.assertEqual(len({row['label'] for row in rows}),12)
        stress=[row for row in rows if row['label'].startswith('wrap-stress')]
        self.assertEqual(len(stress),4);self.assertTrue(all(row['config']['reservoir_open']==0 and row['config']['capacity']==4096 for row in stress))
        self.assertEqual(2*sum(row['config']['records'] for row in rows),2124288)

    def test_real_comparison_preserves_metrics_and_rejects_false_mode_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=NFSPMatrixConfig(records=128,eval_every=64,batch=16,learning_start=32,capacity=64,hidden=8,reservoir_open=0,handoff_end=32)
            for name,mode in (('ref','scalar-reference'),('bulk','terminal-batch')):
                trainer=NFSPMatrixTrainer(cfg,root/name,reserve_bytes=0,collection=mode);trainer.run(verbose=False)
            proof=compare_pair(root/'ref',root/'bulk');self.assertIn('trainer_state',proof['equal_partitions'])
            with self.assertRaises(ValueError):compare_pair(root/'bulk',root/'bulk')
            path=root/'bulk/result.json';result=json.loads(path.read_text());result['loss_stats']['q_loss_sum']+=.001;path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'result'):compare_pair(root/'ref',root/'bulk')

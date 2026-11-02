from pathlib import Path
import json
import tempfile
import unittest
from graygo.nfsp_training import NFSPMatrixConfig,NFSPMatrixTrainer
from tools.audit_nfsp_matrix import audit


class MatrixAuditTests(unittest.TestCase):
    def test_archives_rng_reservoir_and_reported_gate_are_independently_checked(self):
        for learner in ('neural','tabular'):
            with self.subTest(learner=learner),tempfile.TemporaryDirectory() as directory:
                config=NFSPMatrixConfig(learner=learner,seed=2,records=1024,eval_every=128,batch=16,
                    learning_start=32,capacity=32,target_every=64,hidden=8,reservoir_open=32,handoff_end=96,epsilon_decay=160)
                t=NFSPMatrixTrainer(config,directory,reserve_bytes=0);result=t.run(verbose=False)
                checked=audit(directory,require_formal=False)
                self.assertTrue(checked['verified']);self.assertEqual(checked['records'],1024)
                self.assertEqual(checked['average_final_five_passed'],result['average_final_five_passed'])
                report=Path(directory)/'result.json';original=report.read_bytes()
                bad=json.loads(original);bad['average_final_five_passed']=not bad['average_final_five_passed']
                report.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):audit(directory,require_formal=False)
                report.write_bytes(original)
                log=Path(directory)/'episodes.bin';data=bytearray(log.read_bytes());data[0]^=1;log.write_bytes(data)
                with self.assertRaises(ValueError):audit(directory,require_formal=False)

if __name__=='__main__':unittest.main()

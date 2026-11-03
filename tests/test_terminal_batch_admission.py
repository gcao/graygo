from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import subprocess
import unittest
from graygo.nfsp_training import NFSPMatrixConfig
from tools.terminal_batch_admission import VERSION,prove_storage_and_collector,check_matrix_case


class MatrixSourceAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo=Path(__file__).resolve().parents[1]
        cls.old={name:subprocess.check_output(['git','show','c3b8b5f:src/graygo/'+name],cwd=cls.repo) for name in ('buffers.py','nfsp_training.py')}
        cls.new={name:(cls.repo/'src/graygo'/name).read_bytes() for name in cls.old}

    def test_current_extension_preserves_scalar_code_and_checked_substitution(self):
        r=prove_storage_and_collector(self.old,self.new);self.assertTrue(r['scalar_storage_and_format_preserved']);self.assertTrue(r['matrix_other_semantics_preserved'])

    def test_learning_sampler_storage_or_bulk_argument_changes_fail(self):
        mutations=[('nfsp_training.py',b'self.episodes+=games',b'self.episodes+=games+1'),
                   ('nfsp_training.py',b'behavior_prob=probabilities,stream=',b'behavior_prob=probabilities/2,stream='),
                   ('buffers.py',b'if terminated:self._tails.pop(stream,None)',b'if terminated:self._tails.clear()')]
        for name,before,after in mutations:
            new=dict(self.new);self.assertIn(before,new[name]);new[name]=new[name].replace(before,after)
            with self.subTest(before=before),self.assertRaises(ValueError):prove_storage_and_collector(self.old,new)
        old,new=dict(self.old),dict(self.new);old['nfsp.py']=b'old';new['nfsp.py']=b'changed'
        with self.assertRaises(ValueError):prove_storage_and_collector(old,new)

    def test_source_and_runtime_admission_never_loosens_primary_recipe(self):
        proof={'version':VERSION,'status':'qualified_cpu_matrix_terminal_batch','candidate_source_sha256':'b'*64,'dependencies':{'torch':'test'}}
        manifest={'source_sha256':'b'*64,'collection_implementation':'terminal-batch','config':asdict(NFSPMatrixConfig()),'dependencies':{'torch':'test'}}
        self.assertTrue(check_matrix_case(manifest,proof))
        for mutate in (lambda m:m.update(source_sha256='a'*64),lambda m:m.update(collection_implementation='scalar-reference'),
                       lambda m:m['config'].update(records=100000),lambda m:m['config'].update(seed=3),
                       lambda m:m['config'].update(device='cuda'),lambda m:m.update(dependencies={'torch':'other'})):
            changed=deepcopy(manifest);mutate(changed)
            with self.assertRaises(ValueError):check_matrix_case(changed,proof)

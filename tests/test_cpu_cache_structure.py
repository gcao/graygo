from pathlib import Path
import subprocess
import unittest
from tools.cpu_cache_structure import prove


class CPUCacheStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo=Path(__file__).resolve().parents[1]
        cls.candidate_commit='257276e28bbceb10cdbfb715c31c7e246721bc4f'
        files=subprocess.check_output(['git','ls-tree','-r','--name-only',cls.candidate_commit,'src/graygo'],cwd=cls.repo,text=True).splitlines()
        cls.candidate={name.removeprefix('src/graygo/'):subprocess.check_output(['git','show',cls.candidate_commit+':'+name],cwd=cls.repo)
                       for name in files if name.endswith(('.py','.cpp'))}
        cls.reference={}
        for name in cls.candidate:
            r=subprocess.run(['git','show','c3b8b5f:src/graygo/'+name],cwd=cls.repo,capture_output=True)
            if r.returncode==0:cls.reference[name]=r.stdout

    def test_all_original_code_is_recovered_after_only_declared_implementation_additions(self):
        result=prove(self.reference,self.candidate);self.assertTrue(result['literal_original_snapshot_actor_collection_and_serialization_ast_recovered']);self.assertTrue(result['scalar_storage_and_format_preserved']);self.assertFalse(result['checkpoint_schema_changed']);self.assertFalse(result['source_admission_granted'])

    def test_changed_model_call_serialization_learning_or_supported_types_is_rejected(self):
        mutations=[('nfsp_collection.py',b'return self._cpu_forward_cache.forward(model,**inputs)',b'return self._cpu_forward_cache.forward(model,**inputs)*.9'),('nfsp_collection.py',b'self.version=version;',b'self.version=version+1;'),('nfsp.py',b'self.counters.q_updates+=1',b'self.counters.q_updates+=2'),('cpu_frozen_forward.py',b'type(model) in (QNet,PolicyNet)',b'isinstance(model,(QNet,PolicyNet))')]
        for name,before,after in mutations:
            changed=dict(self.candidate);self.assertIn(before,changed[name]);changed[name]=changed[name].replace(before,after)
            with self.assertRaises(ValueError,msg=name):prove(self.reference,changed)


if __name__=='__main__':unittest.main()

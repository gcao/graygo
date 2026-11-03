from copy import deepcopy
import unittest
from graygo.task_training import NFSPTaskConfig
from tools.cpu_cache_admission import VERSION,check_case


class CPUCacheAdmissionTests(unittest.TestCase):
    def fixture(self,task='gray-pure',learner='neural',seed=0):
        proof={'version':VERSION,'status':'qualified_cpu_3x3_raw_cache','source_compatibility_qualified':True,'candidate_source_sha256':'b'*64,'dependencies':{'torch':'test'},'execution_authorized':False}
        manifest={'source_sha256':'b'*64,'config':NFSPTaskConfig(time_cap_seconds=120000,task=task,learner=learner,seed=seed,q_reuse=4,budget_profile_sha256='a'*64).plan(),'runtime':{'device':'cpu','torch_threads':1,'workers':1},'dependencies':{'torch':'test'}}
        return manifest,proof

    def test_each_bounded_original_learner_seed_keeps_its_own_source_and_plan_requirements(self):
        for task in ('gray-pure','gray-mixed'):
            for learner in ('neural','tabular'):
                for seed in (0,1,2):
                    manifest,proof=self.fixture(task,learner,seed);before=deepcopy((manifest,proof));r=check_case(manifest,proof);self.assertTrue(r['original_source_identity_preserved']);self.assertTrue(r['primary_original_plan_and_independent_case_audit_required']);self.assertFalse(r['a2_complete']);self.assertFalse(r['execution_authorized']);self.assertEqual((manifest,proof),before)

    def test_other_device_family_storage_seed_recipe_and_dependency_are_not_admitted(self):
        manifest,proof=self.fixture()
        for field,value in [('device','cuda'),('task','branching'),('seed',3),('records',10000000),('q_reuse',1),('uniform_comparison',True),('storage','rule-state')]:
            changed=deepcopy(manifest);changed['config'][field]=value
            with self.assertRaises(ValueError,msg=field):check_case(changed,proof)
        changed=deepcopy(manifest);changed['source_sha256']='a'*64
        with self.assertRaises(ValueError):check_case(changed,proof)
        changed=deepcopy(manifest);changed['dependencies']={'torch':'other'}
        with self.assertRaises(ValueError):check_case(changed,proof)
        changed=deepcopy(manifest);changed['runtime']['torch_threads']=2
        with self.assertRaises(ValueError):check_case(changed,proof)


if __name__=='__main__':unittest.main()

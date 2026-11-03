from copy import deepcopy
from pathlib import Path
import unittest
from graygo.task_training import NFSPTaskConfig
from tools.profile_nfsp_bounded_cuda import profile_arguments


class BoundedCUDAProfileTests(unittest.TestCase):
    def fixture(self):
        tasks=['gray-pure','gray-mixed'];return {'tasks':tasks,'device':'cuda:0','learner':'neural','seed':0,'q_reuse':4,'records_per_profile':32768,'profile_case_loop_cap_seconds':300,'new_formal_a2_records':0,'deterministic':True,'cudnn_tf32':False,'matmul_tf32':False,'root':'runs/cuda-profile','profiler':'tools/profile_nfsp_tasks.py','learner_source_sha256':'a'*64,'profiles':{task:[{'phase':p,'config':NFSPTaskConfig(time_cap_seconds=300,task=task,learner='neural',records=32768,eval_every=16384,q_reuse=4,device='cuda:0',purpose='profile',reservoir_open=5000000 if p=='pre' else 0,handoff_end=7000000 if p=='pre' else 1000).plan()} for p in ('pre','post')] for task in tasks}}

    def test_both_fixed_profiles_use_actual_complete_source_and_no_resume(self):
        spec=self.fixture();before=deepcopy(spec)
        for task in spec['tasks']:
            root,args=profile_arguments(spec,task,Path('/project'));self.assertEqual(root,Path('/project/runs/cuda-profile/profiles')/task);self.assertEqual(args[args.index('--device')+1],'cuda:0');self.assertEqual(args[args.index('--records')+1],'32768');self.assertNotIn('--resume',args)
        self.assertEqual(spec,before)

    def test_changed_task_precision_seed_records_model_or_output_scope_is_rejected(self):
        spec=self.fixture()
        for field,value in [('device','cpu'),('learner','tabular'),('seed',1),('q_reuse',1),('records_per_profile',15000000),('new_formal_a2_records',1),('deterministic',False),('cudnn_tf32',True),('matmul_tf32',True),('root','/outside'),('root','runs/../outside')]:
            changed=deepcopy(spec);changed[field]=value
            with self.assertRaises(ValueError,msg=field):profile_arguments(changed,'gray-pure','/project')
        changed=deepcopy(spec);changed['profiles']['gray-pure'][0]['config']['width']=32
        with self.assertRaises(ValueError):profile_arguments(changed,'gray-pure','/project')
        with self.assertRaises(ValueError):profile_arguments(spec,'branching','/project')


if __name__=='__main__':unittest.main()

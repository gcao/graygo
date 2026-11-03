from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from graygo.task_training import NFSPTaskConfig
from tools.run_nfsp_cpu_delayed_case import case_settings,fresh_case_directory
from tools.nfsp_delayed_cpu_prices import VERSION


class FundedDelayedCaseTests(unittest.TestCase):
    def fixture(self):
        identity={k:{} for k in ['hostname','cpu','python','platform','dependencies','runtime']}
        price={'q_reuse':4,'device_hours':'0','cpu_process_ceiling_seconds':'19800','raw_budget_sha256':'a'*64}
        proof={'version':VERSION,'status':'qualified_prospective_cpu_delayed_prices','target_runtime_source_sha256':'b'*64,'prices':{'C4':price},'runtime_identity':identity}
        allocation={'seeds':[1,2],'task':'delayed','learner':'neural','target':'one-step','q_reuse':4,'additional_control_reservation_seconds':'0','new_accelerator_jobs':0,'learner_source_sha256':'b'*64,'case_roots':{'1':'runs/seed1','2':'runs/seed2'},'case_configs':{str(s):NFSPTaskConfig(time_cap_seconds=19800,seed=s,q_reuse=4,budget_profile_sha256='a'*64).plan() for s in (1,2)}}
        return allocation,proof,identity

    def test_only_original_remaining_seeds_receive_the_complete_qualified_recipe(self):
        allocation,proof,identity=self.fixture();before=deepcopy(allocation)
        for seed in (1,2):
            config=case_settings(allocation,seed,proof,identity);self.assertTrue(config.formal_protocol());self.assertEqual(config.records,15000000);self.assertEqual(config.eval_every,500000);self.assertEqual(config.seed,seed)
        self.assertEqual(allocation,before)
        for seed in (0,3,True):
            with self.assertRaises(ValueError):case_settings(allocation,seed,proof,identity)

    def test_source_recipe_seed_alias_or_unfunded_resource_change_is_rejected(self):
        allocation,proof,identity=self.fixture()
        for field,value in [('task','branching'),('learner','tabular'),('q_reuse',1),('target','retrace'),('additional_control_reservation_seconds','30'),('new_accelerator_jobs',1),('learner_source_sha256','c'*64)]:
            changed=deepcopy(allocation);changed[field]=value
            with self.assertRaises(ValueError,msg=field):case_settings(changed,1,proof,identity)
        for field,value in [('seed',2),('device','cuda'),('records',10000000),('reservoir_open',0),('time_cap_seconds',18000)]:
            changed=deepcopy(allocation);changed['case_configs']['1'][field]=value
            with self.assertRaises(ValueError,msg=field):case_settings(changed,1,proof,identity)
        for bad_root in ['/tmp/seed1','runs/../unsafe','docs/seed1']:
            changed=deepcopy(allocation);changed['case_roots']['1']=bad_root
            with self.assertRaises(ValueError):case_settings(changed,1,proof,identity)
        changed=deepcopy(allocation);changed['case_roots']['2']=changed['case_roots']['1']
        with self.assertRaises(ValueError):case_settings(changed,1,proof,identity)

    def test_existing_case_is_preserved_instead_of_implicitly_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'original';root.mkdir();marker=root/'run.json';marker.write_text('original data')
            with self.assertRaises(ValueError):
                with fresh_case_directory(root):self.fail('existing original case admitted')
            self.assertEqual(marker.read_text(),'original data')

    def test_fresh_case_lease_excludes_a_second_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'fresh'
            with fresh_case_directory(root):
                self.assertFalse(root.exists())
                with self.assertRaises(BlockingIOError):
                    with fresh_case_directory(root):self.fail('second writer admitted')


if __name__=='__main__':unittest.main()

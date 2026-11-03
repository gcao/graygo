"""Synthetic case eligibility exercises constraints, not real qualification."""
from copy import deepcopy
import unittest
from pathlib import Path
import tempfile
import fcntl
from unittest.mock import patch
from graygo.task_training import NFSPTaskConfig
from graygo.nfsp_training import NFSPMatrixConfig
from tools.task_source_bridge import VERSION,check_case,qualify


OLD='a'*64
NEW='b'*64
DEPENDENCIES={'numpy':'test','scipy':'test','torch':'test'}


def bridge():return {'version':VERSION,'status':'qualified_cpu_features_A2_source_bridge','source_compatibility_qualified':True,
                    'reference_source_sha256':OLD,'candidate_source_sha256':NEW,'dependencies':DEPENDENCIES}


def task(**changes):
    config=NFSPTaskConfig(time_cap_seconds=15300,q_reuse=4,budget_profile_sha256='c'*64).plan();config.update(changes)
    return {'source_sha256':OLD,'config':config,'dependencies':DEPENDENCIES,'runtime':{'device':'cpu','torch_threads':1}}


class SourceBridgeEligibilityTests(unittest.TestCase):
    def check(self,manifest,receipt=None,candidate=NEW):return check_case(manifest,bridge() if receipt is None else receipt,selected_candidate_source=candidate)

    def test_original_primary_case_applies_without_rewriting_source_or_granting_A2(self):
        manifest=task();before=deepcopy(manifest);result=self.check(manifest)
        self.assertTrue(result['source_bridge_applies']);self.assertEqual(manifest,before)
        self.assertTrue(result['original_case_plan_and_audit_required']);self.assertFalse(result['a2_complete']);self.assertFalse(result['execution_authorized'])

    def test_all_primary_task_families_and_original_seeds_are_supported(self):
        for family in ('delayed','branching','gray-pure','gray-mixed'):
            for seed in (0,1,2):
                for learner in ('neural','tabular'):
                    self.assertTrue(self.check(task(task=family,seed=seed,learner=learner))['source_bridge_applies'])
        from dataclasses import asdict
        for family in ('asymmetric','rps'):
            manifest={'source_sha256':OLD,'config':asdict(NFSPMatrixConfig(task=family)),'dependencies':DEPENDENCIES}
            self.assertTrue(self.check(manifest)['source_bridge_applies'])

    def test_source_relabeling_or_another_selected_package_is_rejected(self):
        manifest=task();manifest['source_sha256']=NEW
        with self.assertRaises(ValueError):self.check(manifest)
        with self.assertRaises(ValueError):self.check(task(),candidate='d'*64)
        receipt=bridge();receipt['source_compatibility_qualified']=False
        with self.assertRaises(ValueError):self.check(task(),receipt)

    def test_diagnostics_changed_recipes_devices_seeds_and_storage_are_rejected(self):
        for changes in ({'purpose':'profile'},{'records':65536},{'seed':3},{'seed':True},{'q_reuse':1},
                        {'device':'cuda'},{'uniform_comparison':True},{'epsilon_decay':40000000},
                        {'task':'gray-pure','storage':'rule-state'},{'task':'delayed-forced'}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.check(task(**changes))

    def test_dependency_and_thread_changes_are_rejected(self):
        for mutate in (lambda m:m.update(dependencies={'torch':'other'}),lambda m:m['runtime'].update(torch_threads=2),
                       lambda m:m['runtime'].update(torch_threads=True),lambda m:m.pop('runtime')):
            manifest=deepcopy(task());mutate(manifest)
            with self.assertRaises(ValueError):self.check(manifest)

    def test_busy_original_trainer_blocks_qualification_before_reading_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'runs/nfsp-eligibility-R4-primary-ca45a53/neural';root.mkdir(parents=True)
            lock=root/'.trainer.lock';lock.write_bytes(b'')
            with lock.open('rb') as writer:
                fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch('tools.task_source_bridge._qualify_locked') as body,self.assertRaises(BlockingIOError):
                    qualify(tmp,'unused','unused',expected_allocation_sha='0'*64)
                body.assert_not_called()

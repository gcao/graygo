from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import torch
import unittest
from tools.task_actor_boundary import VERSION, SEMANTIC, fingerprint, verify_pair


def fixture(source):
    value = {key: key for key in SEMANTIC}
    value.update(version=VERSION, status='complete', source_sha256=source, archive_manifest_sha256='c'*64,
                 config={'task':'delayed','learner':'neural'}, source_and_archive_unchanged=True, global_rng_unchanged=True,
                 new_training_records=0, new_evaluation_games=0, execution_authorized=False,
                 runtime={'device':'cpu','threads':1},
                 counts={'observations':256,'raw_forwards':512,'scalar_commits':6400,'batched_commits':204,'forced_commits':36,'restored_continuations':150})
    return value


class ActorBoundaryTests(unittest.TestCase):
    def compare(self, old=None, new=None):
        return verify_pair(fixture('a'*64) if old is None else old, fixture('b'*64) if new is None else new,
                           reference_source='a'*64, candidate_source='b'*64, archive_manifest_sha='c'*64)

    def test_typed_transcript_is_stable_and_preserves_numerical_bits(self):
        self.assertEqual(fingerprint({'a':1,'b':2}), fingerprint({'b':2,'a':1}))
        values = [0, False, 0., -0., '0', [0], (0,), np.array([0],dtype=np.int32), np.array([0],dtype=np.int64),
                  torch.tensor([0.]), torch.tensor([[0.]]), torch.tensor([0.],dtype=torch.float64)]
        self.assertEqual(len({fingerprint(value) for value in values}), len(values))

    def test_transcript_retains_immutable_labels_dataclasses_and_infinity(self):
        @dataclass
        class Label:
            action:int
            greedy:np.ndarray
        a = np.array([True,False]); b = a.copy(); b.flags.writeable=False
        self.assertNotEqual(fingerprint(Label(0,a)), fingerprint(Label(0,b)))
        self.assertEqual(fingerprint(torch.tensor([-torch.inf])), fingerprint(torch.tensor([-torch.inf])))
        with self.assertRaises(ValueError):fingerprint(object())
        with self.assertRaises(ValueError):fingerprint({1:2})

    def test_equal_finite_boundary_grants_no_full_trajectory_or_reuse(self):
        result = self.compare()
        self.assertEqual(result['status'],'finite_actor_boundary_verified')
        for key in ('complete_training_trajectory_verified','source_compatibility_qualified','a2_evidence_reusable','execution_authorized'):
            self.assertFalse(result[key])

    def test_every_changed_semantic_digest_or_work_count_is_rejected(self):
        for key in SEMANTIC:
            if key in ('config','runtime','counts'):continue
            changed = fixture('b'*64); changed[key] = 'different'
            with self.subTest(key=key), self.assertRaises(ValueError):self.compare(new=changed)
        for field in fixture('b'*64)['counts']:
            changed=fixture('b'*64);changed['counts'][field]+=1
            with self.subTest(field=field),self.assertRaises(ValueError):self.compare(new=changed)

    def test_source_archive_mutation_scope_and_partial_partitions_fail(self):
        for key,value in [('status','running'),('source_sha256','x'*64),('archive_manifest_sha256','x'*64),
                          ('source_and_archive_unchanged',False),('global_rng_unchanged',False),('execution_authorized',True),
                          ('new_training_records',1),('new_evaluation_games',2)]:
            changed=fixture('b'*64);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.compare(new=changed)
        for mutate in (lambda r:r['counts'].pop('scalar_commits'),lambda r:r['counts'].update(observations=0),
                       lambda r:r['runtime'].update(device='cuda'),lambda r:r['runtime'].update(threads=2)):
            changed=fixture('b'*64);mutate(changed)
            with self.assertRaises(ValueError):self.compare(new=changed)

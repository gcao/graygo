from pathlib import Path
import subprocess
import unittest
from tools.retrace_observation_extension import prove


class ObservationExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo=Path(__file__).resolve().parents[1]
        base=cls.repo/'src/graygo'
        cls.candidate={str(p.relative_to(base)):p.read_bytes() for p in [*base.glob('*.py'),*(base/'native').glob('*.cpp')]}
        cls.reference={name:subprocess.check_output(['git','show','dc547f2:src/graygo/'+name],cwd=cls.repo) for name in cls.candidate}

    def test_only_checked_detached_observation_additions_extend_the_prior_source(self):
        result=prove(self.reference,self.candidate)
        self.assertTrue(result['default_target_and_update_AST_recovered'])
        self.assertTrue(result['all_other_application_bytes_identical'])
        self.assertFalse(result['source_admission_qualified']);self.assertFalse(result['runtime_prices_reused'])

    def test_default_update_sampling_or_observer_input_mutations_are_rejected(self):
        variants=[
            ('sequence_targets.py',b'raw.cpu().double().numpy()',b'(raw*.9).cpu().double().numpy()'),
            ('sequence_targets.py',b'if target_observer is not None:',b'if True:'),
            ('sequence_targets.py',b'raw += weight * residuals[:, index]',b'raw += .9 * weight * residuals[:, index]'),
            ('nfsp.py',b'self.counters.q_updates+=1',b'self.counters.q_updates+=2')]
        for name,before,after in variants:
            with self.subTest(file=name,mutation=after):
                changed=dict(self.candidate);self.assertIn(before,changed[name]);changed[name]=changed[name].replace(before,after)
                with self.assertRaises(ValueError):prove(self.reference,changed)


if __name__=='__main__':unittest.main()

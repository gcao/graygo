from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from tools.task_pipeline_parity import state_sha, omit_root, omit_recorded_timings, npz_sha, snapshot


class PipelineParityTests(unittest.TestCase):
    def test_optimizer_keys_and_all_parameter_bits_are_preserved(self):
        state={'state':{0:{'exp_avg':torch.tensor([.25]),'step':torch.tensor(1.)}},'groups':[{'params':[0]}]}
        same={'groups':[{'params':[0]}],'state':{0:{'step':torch.tensor(1.),'exp_avg':torch.tensor([.25])}}}
        self.assertEqual(state_sha(state),state_sha(same))
        different={'state':{'0':state['state'][0]},'groups':state['groups']}
        self.assertNotEqual(state_sha(state),state_sha(different))
        same['state'][0]['exp_avg'][0]+=.000001
        self.assertNotEqual(state_sha(state),state_sha(same))

    def test_only_named_root_clocks_are_excluded(self):
        a={'elapsed':1.,'timings':{'learning':2.},'active':{'elapsed':7.},'rng':[1,2]}
        b={'elapsed':99.,'timings':{'learning':30.},'active':{'elapsed':7.},'rng':[1,2]}
        self.assertEqual(state_sha(omit_root(a,('elapsed','timings'))),state_sha(omit_root(b,('elapsed','timings'))))
        b['active']['elapsed']=8.
        self.assertNotEqual(state_sha(omit_root(a,('elapsed','timings'))),state_sha(omit_root(b,('elapsed','timings'))))
        with self.assertRaises(ValueError):omit_root(a,('unknown',))

    def test_npz_container_encoding_is_separate_from_typed_array_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.npz',Path(tmp)/'b.npz'
            np.savez(a,x=np.array([1,2],dtype=np.int32),meta=np.array('schema'))
            np.savez_compressed(b,meta=np.array('schema'),x=np.array([1,2],dtype=np.int32))
            self.assertEqual(npz_sha(a),npz_sha(b))
            np.savez(b,x=np.array([1,2],dtype=np.int64),meta=np.array('schema'))
            self.assertNotEqual(npz_sha(a),npz_sha(b))

    def test_scheduled_evaluation_wall_clocks_are_explicit_but_metrics_and_other_nested_clocks_remain(self):
        a={'elapsed':1.,'timings':{},'evaluations':[{'records':32768,'elapsed_seconds':2.,'gap':.01}], 'active':{'elapsed_seconds':3.}}
        b={'elapsed':9.,'timings':{},'evaluations':[{'records':32768,'elapsed_seconds':8.,'gap':.01}], 'active':{'elapsed_seconds':3.}}
        normalized=lambda value:state_sha(omit_recorded_timings(value,('elapsed','timings')))
        self.assertEqual(normalized(a),normalized(b))
        b['evaluations'][0]['gap']=.02
        self.assertNotEqual(normalized(a),normalized(b))
        b['evaluations'][0]['gap']=.01;b['active']['elapsed_seconds']=4.
        self.assertNotEqual(normalized(a),normalized(b))
        b['evaluations'][0].pop('elapsed_seconds')
        with self.assertRaises(ValueError):normalized(b)

    def test_input_snapshot_detects_extra_files_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'a').write_bytes(b'a');before=snapshot(root)
            (root/'b').write_bytes(b'b');self.assertNotEqual(before,snapshot(root))
            (root/'link').symlink_to(root/'a')
            with self.assertRaises(ValueError):snapshot(root)

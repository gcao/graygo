"""Integration: a verified bundle restores real model/optimizer/RNG/environment state."""
import json
import random
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
from graygo.engine import State,resolve
from graygo.models import PolicyValueNet,ModelConfig
from graygo.storage import CheckpointStore


class ResumeIntegration(unittest.TestCase):
    def test_model_optimizer_rng_and_environment_restore(self):
        torch.set_num_threads(1);torch.manual_seed(12)
        model=PolicyValueNet(ModelConfig(8,1,2))
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        x=torch.zeros(1,30,3,3);g=torch.zeros(1,14)
        logits,value=model(x,g);loss=logits.square().mean()+value.square().mean()
        loss.backward();optimizer.step()
        py_rng=random.Random(22);np_rng=np.random.default_rng(33)
        state=resolve(State(),0,1).state
        before={k:v.clone() for k,v in model.state_dict().items()}
        with TemporaryDirectory() as root:
            store=CheckpointStore(root,reserve_bytes=0)
            def write(path):
                torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                            "torch_rng":torch.get_rng_state(),"python_rng":py_rng.getstate()},path/"training.pt")
                (path/"environment.json").write_text(state.canonical_json())
                (path/"numpy_rng.json").write_text(json.dumps(np_rng.bit_generator.state))
                np.savez(path/"buffer_example.npz",observations=np.arange(60,dtype=np.uint8),cursor=np.array(12),seen=np.array(99))
            bundle=store.save(512,write,metadata={"fresh_records":512,"encoding_version":"graygo/encoding/1"},estimated_peak_bytes=1_000_000)
            expected=(py_rng.random(),np_rng.random(4),torch.rand(4))
            for parameter in model.parameters():parameter.data.zero_()
            torch.rand(10);py_rng.random();np_rng.random(10)
            loaded=torch.load(bundle/"training.pt",weights_only=True)
            model.load_state_dict(loaded["model"]);optimizer.load_state_dict(loaded["optimizer"])
            torch.set_rng_state(loaded["torch_rng"]);py_rng.setstate(loaded["python_rng"])
            np_rng.bit_generator.state=json.loads((bundle/"numpy_rng.json").read_text())
            self.assertEqual(py_rng.random(),expected[0])
            np.testing.assert_array_equal(np_rng.random(4),expected[1])
            torch.testing.assert_close(torch.rand(4),expected[2],atol=0,rtol=0)
            for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
            self.assertTrue(optimizer.state_dict()["state"])
            self.assertEqual(State.from_dict(json.loads((bundle/"environment.json").read_text())),state)
            with np.load(bundle/"buffer_example.npz") as data:
                self.assertEqual(int(data["seen"]),99);self.assertEqual(int(data["cursor"]),12)
                np.testing.assert_array_equal(data["observations"],np.arange(60,dtype=np.uint8))

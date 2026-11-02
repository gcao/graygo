from dataclasses import replace
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
import numpy as np
from graygo.engine import BLACK,WHITE,Rules,State,resolve,transform
from graygo.native_decode import build_native,NativeRuleDecoder
from graygo.rule_state import RuleStateCodec,RuleStateSpec,RuleObservation


@unittest.skipIf(shutil.which('c++') is None,'C++17 compiler unavailable')
class NativeDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory();cls.build=build_native(Path(cls.directory.name)/'native')
        cls.library=cls.build['library']
    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def compare(self,rules,states):
        reference=RuleStateSpec(rules);native=RuleStateSpec(rules,self.library)
        rows=np.stack([np.frombuffer(reference.codec.pack(s,p),dtype=np.uint8) for s in states for p in (BLACK,WHITE)])
        empty_g=np.empty((len(rows),0),np.float32);empty_l=np.empty((len(rows),0),np.uint8)
        a=reference.unpack(rows,empty_g,empty_l);b=native.unpack(rows,empty_g,empty_l)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(reference.legal_masks(rows,empty_g,empty_l),native.legal_masks(rows,empty_g,empty_l))

    def test_generated_states_both_rules_sizes_perspectives_and_terminal_modes(self):
        rng=random.Random(93)
        for rules,rounds in ((Rules(),700),(Rules(9,200,'remove'),500),(Rules(3,10),150),(Rules(4,256),30),(Rules(13,8),80),(Rules(16,5),20),(Rules(19,4),40)):
            states=[];state=State(rules)
            for _ in range(rounds):
                states.append(state)
                state=resolve(state,rng.choice(state.legal_actions(BLACK)),rng.choice(state.legal_actions(WHITE))).state
                if state.done:states.append(state);state=State(rules)
            terminal=resolve(resolve(State(rules),rules.pass_action,rules.pass_action).state,rules.pass_action,rules.pass_action).state
            states.append(terminal);self.compare(rules,states)

    def test_reachable_collision_probes_and_all_terminal_leaves_match_reference(self):
        from graygo.probe_benchmark import ProbeBenchmark
        path=Path(__file__).parent/'fixtures/collision-benchmark-v1.json'
        benchmark=ProbeBenchmark.from_dict(json.loads(path.read_text()));states=[]
        for family in benchmark.families:
            for root in (family.before,family.after):
                states.append(root.state)
                for black in root.certificate.black_actions:
                    for white in root.certificate.white_actions:states.append(resolve(root.state,black,white).state)
        self.assertEqual(len(states),480);self.compare(Rules(),states)

    def test_transforms_history_restrictions_and_counter_width(self):
        rules=Rules(3,1000);state=State(rules)
        for b,w in ((1,7),(6,4),(2,5),(9,3)):state=resolve(state,b,w).state
        state=replace(state,forbidden_black=frozenset({0,8}),forbidden_white=frozenset({0}),rounds_played=512)
        states=[transform(state,rotation=r,reflect=f,swap=c,shift=(1,2)) for r in range(4) for f in (False,True) for c in (False,True)]
        self.compare(rules,states)
        wide_rules=Rules(3,2**32-1)
        wide=replace(state,rules=wide_rules,rounds_played=2**32-2)
        self.compare(wide_rules,[wide])

    def test_invalid_packets_match_reference_rejection_and_padding(self):
        rules=Rules();codec=RuleStateCodec(rules);native=NativeRuleDecoder(self.library)
        good=codec.pack(State(rules),BLACK);bad=[]
        for index,value in ((0,0),(3,255),(11,0),(12,3),(13,2),(14,201),(codec.header_bytes+codec.board_bytes,1)):
            row=bytearray(good);row[index]=value;bad.append(bytes(row))
        row=bytearray(good);row[codec.header_bytes+codec.board_bytes-1]=128;bad.append(bytes(row))
        row=bytearray(good);row[codec.header_bytes+3*codec.board_bytes+codec.mask_bytes-1]=128;bad.append(bytes(row))
        row=bytearray(good);row[codec.header_bytes:codec.header_bytes+codec.board_bytes]=b'\x55'*codec.board_bytes;row[codec.header_bytes+codec.board_bytes-1]=1;bad.append(bytes(row))
        for payload in bad:
            with self.assertRaises(ValueError):codec.unpack(payload)
            with self.assertRaises(ValueError):native.decode(codec,np.frombuffer(payload,dtype=np.uint8)[None])
        # Missing rows are explicitly masked; all-masked bytes are never a state.
        packets=np.stack([np.frombuffer(good,dtype=np.uint8),np.zeros(codec.packet_bytes,dtype=np.uint8)])
        x,g,l=native.decode(codec,packets,np.array([True,False]))
        self.assertFalse(x[1].any());self.assertFalse(g[1].any());self.assertFalse(l[1].any())
        with self.assertRaises(ValueError):native.decode(codec,packets)
        x,g,l=native.decode(codec,np.empty((0,codec.packet_bytes),dtype=np.uint8))
        self.assertEqual(x.shape,(0,30,9,9));self.assertEqual(g.shape,(0,14));self.assertEqual(l.shape,(0,82))

    def test_mutated_packets_have_identical_acceptance_and_observations(self):
        rules=Rules(3,12);codec=RuleStateCodec(rules);native=NativeRuleDecoder(self.library);rng=random.Random(7)
        state=resolve(resolve(State(rules),0,1).state,2,3).state;baseline=codec.pack(state,WHITE)
        from graygo.encoding import encode
        for _ in range(400):
            packet=bytearray(baseline);index=rng.randrange(len(packet));packet[index]^=1<<rng.randrange(8)
            try:
                s,p=codec.unpack(packet);expected=encode(s,p)
            except ValueError:
                with self.assertRaises(ValueError):native.decode(codec,np.frombuffer(packet,dtype=np.uint8)[None])
            else:
                x,g,l=native.decode(codec,np.frombuffer(packet,dtype=np.uint8)[None])
                np.testing.assert_array_equal(x[0],expected.spatial);np.testing.assert_array_equal(g[0],expected.global_features);np.testing.assert_array_equal(l[0],expected.legal)

    def test_build_reuse_and_descriptor_hash_enforcement(self):
        reused=build_native(Path(self.directory.name)/'native');self.assertTrue(reused['reused'])
        spec=RuleStateSpec(Rules(),self.library);self.assertEqual(RuleStateSpec.from_dict(spec.to_dict()),spec)
        bad=spec.to_dict();bad['native_decoder']['sha256']='0'*64
        with self.assertRaises(ValueError):RuleStateSpec.from_dict(bad)
        bad=spec.to_dict();bad['native_decoder']['sha256']=None
        with self.assertRaises(ValueError):RuleStateSpec.from_dict(bad)
        bad=RuleStateSpec().to_dict();bad['native_decoder']=None
        with self.assertRaises(ValueError):RuleStateSpec.from_dict(bad)
        with self.assertRaises(ValueError):RuleStateSpec(Rules(21,20),self.library)

    def test_real_neural_training_and_active_checkpoint_match_reference(self):
        import torch
        from graygo.task_training import NFSPTaskConfig,NFSPTaskTrainer
        config=NFSPTaskConfig(time_cap_seconds=300,task='gray-mixed',learner='neural',records=32,eval_every=16,
            batch=4,learning_start=4,capacity=32,target_every=8,reservoir_open=0,handoff_end=8,epsilon_decay=40,
            horizons=(5,),equivalent_actions=3,hidden=8,width=8,blocks=1,norm_groups=2,purpose='profile',storage='rule-state')
        with tempfile.TemporaryDirectory() as directory:
            a=NFSPTaskTrainer(config,Path(directory)/'reference',reserve_bytes=0);a.run(verbose=False)
            cfg=replace(config,native_library=self.library)
            b=NFSPTaskTrainer(cfg,Path(directory)/'native',reserve_bytes=0);b.run(until_records=2,verbose=False)
            restored=NFSPTaskTrainer(cfg,Path(directory)/'native',resume=True,reserve_bytes=0);restored.run(verbose=False)
            for name in ('q','target','average'):
                for key,data in getattr(a.backend,name).state_dict().items():torch.testing.assert_close(data,getattr(restored.backend,name).state_dict()[key],rtol=0,atol=0)
            for name,array in a.metrics.items():np.testing.assert_array_equal(array,restored.metrics[name])
            self.assertEqual((Path(directory)/'reference/episodes.bin').read_bytes(),(Path(directory)/'native/episodes.bin').read_bytes())
            for left,right in zip(a.evaluations,restored.evaluations):self.assertEqual(left['roots'],right['roots'])

if __name__=='__main__':unittest.main()

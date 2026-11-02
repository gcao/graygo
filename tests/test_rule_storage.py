from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import random
import tempfile
import unittest
import numpy as np
from graygo.engine import BLACK,WHITE,Rules,State,resolve,transform,score
from graygo.encoding import encode
from graygo.buffers import ObservationSpec,ReplayBuffer,ReservoirBuffer
from graygo.rule_state import RuleStateCodec,RuleObservation,RuleStateSpec


class RuleStorageTests(unittest.TestCase):
    def test_complete_state_roundtrip_and_feature_equality_across_rules(self):
        rng=random.Random(17)
        for rules in (Rules(),Rules(3,12),Rules(3,12,'remove'),Rules(13,4),Rules(19,3)):
            codec=RuleStateCodec(rules);spec=RuleStateSpec(rules);state=State(rules)
            for _ in range(35):
                for player in (BLACK,WHITE):
                    payload=codec.pack(state,player);restored,role=codec.unpack(payload)
                    self.assertEqual(restored,state);self.assertEqual(role,player)
                    self.assertEqual(codec.pack(restored,role),payload)
                    x,g,legal=spec.unpack(*spec.pack(RuleObservation(state,player)))
                    reference=encode(state,player)
                    np.testing.assert_array_equal(x,reference.spatial);np.testing.assert_array_equal(g,reference.global_features);np.testing.assert_array_equal(legal,reference.legal)
                if state.done:state=State(rules)
                state=resolve(state,rng.choice(state.legal_actions(BLACK)),rng.choice(state.legal_actions(WHITE))).state
            terminal=resolve(resolve(State(rules),rules.pass_action,rules.pass_action).state,rules.pass_action,rules.pass_action).state
            decoded,player=codec.unpack(codec.pack(terminal,WHITE))
            self.assertTrue(decoded.done);self.assertFalse(encode(decoded,player).legal.any())

    def test_history_restrictions_symmetry_and_wide_round_counters(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/mixed_subgame.json').read_text())
        state=State.from_dict(fixture['state'])
        state=replace(state,forbidden_black=frozenset({0,8}),forbidden_white=frozenset({0}))
        codec=RuleStateCodec(state.rules)
        for swap in (False,True):
            s=transform(state,shift=(1,2),rotation=1,reflect=True,swap=swap)
            for player in (BLACK,WHITE):self.assertEqual(codec.unpack(codec.pack(s,player)),(s,player))
        self.assertNotEqual(encode(state,BLACK).legal.tolist(),encode(state,WHITE).legal.tolist())
        rules=Rules(3,1000)
        wide=replace(State(rules),rounds_played=512,history=state.history)
        self.assertEqual(RuleStateCodec(rules).unpack(RuleStateCodec(rules).pack(wide,BLACK))[0],wide)

    def test_noncanonical_invalid_packets_and_wrong_rule_context_rejected(self):
        rules=Rules();codec=RuleStateCodec(rules);good=codec.pack(State(rules),BLACK)
        bad=[]
        for index,value in ((0,0),(11,0),(12,3),(13,2),(14,201)):
            packet=bytearray(good);packet[index]=value;bad.append(packet)
        packet=bytearray(good);packet[codec.header_bytes+codec.board_bytes]=1;bad.append(packet) # absent older history
        packet=bytearray(good);packet[codec.header_bytes+codec.board_bytes-1]=128;bad.append(packet) # padding
        packet=bytearray(good);packet[codec.header_bytes+3*codec.board_bytes+codec.mask_bytes-1]=128;bad.append(packet)
        packet=bytearray(good);packet[codec.header_bytes:codec.header_bytes+codec.board_bytes]=b'\x55'*codec.board_bytes;packet[codec.header_bytes+codec.board_bytes-1]=1;bad.append(packet) # dead full black board
        for packet in bad:
            with self.assertRaises(ValueError):codec.unpack(packet)
        with self.assertRaises(ValueError):codec.unpack(good[:-1])
        with self.assertRaises(ValueError):RuleStateCodec(Rules(9,200,'remove')).unpack(good)
        with self.assertRaises(ValueError):RuleStateCodec(Rules(9,201)).unpack(good)
        with self.assertRaises(ValueError):codec.pack(State(rules),True)
        with self.assertRaises(ValueError):RuleStateSpec(rules).pack(encode(State(rules),BLACK))

    def test_fast_color_normalization_keeps_numpy_support_and_rejects_invalid_values(self):
        from graygo.engine import RoundRecord
        rules=Rules(3,4)
        board=np.zeros(9,dtype=np.uint8);board[0]=1
        state=State(rules,board)
        self.assertTrue(all(type(v) is int for v in state.board))
        before=[0,np.int64(1),0,0,0,0,0,0,0]
        record=RoundRecord(before,np.int64(9),np.int64(2))
        self.assertTrue(all(type(v) is int for v in record.before))
        for invalid in (True,np.bool_(True),1.,'1',-1,4,None):
            bad=[0]*9;bad[4]=invalid
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaises(ValueError):State(rules,bad)
                with self.assertRaises(ValueError):RoundRecord(bad,0,0)
        # Validation of zero-liberty resolved states remains active.
        with self.assertRaises(ValueError):State(rules,[1]*9)

    def test_sequence_padding_is_explicit_and_terminal_state_is_retained(self):
        rules=Rules(3,3);spec=RuleStateSpec(rules);state=State(rules)
        row,g,l=spec.pack(RuleObservation(state,BLACK))
        packets=np.stack([row,np.zeros_like(row)]).reshape(1,2,-1)
        x,globals,legal=spec.unpack(packets,np.empty((1,2,0),np.float32),np.empty((1,2,0),np.uint8),valid=np.array([[True,False]]))
        self.assertEqual(x.shape,(1,2,30,3,3));self.assertFalse(x[0,1].any());self.assertFalse(legal[0,1].any());self.assertTrue(legal[0,0].all())
        with self.assertRaises(ValueError):spec.unpack(packets,np.empty((1,2,0),np.float32),np.empty((1,2,0),np.uint8))

    def collect(self,spec,records=80,capacity=32):
        rules=Rules(3,8);replay=ReplayBuffer(capacity,spec);reservoir=ReservoirBuffer(capacity,spec)
        moves=random.Random(6);labels=np.random.default_rng(9);state=State(rules);episode=0
        observation=RuleObservation if isinstance(spec,RuleStateSpec) else encode
        for record in range(0,records,2):
            actions=[moves.choice(state.legal_actions(p)) for p in (BLACK,WHITE)];outcome=resolve(state,*actions)
            for slot,player in enumerate((BLACK,WHITE)):
                obs,nxt=observation(state,player),observation(outcome.state,player)
                replay.add(obs,actions[slot],outcome.reward_black*(1 if player==BLACK else -1),nxt,outcome.state.done,
                    behavior_prob=1/len(state.legal_actions(player)),stream=2*episode+slot,step=state.rounds_played)
                reservoir.offer(obs,obs.legal,record=record+slot,policy_version=3,rng=labels,best_response_mode=True)
            state=outcome.state
            if state.done:state=State(rules);episode+=1
        return replay,reservoir

    def test_shared_replay_reservoir_sampling_and_restore_match_feature_storage(self):
        a,ar=self.collect(ObservationSpec.board(3));b,br=self.collect(RuleStateSpec(Rules(3,8)))
        for one,two in ((a.sample_sequences(32,12,np.random.default_rng(3)),b.sample_sequences(32,12,np.random.default_rng(3))),
                        (ar.sample(32,np.random.default_rng(4)),br.sample(32,np.random.default_rng(4)))):
            for key in one:np.testing.assert_array_equal(one[key],two[key])
        with tempfile.TemporaryDirectory() as d:
            b.save(Path(d)/'replay.npz');br.save(Path(d)/'reservoir.npz')
            restored=ReplayBuffer.load(Path(d)/'replay.npz');rr=ReservoirBuffer.load(Path(d)/'reservoir.npz')
            self.assertEqual(restored.spec,b.spec);self.assertEqual(rr.spec,br.spec)
            for one,two in ((b.sample(16,np.random.default_rng(5)),restored.sample(16,np.random.default_rng(5))),
                            (br.sample(16,np.random.default_rng(7)),rr.sample(16,np.random.default_rng(7)))):
                for key in one:np.testing.assert_array_equal(one[key],two[key])

    def test_default_logical_allocations_and_payload_size(self):
        spec=RuleStateSpec();self.assertEqual(spec.spatial_bytes,104)
        replay=ReplayBuffer(10,spec);reservoir=ReservoirBuffer(10,spec)
        self.assertEqual(replay.allocated_bytes,2750);self.assertEqual(reservoir.allocated_bytes,1310)
        self.assertEqual((replay.allocated_bytes+reservoir.allocated_bytes)*50000,203000000)
        self.assertEqual(replay.arrays['globals'].shape,(10,0));self.assertEqual(reservoir.arrays['legal'].shape,(10,0))

    def test_actual_neural_updates_are_identical_after_storage_regeneration(self):
        import torch
        from graygo.models import QNet,PolicyNet,ModelConfig
        from graygo.nfsp import NFSPBackend,NFSPConfig
        torch.set_num_threads(1);torch.manual_seed(4)
        q=QNet(ModelConfig(8,1,2));p=PolicyNet(ModelConfig(8,1,2));c=NFSPConfig(batch=4,learning_start=4,target_every=8)
        a=NFSPBackend(deepcopy(q),deepcopy(p),c);b=NFSPBackend(deepcopy(q),deepcopy(p),c)
        r1,s1=self.collect(ObservationSpec.board(3),records=4);r2,s2=self.collect(RuleStateSpec(Rules(3,8)),records=4)
        stats1=a.learn_after_fresh(4,4,r1,s1,np.random.default_rng(10),epsilon=.1)
        stats2=b.learn_after_fresh(4,4,r2,s2,np.random.default_rng(10),epsilon=.1)
        self.assertEqual(stats1,stats2)
        for name in ('q','target','average'):
            for key,value in getattr(a,name).state_dict().items():torch.testing.assert_close(value,getattr(b,name).state_dict()[key],rtol=0,atol=0)

if __name__=='__main__':unittest.main()

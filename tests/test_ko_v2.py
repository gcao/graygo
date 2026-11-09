import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from graygo.engine import State,Rules,RoundRecord,resolve,resolve_reference,transform,BLACK,WHITE
from graygo.encoding import encode
from graygo.rule_state import RuleStateCodec,RuleStateSpec,RuleObservation
from graygo.native_decode import build_native
from graygo.board_runtime import BoardRuntimeConfig,BoardNFSPRuntime
from graygo.board_execution import BoardExecutor,BoardExecutionPlan
from graygo.board_journal import BoardJournalPlan
from test_learning_loop import exact,normalized


def fixture():return json.loads((Path(__file__).parent/'fixtures/ko-v2-four-round.json').read_text())
def board(key):return tuple('+BWG'.index(c) for c in key.replace('/',''))
def action(label):return 81 if label=='PASS' else (ord(label[0])-97)*9+int(label[1:])-1


class KoV2Tests(unittest.TestCase):
    def test_recorded_four_round_loop_is_rejected_without_committing_captures(self):
        f=fixture();s=State(Rules(ko='full-history'),board(f['board_key']))
        for i,pair in enumerate(f['moves']):
            before=s;t=resolve(s,action(pair['black']),action(pair['white']));s=t.state
            self.assertEqual(t.repetition_rejected,i==3)
            if i<3:self.assertEqual(s.board,board(pair['legacy_post_board_key']))
        self.assertEqual(s.board,before.board);self.assertFalse(t.board_changed)
        self.assertEqual(t.captured,());self.assertEqual(t.suicidal,());self.assertIsNone(t.collision)
        self.assertIn(action('i1'),s.forbidden_black);self.assertIn(action('f3'),s.forbidden_white)
        self.assertEqual(s.rounds_played,4);self.assertEqual(s.pass_streak,0)
        self.assertEqual(State.from_dict(s.to_dict()),s)
        with self.assertRaises(ValueError):resolve(s,action('i1'),81)

    def test_cycle_with_interleaved_passes_and_true_pass_termination(self):
        f=fixture();s=State(Rules(ko='full-history'),board(f['board_key']))
        for i,pair in enumerate(f['moves']):
            if i:s=resolve(s,81,81).state
            t=resolve(s,action(pair['black']),action(pair['white']));s=t.state
        self.assertTrue(t.repetition_rejected);self.assertEqual(s.rounds_played,7);self.assertEqual(s.pass_streak,0)
        fb=s.forbidden_black;fw=s.forbidden_white
        s=resolve(s,81,81).state;self.assertFalse(s.done);self.assertEqual((s.forbidden_black,s.forbidden_white),(fb,fw))
        s=resolve(s,81,81).state;self.assertTrue(s.done);self.assertEqual(s.terminal_reason,'mutual_pass')

    def test_rejected_points_accumulate_per_player_and_reset_pass_streak(self):
        values=[0]*81
        for p in [1,9,11,19,31,39,41,49]:values[p]=WHITE
        s=State(Rules(ko='full-history'),tuple(values));s=resolve(s,81,81).state
        t=resolve(s,40,81);s=t.state;self.assertTrue(t.repetition_rejected);self.assertEqual(s.pass_streak,0)
        self.assertEqual(s.forbidden_black,frozenset({40}));self.assertEqual(s.forbidden_white,frozenset())
        s=resolve(s,81,81).state;t=resolve(s,10,10);s=t.state
        self.assertTrue(t.repetition_rejected);self.assertEqual(s.forbidden_black,frozenset({40,10}));self.assertEqual(s.forbidden_white,frozenset({10}))
        s=resolve(s,80,81).state;self.assertEqual((s.forbidden_black,s.forbidden_white),(frozenset(),frozenset()))

    def test_full_history_transforms_and_observation_packets_cannot_simulate(self):
        f=fixture();s=State(Rules(ko='full-history'),board(f['board_key']))
        for pair in f['moves']:s=resolve(s,action(pair['black']),action(pair['white'])).state
        altered=transform(s,shift=(2,3),rotation=1,reflect=True,swap=True)
        self.assertEqual(len(altered.seen_boards),len(s.seen_boards));self.assertIn(altered.board,altered.seen_boards)
        codec=RuleStateCodec(s.rules);packet=codec.pack(s,BLACK);view,player=codec.unpack(packet)
        self.assertIsNone(view.seen_boards);self.assertEqual(player,BLACK)
        for field in ('spatial','global_features','legal'):np.testing.assert_array_equal(getattr(encode(view,player),field),getattr(encode(s,player),field))
        with self.assertRaisesRegex(ValueError,'complete ko history'):resolve(view,81,81)
        with self.assertRaises(ValueError):RuleStateCodec(Rules()).unpack(packet)
        self.assertEqual(RuleStateSpec.from_dict(RuleStateSpec(s.rules).to_dict()),RuleStateSpec(s.rules))

    def test_native_decoder_keeps_existing_network_features_under_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            built=build_native(directory);rules=Rules(ko='full-history');spec=RuleStateSpec(rules,built['library']);f=fixture();s=State(rules,board(f['board_key']));states=[s]
            for pair in f['moves']:s=resolve(s,action(pair['black']),action(pair['white'])).state;states.append(s)
            rows=np.stack([np.frombuffer(spec.codec.pack(s,p),dtype=np.uint8) for s in states for p in (BLACK,WHITE)])
            x,g,legal=spec.unpack(rows,np.empty((len(rows),0),np.float32),np.empty((len(rows),0),np.uint8))
            for i,(s,p) in enumerate((s,p) for s in states for p in (BLACK,WHITE)):
                expected=encode(s,p);np.testing.assert_array_equal(x[i],expected.spatial);np.testing.assert_array_equal(g[i],expected.global_features);np.testing.assert_array_equal(legal[i],expected.legal)

    def test_v2_warm_start_and_checkpoint_resume_keep_weights_and_full_history(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=BoardRuntimeConfig(rules=Rules(3,5),envs=2,capacity=64,batch=4,learning_start=4,target_every=8,width=8,blocks=1,norm_groups=2)
            with BoardExecutor(root/'v1',base,BoardJournalPlan('diagnostic',8,(0,8)),BoardExecutionPlan(60),reserve_bytes=0) as executor:executor.run()
            archive=root/'v1/weights/000000000008';saved=torch.load(archive/'models.pt',map_location='cpu',weights_only=True)
            cfg=replace(base,rules=Rules(3,5,ko='full-history'),initial_weights_archive=str(archive),reservoir_open=0,handoff_end=2)
            from graygo.evaluation import FrozenNeuralPolicy
            transferred=FrozenNeuralPolicy.from_archive(archive,evaluation_rules=cfg.rules)
            self.assertEqual(transferred.identity['origin']['rules'],cfg.rules.to_dict())
            self.assertEqual(transferred.identity['origin']['training_rules'],base.rules.to_dict())
            self.assertEqual(transferred.identity['origin']['transfer'],'weights-only')
            transferred.assert_frozen()
            with self.assertRaises(ValueError):FrozenNeuralPolicy.from_archive(archive,evaluation_rules=Rules(5,5,ko='full-history'))
            runtime=BoardNFSPRuntime(cfg)
            for name,model in [('q',runtime.backend.q),('average',runtime.backend.average)]:
                for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,saved[name][key]))
            self.assertEqual(runtime.records,0);self.assertEqual(runtime.replay.size,0);self.assertTrue(runtime.backend.average_ready)
            runtime.run_profile(4,time_cap_seconds=60);checkpoint=runtime.save(root/'checkpoint',reserve_bytes=0)
            restored=BoardNFSPRuntime(cfg);restored.restore(checkpoint)
            exact(normalized(restored.collector.state_dict()),normalized(runtime.collector.state_dict()))
            runtime.run_profile(8,time_cap_seconds=60);restored.run_profile(8,time_cap_seconds=60)
            for key,value in runtime.backend.q.state_dict().items():self.assertTrue(torch.equal(value,restored.backend.q.state_dict()[key]))
            with self.assertRaises(ValueError):BoardNFSPRuntime(base).restore(checkpoint)


    def test_rejected_final_round_is_scored_on_the_unchanged_board(self):
        f=fixture();s=State(Rules(9,4,ko='full-history'),board(f['board_key']))
        for pair in f['moves']:
            before=s;t=resolve(s,action(pair['black']),action(pair['white']));s=t.state
        self.assertTrue(t.repetition_rejected);self.assertEqual(s.board,before.board)
        self.assertTrue(s.done);self.assertEqual(s.terminal_reason,'round_limit')
        from graygo.engine import score
        self.assertEqual(t.reward_black,score(before.board,9).utility_black)

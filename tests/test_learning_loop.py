from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from graygo.board_execution import BoardExecutor,BoardExecutionPlan
from graygo.board_journal import BoardJournalPlan
from graygo.board_runtime import BoardRuntimeConfig
from graygo.confidence import canonical_sha
from graygo.engine import Rules
from tools.champion_gate import GatePlan,IndependentGate
from tools.champion_loop import ChampionLoop,LoopPlan,parse_args
from tools.learning_loop import compatible_sources,initialize,prepare,source_files


def normalized(value):
    if isinstance(value,dict):return {k:normalized(v) for k,v in value.items() if k!='timings'}
    return value


def exact(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:exact(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


class LearningLoopTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_resume_can_mix_restored_and_new_episodes_before_the_next_optimizer_update(self):
        cfg=BoardRuntimeConfig(rules=Rules(3,5),envs=2,capacity=256,batch=8,learning_start=8,
                              target_every=16,width=8,blocks=1,norm_groups=2)
        journal=BoardJournalPlan('loop',32,(0,2,32));execution=BoardExecutionPlan(120)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            with BoardExecutor(base/'full',cfg,journal,execution,reserve_bytes=0) as learner:
                learner.run(2);learner.run()
                expected=deepcopy(learner.runtime.state_dict())
            with BoardExecutor(base/'resumed',cfg,journal,execution,reserve_bytes=0) as learner:
                learner.run(2)
                self.assertIsNotNone(learner.runtime.collector.active[0])
                self.assertIsNone(learner.runtime.collector.active[1])
                self.assertEqual(learner.runtime.backend.counters.q_updates,0)
            with BoardExecutor(base/'resumed',cfg,journal,execution,reserve_bytes=0,resume=True) as learner:
                learner.run()
                exact(normalized(expected),normalized(learner.runtime.state_dict()))

    def test_handoff_preserves_the_complete_learner_and_continues_publishing(self):
        checkout=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);parent=base/'parent';child=base/'child';publication=base/'champion.json'
            cfg=BoardRuntimeConfig(rules=Rules(3,5),envs=2,capacity=256,batch=4,learning_start=4,
                                  target_every=8,width=8,blocks=1,norm_groups=2)
            with BoardExecutor(parent,cfg,BoardJournalPlan('practice',16,(0,16)),BoardExecutionPlan(120),reserve_bytes=0) as learner:
                with ChampionLoop(learner,LoopPlan((16,)),IndependentGate(GatePlan(1),reserve_bytes=0),publish_manifest=publication) as loop:
                    loop.run();expected=deepcopy(learner.runtime.state_dict())
                    replay={k:v.copy() for k,v in learner.runtime.replay.arrays.items()}
            parent_seal=(parent/'resume/000000000016/manifest.json').read_bytes()
            body=prepare(child,parent,checkout,publication,total_seconds=120,records=32,block=16,reserve_bytes=0)
            self.assertLess(body['remaining_seconds'],120)
            with self.assertRaisesRegex(ValueError,'already been handed off'):
                prepare(base/'second-child',parent,checkout,publication,total_seconds=120,records=32,block=16,reserve_bytes=0)
            with BoardExecutor(child/'learner',cfg,BoardJournalPlan('loop',32,(0,16,32)),
                BoardExecutionPlan(body['remaining_seconds']),reserve_bytes=0,initialization=body['initialization']) as learner:
                with ChampionLoop(learner,LoopPlan((32,)),IndependentGate(GatePlan(1),reserve_bytes=0),
                    initial_champion=body['initial_champion'],publish_manifest=publication) as loop:
                    initialize(learner,loop,body)
                    exact(normalized(expected),normalized(learner.runtime.state_dict()))
                    for key,value in replay.items():np.testing.assert_array_equal(value,learner.runtime.replay.arrays[key])
                    self.assertEqual(loop.state['completed_records'],16)
                    result=loop.run();self.assertEqual(result['learner_records'],32)
                    self.assertGreater(learner.runtime.backend.counters.q_updates,expected['backend']['counters']['q_updates'])
                    self.assertEqual(learner.runtime.reservoir.size,0)
                    published=json.loads(publication.read_text())['body']
                    self.assertEqual(published['loop_contract_sha256'],canonical_sha(loop.contract))
            self.assertEqual(parent_seal,(parent/'resume/000000000016/manifest.json').read_bytes())

    def test_numerical_source_changes_cannot_be_hidden_as_a_controller_handoff(self):
        checkout=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory)
            for name,path in source_files(checkout).items():
                target=parent/'src/graygo'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
            compatible_sources(parent,checkout)
            with (parent/'src/graygo/models.py').open('a') as stream:stream.write('\n# different model implementation\n')
            with self.assertRaisesRegex(ValueError,'numerical learner source changed'):compatible_sources(parent,checkout)

    def test_loop_mode_is_bounded_and_keeps_primary_averaging_schedule(self):
        args=parse_args(['--loop','--run-dir','unused','--time-cap-seconds','120'])
        self.assertEqual((args.records,args.reservoir_open,args.handoff_end),(20000000,5000000,7000000))
        with self.assertRaisesRegex(ValueError,'early-opening-diagnostic'):
            parse_args(['--loop','--run-dir','unused','--time-cap-seconds','120','--reservoir-open','0','--handoff-end','1000'])
        with self.assertRaisesRegex(ValueError,'twenty million'):BoardJournalPlan('loop',20000002,(0,20000002))

    def test_bad_source_admission_is_rejected_by_ordinary_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=BoardRuntimeConfig(rules=Rules(3,5),envs=2,capacity=256,batch=4,learning_start=4,target_every=8,width=8,blocks=1,norm_groups=2)
            with BoardExecutor(directory,cfg,BoardJournalPlan('loop',16,(0,16)),BoardExecutionPlan(120),reserve_bytes=0) as learner:
                with self.assertRaisesRegex(ValueError,'fingerprint'):
                    learner.runtime.restore(Path(directory)/'resume/000000000000',compatible_source_sha256='unverified')


if __name__=='__main__':unittest.main()

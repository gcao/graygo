from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from graygo.engine import BLACK,WHITE
from graygo.task_training import NFSPTaskConfig,NFSPTaskTrainer,HEADER_DTYPE
from graygo.task_adapter import TaskAdapter
from graygo.task_evaluation import evaluate_task
from graygo.nfsp import NFSPBackend


class TaskTrainerTests(unittest.TestCase):
    def config(self,*,task='delayed',learner='tabular',records=80):
        return NFSPTaskConfig(time_cap_seconds=300,task=task,learner=learner,seed=3,records=records,eval_every=16,
            batch=4,learning_start=4,capacity=64,target_every=8,reservoir_open=0,handoff_end=8,
            epsilon_decay=40,horizons=(5,),equivalent_actions=3,hidden=8,width=8,blocks=1,norm_groups=2,purpose='profile')

    def test_mid_episode_complete_resume_preserves_old_frozen_actor(self):
        for task,kind,stop in (('delayed','tabular',6),('delayed','neural',6),('gray-mixed','neural',2),('gray-mixed','tabular',2)):
            with self.subTest(task=task,kind=kind),tempfile.TemporaryDirectory() as d:
                config=self.config(task=task,learner=kind,records=32 if task=='gray-mixed' else 80)
                full=NFSPTaskTrainer(config,Path(d)/'full',reserve_bytes=0);a=full.run(verbose=False)
                part=NFSPTaskTrainer(config,Path(d)/'part',reserve_bytes=0);part.run(until_records=stop,verbose=False)
                self.assertIsNotNone(part.active)
                if task=='delayed':self.assertLess(part.active.actors[BLACK].snapshot.version,part.backend.counters.q_updates+part.backend.counters.supervised_updates)
                resumed=NFSPTaskTrainer(config,Path(d)/'part',resume=True,reserve_bytes=0);b=resumed.run(verbose=False)
                for name in ('q','target','average'):
                    for key,value in getattr(full.backend,name).state_dict().items():
                        torch.testing.assert_close(value,getattr(resumed.backend,name).state_dict()[key],rtol=0,atol=0)
                for name,array in full.replay.arrays.items():np.testing.assert_array_equal(array,resumed.replay.arrays[name])
                for name,array in full.reservoir.arrays.items():np.testing.assert_array_equal(array,resumed.reservoir.arrays[name])
                for name,array in full.metrics.items():np.testing.assert_array_equal(array,resumed.metrics[name])
                for name,rng in full.rng.items():self.assertEqual(rng.bit_generator.state,resumed.rng[name].bit_generator.state)
                self.assertEqual((Path(d)/'full/episodes.bin').read_bytes(),(Path(d)/'part/episodes.bin').read_bytes())
                self.assertEqual(a['counters'],b['counters']);self.assertEqual(a['acting_policy_counts'],b['acting_policy_counts'])
                for x,y in zip(a['evaluations'],b['evaluations']):self.assertEqual(x['roots'],y['roots'])
                self.assertFalse(a['formal_protocol']);self.assertFalse(a['formal_case_passed']);self.assertFalse(a['a2_complete'])

    def test_analytic_chain_values_detect_unlearned_avoided_root_action(self):
        config=replace(self.config(),horizons=(3,5),equivalent_actions=3)
        adapter=TaskAdapter(config);q,average=adapter.models();backend=NFSPBackend(q,average,config.backend_config())
        epsilon=.05
        with torch.no_grad():
            for state in adapter.states:
                for player in (BLACK,WHITE):
                    token=int(adapter.game.observation(state,player).spatial.argmax())
                    if state.depth==0:
                        if player==state.controller:q.values[token,0]=1;q.values[token,1]=-1
                        else:q.values[token,0]=-(1-epsilon)
                    else:q.values[token,:]=(1 if state.branch==0 else -1)*(1 if player==state.controller else -1)
                    average.probabilities[token]=1/3
                    if state.depth==0 and player==state.controller:average.probabilities[token]=torch.tensor([.95,.05,0.],dtype=torch.float64)
        result=evaluate_task(adapter,backend,epsilon,batch_size=7)
        for row in result:
            self.assertLess(row['max_early_error'],1e-6);self.assertLess(row['max_late_error'],1e-6)
            self.assertAlmostEqual(max(row['average_gaps']),.1);self.assertEqual(max(row['greedy_q_gaps']),0.)
        root=adapter.roots[0];token=int(adapter.game.observation(root,root.controller).spatial.argmax())
        with torch.no_grad():q.values[token,1]=.4
        bad=evaluate_task(adapter,backend,epsilon)[0]
        self.assertAlmostEqual(bad['max_early_error'],1.4,places=6)

    def test_delayed_credit_is_separate_from_weak_average_opponent(self):
        config=replace(self.config(),horizons=(3,),equivalent_actions=3)
        adapter=TaskAdapter(config);q,average=adapter.models();backend=NFSPBackend(q,average,config.backend_config())
        backend.counters.fresh_q=8;backend.counters.supervised_updates=1
        epsilon=.05;native_good=.1*(1-epsilon/2)+.9*.5
        with torch.no_grad():
            for state in adapter.states:
                for player in (BLACK,WHITE):
                    token=int(adapter.game.observation(state,player).spatial.argmax())
                    if state.depth==0:
                        if player==state.controller:q.values[token,0]=1;q.values[token,1]=-1
                        else:q.values[token,0]=-(2*native_good-1)
                    else:q.values[token,:]=(1 if state.branch==0 else -1)*(1 if player==state.controller else -1)
        rows=evaluate_task(adapter,backend,epsilon)
        for row in rows:
            self.assertLess(row['max_early_error'],1e-6)
            self.assertEqual(max(row['greedy_q_gaps']),0.)
            self.assertAlmostEqual(max(row['average_gaps']),1.)
            np.testing.assert_allclose(row['native_opponent_weights'],[.1,.9,0],atol=1e-15,rtol=0)

    def test_native_opponent_belief_conditions_on_revealed_branch(self):
        from graygo.task_evaluation import _native_q_values
        config=self.config(task='branching');adapter=TaskAdapter(config);game=adapter.game
        shape=(len(adapter.states),2,2);target=np.zeros(shape);average=np.zeros(shape)
        for i,state in enumerate(adapter.states):
            for role in (BLACK,WHITE):
                acts=list(game.actions(state,role));target[i,role-1,acts]=1/len(acts);average[i,role-1,acts]=1/len(acts)
        root=game.root(WHITE);later=game.transition(root,0,1)
        target[adapter.index[root],WHITE-1]=[.95,.05]
        average[adapter.index[root],WHITE-1]=[.05,.95]
        target[adapter.index[later],WHITE-1]=[.95,.05]
        average[adapter.index[later],WHITE-1]=[0.,1.]
        result=_native_q_values(adapter,root,BLACK,target,(target,average),[.1,.9])
        posterior=.1*.05/(.1*.05+.9*.95)
        self.assertAlmostEqual(result[later]['belief'][0],posterior)
        # Black action 0 receives -.5 vs white 0 and +1 vs white 1.
        expected=1-1.5*(posterior*.95)
        self.assertAlmostEqual(result[later]['q'][0],expected)
        self.assertGreater(abs(expected-(1-1.5*(.1*.95))),.1)

    def test_branch_values_and_bounded_gray_roots_are_actual_games(self):
        c=self.config(task='branching');adapter=TaskAdapter(c);q,avg=adapter.models();backend=NFSPBackend(q,avg,c.backend_config())
        rows=evaluate_task(adapter,backend,.1)
        self.assertAlmostEqual(rows[0]['value'],-1/7);self.assertAlmostEqual(rows[1]['value'],1/7)
        black_actions=[x['expected'] for x in rows[0]['root_action_values'] if x['player']==BLACK]
        np.testing.assert_allclose(black_actions,[-.125,-.3125],atol=1e-12,rtol=0)
        for task,value in (('gray-pure',0.),('gray-mixed',-.25)):
            a=TaskAdapter(self.config(task=task));self.assertAlmostEqual(a.values[0],value)
            self.assertLess(a.oracle['matrix_certificate_gap'],1e-7)
            if task=='gray-mixed':
                fixture=__import__('json').loads((Path(__file__).parent/'fixtures/mixed_subgame.json').read_text())
                self.assertEqual(a.roots[0].to_dict(),fixture['state'])

    def test_full_supported_coverage_counts_and_replay_overlap(self):
        config=replace(self.config(records=120),horizons=(3,),eval_every=24,q_reuse=4)
        with tempfile.TemporaryDirectory() as d:
            trainer=NFSPTaskTrainer(config,d,reserve_bytes=0);result=trainer.run(verbose=False)
            self.assertEqual(result['completed_episodes'],20);self.assertEqual(result['counters']['fresh_q'],120)
            self.assertEqual(result['counters']['fresh_ordinary'],96);self.assertEqual(result['counters']['target_refreshes'],15)
            self.assertEqual(result['per_role_records'],[60,60]);self.assertEqual(int(trainer.metrics['visits'][...,1].sum()),24)
            self.assertGreater(int(trainer.metrics['replay_starts'][...,1].sum()),0)
            self.assertEqual(int(trainer.metrics['replay_starts'].sum()),result['counters']['q_loss_targets'])
            self.assertEqual(int(trainer.metrics['overlap_counts'].sum()),result['counters']['q_loss_targets'])
            rows=np.fromfile(Path(d)/'episodes.bin',dtype=HEADER_DTYPE)
            self.assertEqual(int(rows['supplemental'].sum()),4)
            self.assertEqual(set(rows['forced_action'][rows['supplemental']]),{0,1})
            self.assertTrue(np.isfinite(trainer.metrics['overlap_sums']).all())
            # Reservoir records are ordinary best-response visits even when Q fallback acted elsewhere.
            self.assertLessEqual(trainer.reservoir.seen,sum(trainer.best_response_visits))

    def test_budget_endpoint_preserves_continuing_game_and_restore(self):
        config=self.config(records=82)
        with tempfile.TemporaryDirectory() as d:
            t=NFSPTaskTrainer(config,d,reserve_bytes=0);r=t.run(verbose=False)
            self.assertEqual(r['status'],'complete');self.assertIsNotNone(t.active);self.assertFalse(t.adapter.game.terminal(t.active.state))
            last=(t.replay.cursor-1)%t.replay.capacity
            self.assertFalse(t.replay.arrays['terminated'][last])
            restored=NFSPTaskTrainer(config,d,resume=True,reserve_bytes=0)
            self.assertEqual(restored.active.state,t.active.state);self.assertEqual(restored.active.log,t.active.log)

    def test_overlap_uses_conditional_logged_behavior_not_mode_probability(self):
        from graygo.nfsp import OneStepBatch
        config=replace(self.config(),horizons=(3,),equivalent_actions=2)
        with tempfile.TemporaryDirectory() as d:
            t=NFSPTaskTrainer(config,d,reserve_bytes=0);ordinary=t.collect_round()
            t.backend.counters.advance(2,ordinary,t.backend.config)
            controller=t.active.state.controller
            data=t.replay.sample(32,np.random.default_rng(2))
            selected=(data['stream']%2)==controller-1
            data={key:value[selected] for key,value in data.items()}
            batch=OneStepBatch.from_replay(data,t.device)
            predictions=torch.zeros_like(batch.legal,dtype=torch.float32)
            predictions.scatter_(1,batch.actions[:,None],-.9)
            other=1-batch.actions
            predictions.scatter_(1,other[:,None],.9)
            t._observe_q(data,predictions,batch,.1)
            count=int(t.metrics['overlap_counts'].sum())
            sums=t.metrics['overlap_sums'].sum(axis=(0,1,2))
            expected=np.minimum(1.,.05/data['behavior_prob'].astype(float))
            self.assertAlmostEqual(sums[0],float(expected.sum()),places=6)
            self.assertAlmostEqual(sums[1],.05*count,places=6)
            self.assertAlmostEqual(sums[2],float(data['behavior_prob'].sum()),places=6)

    def test_unfunded_full_plan_and_changed_resume_plan_fail(self):
        with self.assertRaises(ValueError):NFSPTaskConfig(time_cap_seconds=3600)
        self.assertTrue(NFSPTaskConfig(time_cap_seconds=3600,budget_profile_sha256='0'*64).formal_protocol())
        self.assertFalse(replace(self.config(),q_reuse=4).formal_protocol())
        with tempfile.TemporaryDirectory() as d:
            config=self.config();trainer=NFSPTaskTrainer(config,d,reserve_bytes=0);trainer.run(until_records=6,verbose=False)
            with self.assertRaises(ValueError):NFSPTaskTrainer(replace(config,horizons=(6,)),d,resume=True,reserve_bytes=0)

if __name__=='__main__':unittest.main()

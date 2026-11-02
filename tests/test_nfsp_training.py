from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from graygo.nfsp_training import NFSPMatrixConfig,NFSPMatrixTrainer,draw_actions
from graygo.storage import CheckpointStore


class TrainerTests(unittest.TestCase):
    def config(self,learner='neural'):
        return NFSPMatrixConfig(learner=learner,seed=2,records=320,eval_every=64,batch=16,learning_start=32,
               capacity=128,target_every=64,hidden=8,reservoir_open=32,handoff_end=96,epsilon_decay=160)

    def test_exact_resume_matches_uninterrupted_both_learners(self):
        for kind in ('neural','tabular'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                config=self.config(kind);a=NFSPMatrixTrainer(config,Path(d)/'a',reserve_bytes=0)
                result=a.run(verbose=False)
                b=NFSPMatrixTrainer(config,Path(d)/'b',reserve_bytes=0)
                paused=b.run(until_records=118,verbose=False)
                self.assertEqual(paused['status'],'segment_paused')
                resumed=NFSPMatrixTrainer(config,Path(d)/'b',resume=True,reserve_bytes=0)
                other=resumed.run(verbose=False)
                for model in ('q','average','target'):
                    for name,value in getattr(a.backend,model).state_dict().items():
                        torch.testing.assert_close(value,getattr(resumed.backend,model).state_dict()[name],rtol=0,atol=0)
                self.assertEqual(result['counters'],other['counters']);self.assertEqual(result['acting_policy_counts'],other['acting_policy_counts'])
                self.assertEqual(result['reservoir_seen'],other['reservoir_seen']);self.assertEqual(result['best_response_visits'],other['best_response_visits'])
                self.assertEqual(result['per_role_records'],[160,160]);self.assertEqual(result['physical_games'],160)
                self.assertEqual([e['records'] for e in result['evaluations']],[64,128,192,256,320])
                for x,y in zip(result['evaluations'],other['evaluations']):
                    for field in ('average_gaps','greedy_q_gaps','average_policies','q_values'):self.assertEqual(x[field],y[field])
                for name,array in a.replay.arrays.items():np.testing.assert_array_equal(array,resumed.replay.arrays[name])
                for name,array in a.reservoir.arrays.items():np.testing.assert_array_equal(array,resumed.reservoir.arrays[name])
                for name in a.rng:self.assertEqual(a.rng[name].bit_generator.state,resumed.rng[name].bit_generator.state)
                self.assertEqual((Path(d)/'a/episodes.bin').read_bytes(),(Path(d)/'b/episodes.bin').read_bytes())
                self.assertFalse(result['formal_protocol']);self.assertFalse(result['formal_gate_passed'])
                self.assertEqual(len([p for p in (Path(d)/'a/resume').iterdir() if p.is_dir()]),1)

    def test_audit_tail_recovery_and_committed_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            config=self.config();t=NFSPMatrixTrainer(config,d,reserve_bytes=0)
            t.run(until_records=118,verbose=False)
            path=Path(d)/'episodes.bin';expected=path.read_bytes()
            with path.open('ab') as file:file.write(b'uncommitted crash tail')
            restored=NFSPMatrixTrainer(config,d,resume=True,reserve_bytes=0)
            self.assertEqual(path.read_bytes(),expected)
            corrupt=bytearray(expected);corrupt[0]^=1;path.write_bytes(corrupt)
            with self.assertRaises(ValueError):NFSPMatrixTrainer(config,d,resume=True,reserve_bytes=0)

    def test_no_labels_before_opening_and_mode_counts_are_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            config=replace(self.config(),reservoir_open=1000,handoff_end=2000)
            t=NFSPMatrixTrainer(config,d,reserve_bytes=0);result=t.run(verbose=False)
            self.assertEqual(result['reservoir_seen'],0);self.assertEqual(result['counters']['supervised_updates'],0)
            self.assertGreater(sum(result['best_response_visits']),0)
            self.assertEqual(result['acting_policy_counts'],{'q':320,'average':0,'uniform':0})
            self.assertFalse(result['average_final_five_passed'])
            self.assertEqual(result['counters']['target_refreshes'],5)

    def test_exact_table_published_policy_equals_final_reservoir_labels(self):
        with tempfile.TemporaryDirectory() as d:
            t=NFSPMatrixTrainer(self.config('tabular'),d,reserve_bytes=0);t.run(verbose=False)
            a=t.reservoir.arrays;n=t.reservoir.size
            x,_,_=t.spec.unpack(a['spatial'][:n],a['globals'][:n],a['legal'][:n])
            labels=np.unpackbits(a['greedy'][:n],axis=-1,count=t.actions,bitorder='little').astype(float)
            labels/=labels.sum(-1,keepdims=True)
            for role in (0,1):
                expected=labels[x[:,role].astype(bool)].mean(0)
                np.testing.assert_allclose(t.backend.average.probabilities[role].numpy(),expected,atol=1e-12,rtol=0)

    def test_budget_stop_and_source_plan_guards_preserve_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            config=replace(self.config(),time_cap_seconds=1e-12)
            t=NFSPMatrixTrainer(config,d,reserve_bytes=0);result=t.run(verbose=False)
            self.assertEqual(result['status'],'budget_limited');self.assertEqual(result['records'],0)
            CheckpointStore(d,reserve_bytes=0).verify(t.store.latest())
            with self.assertRaises(FileExistsError):NFSPMatrixTrainer(config,d,reserve_bytes=0)
            with self.assertRaises(ValueError):NFSPMatrixTrainer(replace(config,seed=7),d,resume=True,reserve_bytes=0)

    def test_sampler_matches_scalar_draws_and_keeps_zero_mass_unselected(self):
        rows=np.array([[0.,.25,.75],[1.,0.,0.],[0.,0.,1.]])
        rng=np.random.default_rng(4);other=np.random.default_rng(4)
        actions,prob=draw_actions(rows,rng)
        scalar=[draw_actions(row[None],other) for row in rows]
        self.assertEqual(list(actions),[int(x[0][0]) for x in scalar]);self.assertEqual(list(prob),[float(x[1][0]) for x in scalar])
        np.testing.assert_array_equal(actions[1:],[0,2])
        with self.assertRaises(ValueError):draw_actions(np.array([[.5,np.nan,.5]]),rng)

    def test_executed_label_control_isolates_exploration_noise(self):
        with tempfile.TemporaryDirectory() as d:
            trials=[]
            for label_rule in ('greedy','executed'):
                config=NFSPMatrixConfig(task='pure82',learner='tabular',seed=4,records=2048,eval_every=2048,
                        reservoir_open=0,handoff_end=1_000_000_000_000,label_rule=label_rule)
                t=NFSPMatrixTrainer(config,Path(d)/label_rule,reserve_bytes=0)
                with torch.no_grad():t.backend.q.values[0,0]=.5
                t.run(verbose=False);trials.append(t)
            a,b=trials
            self.assertGreater(a.backend.counters.supervised_updates,0)
            self.assertEqual(float(a.backend.average.probabilities[0,0]),1.)
            self.assertLess(float(b.backend.average.probabilities[0,0]),1.)
            self.assertGreater(float(b.backend.average.probabilities[0,0]),.7)
            np.testing.assert_array_equal(a.replay.arrays['action'],b.replay.arrays['action'])
            self.assertEqual(a.best_response_visits,b.best_response_visits)

    def test_stale_writer_refuses_to_overwrite_new_progress(self):
        with tempfile.TemporaryDirectory() as d:
            config=self.config();first=NFSPMatrixTrainer(config,d,reserve_bytes=0)
            first.run(until_records=118,verbose=False)
            stale=NFSPMatrixTrainer(config,d,resume=True,reserve_bytes=0)
            first.run(until_records=128,verbose=False)
            expected=(Path(d)/'episodes.bin').read_bytes()
            with self.assertRaises(RuntimeError):stale.run(until_records=160,verbose=False)
            self.assertEqual((Path(d)/'episodes.bin').read_bytes(),expected)

    def test_formal_flag_requires_primary_recipe_and_sufficient_final_evaluations(self):
        self.assertTrue(NFSPMatrixConfig().formal_protocol())
        self.assertFalse(replace(NFSPMatrixConfig(),records=320).formal_protocol())
        self.assertFalse(replace(NFSPMatrixConfig(),reservoir_open=0).formal_protocol())
        with self.assertRaises(ValueError):NFSPMatrixConfig(task='rps',label_rule='executed')

if __name__=='__main__':unittest.main()

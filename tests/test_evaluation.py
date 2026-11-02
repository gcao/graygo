from copy import deepcopy
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest
import numpy as np
from graygo.confidence import ConfidenceFamily,lower_bound,validate_estimate
from graygo.engine import BLACK,WHITE,Rules,State,resolve
from graygo.evaluation import BaselinePolicy,EpisodeMixture,EmptyStarts,FixedStarts,EvaluationPlan,EvaluationRun,evaluate_pair,play_frozen,seed_for


class ConfidenceTests(unittest.TestCase):
    def test_scaled_empirical_bernstein_matches_independent_formula(self):
        data=np.array([-1.,-.5,0.,1.]);delta=.01;n=len(data)
        variance=sum((float(x)-float(data.mean()))**2 for x in data)/(n-1)
        expected=math.sqrt(2*variance*math.log(2/delta)/n)+14*math.log(2/delta)/(3*(n-1))
        read=lower_bound(data,delta=delta)
        self.assertAlmostEqual(read['sample_variance'],variance);self.assertAlmostEqual(read['radius'],expected)
        transformed=lower_bound((data+1)/2,minimum=0,maximum=1,delta=delta)
        self.assertAlmostEqual(read['lower'],2*transformed['lower']-1)
        with self.assertRaises(ValueError):lower_bound([0,float('nan')])
        with self.assertRaises(ValueError):lower_bound([0,2])

    def test_family_counts_all_endpoints_and_planned_looks(self):
        family=ConfidenceFamily('f',('a','b'),(4,8),.05)
        self.assertEqual(family.delta,.05/4)
        read=family.estimate('a',4,[.5,1.]);validate_estimate(read,family)
        with self.assertRaises(ValueError):family.estimate('a',8,[.5,1.])
        with self.assertRaises(ValueError):family.estimate('c',4,[.5,1.])
        bad=dict(read,lower=1.)
        with self.assertRaises(ValueError):validate_estimate(bad,family)

    def test_exact_small_bernoulli_coverage_check(self):
        from math import comb
        n=10;delta=.05
        for probability in (0.,.01,.1,.3,.5,.9,.99,1.):
            mass=0.
            for successes in range(n+1):
                values=[1.]*successes+[0.]*(n-successes)
                bound=lower_bound(values,minimum=0,maximum=1,delta=delta)['lower']
                if bound>probability:mass+=comb(n,successes)*probability**successes*(1-probability)**(n-successes)
            self.assertLessEqual(mass,delta+1e-12)


class PairEvaluationTests(unittest.TestCase):
    def test_private_simultaneous_commitments_and_component_selection(self):
        class Actor:
            component=None
            def __init__(self,move,seen):self.move=move;self.seen=seen
            def action(self,state,player):self.seen.append((state,player));return self.move
        class Factory:
            sha256='f'*64
            def __init__(self,move):self.move=move;self.seen=[];self.started=0
            def begin_episode(self,seed):self.started+=1;return Actor(self.move,self.seen)
        root=State(Rules(3,1));a,b=Factory(0),Factory(1)
        result=play_frozen(a,b,root,BLACK,subject_seed=1,opponent_seed=2)
        self.assertIs(a.seen[0][0],b.seen[0][0]);self.assertEqual(a.started,b.started)
        self.assertEqual(result['rounds'],1)
        # The same opponent RNG is independent of the subject's sampling stream.
        random=BaselinePolicy('random');tactical=BaselinePolicy('tactical')
        a=evaluate_pair(random,tactical,State(Rules(3,5)),unit_seed=123)
        b=evaluate_pair(random,tactical,State(Rules(3,5)),unit_seed=123)
        self.assertEqual(a,b)

    def test_identical_policies_cancel_asymmetric_opening_advantage(self):
        class Pass:
            sha256='a'*64
            def begin_episode(self,seed):return self
            def action(self,state,player):return state.rules.pass_action
        board=[0]*81
        for point in (0,1,2):board[point]=BLACK
        board[40]=WHITE
        root=State(Rules(),board)
        result=evaluate_pair(Pass(),Pass(),root,unit_seed=11)
        self.assertEqual([g['subject_utility'] for g in result['subject']],[1.,-1.])
        self.assertEqual(result['utility'],0.)
        # Both role assignments retain the exact original public opening.
        self.assertEqual([g['black_score_units'] for g in result['subject']],[166,166])

    def test_resumable_matched_incumbent_evidence_is_identical(self):
        rules=Rules(3,4);policy=BaselinePolicy('random');opponent=BaselinePolicy('tactical')
        family=ConfidenceFamily('test',('delta',),(4,8))
        plan=EvaluationPlan('test','subject','delta',rules,7,(4,8),confidence_sha256=family.sha256,comparison='improvement')
        with tempfile.TemporaryDirectory() as d:
            full=EvaluationRun(plan,Path(d)/'full',policy,opponent,incumbent=policy,reserve_bytes=0)
            full.run(8,time_cap_seconds=30)
            part=EvaluationRun(plan,Path(d)/'part',policy,opponent,incumbent=policy,reserve_bytes=0)
            part.run(4,time_cap_seconds=30)
            resumed=EvaluationRun(plan,Path(d)/'part',policy,opponent,incumbent=policy,resume=True,reserve_bytes=0)
            resumed.run(8,time_cap_seconds=30)
            self.assertEqual(full.records(),resumed.records())
            self.assertTrue(all(r['improvement']==0 for r in resumed.records()))
            for row in resumed.records():self.assertEqual(row['subject'],row['incumbent'])
            result=resumed.estimate(family,8,comparison='improvement')
            self.assertEqual(result['independent_units'],4);self.assertEqual(result['mean'],0.)
            self.assertLess(result['lower'],0.)
            with self.assertRaises(ValueError):resumed.estimate(ConfidenceFamily('other',('delta',),(4,8)),8,comparison='improvement')
            with self.assertRaises(ValueError):resumed.estimate(family,8,comparison='utility')
            with self.assertRaises(ValueError):EvaluationRun(EvaluationPlan('changed','subject','delta',rules,7,(4,8),confidence_sha256=family.sha256,comparison='improvement'),Path(d)/'part',policy,opponent,incumbent=policy,resume=True,reserve_bytes=0)

    def test_repeated_trajectories_remain_independent_seed_units(self):
        class Pass:
            identity={'kind':'pass'};sha256='e'*64
            def assert_frozen(self):pass
            def begin_episode(self,seed):return self
            def action(self,state,player):return state.rules.pass_action
        family=ConfidenceFamily('repeat',('u',),(4,8));plan=EvaluationPlan('repeat','pass','u',Rules(3,4),3,(4,8),confidence_sha256=family.sha256)
        with tempfile.TemporaryDirectory() as d:
            run=EvaluationRun(plan,d,Pass(),Pass(),reserve_bytes=0);run.run(8,time_cap_seconds=20)
            result=run.estimate(family,8)
            self.assertEqual(len(run.records()),4);self.assertGreater(result['repeated_trajectory_fraction'],.5)
            self.assertEqual(result['independent_units'],4)

    def test_database_corruption_missing_evidence_and_interrupted_prefix(self):
        p=BaselinePolicy('random');family=ConfidenceFamily('x',('x',),(4,8));plan=EvaluationPlan('x','x','x',Rules(3,3),2,(4,8),confidence_sha256=family.sha256)
        with tempfile.TemporaryDirectory() as d:
            run=EvaluationRun(plan,d,p,p,reserve_bytes=0)
            state=run.run(8,time_cap_seconds=10,stop_requested=lambda:True)
            self.assertEqual(state['status'],'interrupted');self.assertEqual(run.records(),[])
            with self.assertRaises(ValueError):run.estimate(ConfidenceFamily('x',('x',),(4,8)),4)
            run.run(4,time_cap_seconds=10)
            db=sqlite3.connect(run.database);db.execute("UPDATE units SET payload='{}' WHERE idx=0");db.commit();db.close()
            with self.assertRaises(ValueError):run.records()
        with tempfile.TemporaryDirectory() as d:
            run=EvaluationRun(plan,d,p,p,reserve_bytes=0);run.database.unlink()
            with self.assertRaises(ValueError):EvaluationRun(plan,d,p,p,resume=True,reserve_bytes=0)

    def test_agent_color_exchange_complete_opening_and_private_episode_mixture(self):
        root=resolve(State(Rules(3,4)),0,1).state
        catalog=FixedStarts((root,),label='fixed')
        a,b=BaselinePolicy('random'),BaselinePolicy('tactical')
        mixture=EpisodeMixture((a,b),(.5,.5))
        one=evaluate_pair(mixture,a,catalog.sample(3,root.rules),unit_seed=123)
        two=evaluate_pair(mixture,a,root,unit_seed=123)
        self.assertEqual(one,two)
        for game in one['subject']:self.assertIn(game['subject_component'],[a.sha256,b.sha256])
        self.assertNotEqual(seed_for(1,'subject',0),seed_for(1,'opponent',0))

    def test_frozen_archive_origin_is_bound_to_verified_training_clock(self):
        from graygo.evaluation import FrozenNeuralPolicy
        from graygo.task_training import NFSPTaskConfig,NFSPTaskTrainer
        config=NFSPTaskConfig(time_cap_seconds=60,task='gray-mixed',learner='neural',records=8,eval_every=4,batch=4,learning_start=4,
            capacity=16,target_every=8,hidden=8,width=8,blocks=1,norm_groups=2,purpose='profile')
        with tempfile.TemporaryDirectory() as directory:
            trainer=NFSPTaskTrainer(config,directory,reserve_bytes=0);trainer.run(verbose=False)
            policy=FrozenNeuralPolicy.from_archive(Path(directory)/'weights/000000000008',representation='average')
            origin=policy.identity['origin']
            self.assertEqual(origin['records'],8);self.assertEqual(origin['seed'],0)
            self.assertEqual(origin['rules'],{'size':3,'max_rounds':6,'collision':'gray'})
            policy.assert_frozen()

    def test_frozen_neural_policy_does_not_follow_training_mutation(self):
        import torch
        from graygo.models import QNet,ModelConfig
        from graygo.evaluation import FrozenNeuralPolicy
        torch.set_num_threads(1);torch.manual_seed(3);model=QNet(ModelConfig(8,1,2));policy=FrozenNeuralPolicy(model,representation='greedy_q')
        before=policy.sha256
        with torch.no_grad():next(model.parameters()).add_(1.)
        policy.assert_frozen();self.assertEqual(policy.sha256,before)
        result=play_frozen(policy,BaselinePolicy('random'),State(Rules(3,2)),WHITE,subject_seed=4,opponent_seed=5)
        self.assertIn(result['subject_utility'],(-1.,0.,1.))
        with torch.no_grad():next(policy._model.parameters()).add_(1.)
        with self.assertRaises(ValueError):policy.assert_frozen()

if __name__=='__main__':unittest.main()

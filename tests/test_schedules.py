import unittest
import numpy as np
from graygo.schedules import LearnerSchedule
from graygo.buffers import ObservationSpec,ReservoirBuffer
from graygo.encoding import Observation


class FixedRng:
    def __init__(self,mode,ramp):self.values=np.array([mode,ramp])
    def random(self,count):return self.values.copy()


class ScheduleTests(unittest.TestCase):
    def test_fixed_board_epsilon_is_stage_independent(self):
        s=LearnerSchedule.board()
        for records,expected in ((0,.1),(5_000_000,.08875),(8_000_000,.082),(20_000_000,.055),(40_000_000,.01),(80_000_000,.01)):
            self.assertAlmostEqual(s.epsilon(records),expected)
        self.assertAlmostEqual(LearnerSchedule.a2().epsilon(15_000_000),.01)
        self.assertAlmostEqual(LearnerSchedule.a2().epsilon(17_000_000),.01)
        with self.assertRaises(ValueError):s.epsilon(-1)

    def test_ramp_and_unready_network_fallback(self):
        s=LearnerSchedule.board()
        self.assertEqual(s.ramp(5_000_000),0)
        self.assertEqual(s.ramp(6_000_000),.5)
        self.assertEqual(s.ramp(7_000_000),1)
        choice=s.choose_episode(8_000_000,FixedRng(.8,.1),average_ready=False)
        self.assertEqual(choice.acting_policy,'q')
        self.assertFalse(choice.best_response_mode)
        ready=s.choose_episode(8_000_000,FixedRng(.8,.1),average_ready=True)
        self.assertEqual(ready.acting_policy,'average')
        best=s.choose_episode(8_000_000,FixedRng(.01,.1),average_ready=True)
        self.assertTrue(best.best_response_mode)
        self.assertEqual(best.acting_policy,'q')

    def test_q_fallback_in_average_mode_never_creates_reservoir_labels(self):
        s=LearnerSchedule.board()
        choice=s.choose_episode(6_000_000,FixedRng(.8,.9),average_ready=True)
        self.assertEqual(choice.acting_policy,'q')
        self.assertFalse(s.label_eligible(choice,6_000_001))
        spec=ObservationSpec((1,),0,1,'schedule-test/1')
        obs=Observation(np.ones(1,np.uint8),np.zeros(0,np.float32),np.ones(1,bool))
        r=ReservoirBuffer(10,spec,opening_record=s.reservoir_open)
        r.offer(obs,np.ones(1,bool),record=6_000_001,policy_version=1,rng=np.random.default_rng(1),best_response_mode=choice.best_response_mode)
        self.assertEqual(r.seen,0)
        best=s.choose_episode(6_000_000,FixedRng(.01,.9),average_ready=True)
        self.assertTrue(s.label_eligible(best,6_000_001))
        self.assertFalse(s.label_eligible(best,6_000_001,supplemental=True))

    def test_confirmation_is_fixed_and_development_needs_its_gate(self):
        s=LearnerSchedule.board()
        self.assertFalse(s.opening_allowed(4_999_999,scope='confirmation'))
        self.assertTrue(s.opening_allowed(5_000_000,scope='confirmation',pilot_ready=False))
        self.assertFalse(s.opening_allowed(5_000_000,scope='development',pilot_ready=False))
        self.assertTrue(s.opening_allowed(5_000_000,scope='development',pilot_ready=True))

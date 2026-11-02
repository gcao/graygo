from copy import deepcopy
from dataclasses import asdict
import unittest
import numpy as np
import torch
from torch import nn
from graygo.buffers import ReplayBuffer,ReservoirBuffer,ObservationSpec
from graygo.encoding import Observation
from graygo.engine import BLACK,WHITE,State,Rules
from graygo.nfsp_collection import FrozenPolicies,EpisodeActor,JointEpisode,BoardGame
from graygo.schedules import LearnerSchedule,EpisodeChoice
from graygo.task_models import CompactPolicy,CompactQ
from graygo.tasks import DelayedGame,DelayedState


class Constant(nn.Module):
    def __init__(self,values):super().__init__();self.values=nn.Parameter(torch.tensor(values,dtype=torch.float32))
    def forward(self,spatial,globals):return self.values[None].expand(len(spatial),-1)


def choice(*,best=True,acting='q',eps=1.):return EpisodeChoice(0,best,acting,eps,0.,.05 if best else .5,.5)


class SimultaneousMatrix:
    spec=ObservationSpec((2,),0,2,'test/joint/1')
    @staticmethod
    def terminal(state):return state is not None
    @staticmethod
    def actions(state,player):return () if state is not None else (0,1)
    @staticmethod
    def observation(state,player):
        return Observation(np.eye(2)[player-1] if state is None else np.zeros(2),np.empty(0),np.full(2,state is None,dtype=bool))
    @staticmethod
    def transition(state,b,w):return (b,w)
    @staticmethod
    def utility(state):return 1 if state[0]==state[1] else -1


class CollectionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(12)

    def actors(self,snapshot,bchoice=None,wchoice=None):
        return EpisodeActor(snapshot,bchoice or choice(),np.random.default_rng(0)),EpisodeActor(snapshot,wchoice or choice(),np.random.default_rng(1))

    def test_executed_exploration_and_frozen_greedy_labels_are_distinct(self):
        game=SimultaneousMatrix();q=Constant([.9,-.9]);snapshot=FrozenPolicies(q,Constant([0.,0.]),version=3,average_ready=False)
        # Mutating online Q does not relabel or change the already frozen actor.
        q.values.data.mul_(-1)
        black,white=self.actors(snapshot);episode=JointEpisode(game,None,black,white,episode_id=4)
        replay=ReplayBuffer(8,game.spec);reservoir=ReservoirBuffer(8,game.spec)
        schedule=LearnerSchedule(100,reservoir_open=0,handoff_end=2)
        event=episode.step(replay,reservoir,np.random.default_rng(3),schedule,record=0)
        self.assertEqual(event['actions'][0],1);self.assertEqual(event['probabilities'],[.5,.5])
        self.assertEqual(int(replay.arrays['action'][0]),1)
        targets=reservoir.sample(8,np.random.default_rng(4))['targets']
        np.testing.assert_array_equal(targets,np.tile([1.,0.],(8,1)))
        self.assertEqual(replay.size,2);self.assertEqual(reservoir.seen,2)
        self.assertEqual(int(replay.arrays['policy_version'][0]),6)
        self.assertEqual(float(replay.arrays['reward'][0]),-float(replay.arrays['reward'][1]))

    def test_actual_marginal_probability_and_no_current_move_leak(self):
        game=SimultaneousMatrix();snapshot=FrozenPolicies(Constant([.9,-.9]),Constant([0.,0.]),version=0,average_ready=False)
        observation=game.observation(None,BLACK)
        actor=EpisodeActor(snapshot,choice(eps=.1),np.random.default_rng(2))
        for _ in range(40):
            result=actor.commit(observation)
            self.assertAlmostEqual(result.probability,.95 if result.action==0 else .05,places=7)
        actions=[]
        for seed in (0,1,2,3):
            b=EpisodeActor(snapshot,choice(),np.random.default_rng(seed))
            w=EpisodeActor(snapshot,choice(),np.random.default_rng(12))
            episode=JointEpisode(game,None,b,w,episode_id=seed)
            replay=ReplayBuffer(2,game.spec);reservoir=ReservoirBuffer(2,game.spec,opening_record=5_000_000)
            event=episode.step(replay,reservoir,np.random.default_rng(5),LearnerSchedule(100),record=0)
            actions.append(event['actions'])
        self.assertGreater(len({a[0] for a in actions}),1)
        self.assertEqual(len({a[1] for a in actions}),1)

    def test_average_mode_q_fallback_does_not_gain_reservoir_labels(self):
        game=SimultaneousMatrix();snapshot=FrozenPolicies(Constant([.9,-.9]),Constant([0.,0.]),version=1,average_ready=False)
        b,w=self.actors(snapshot,choice(best=False),choice(best=False))
        replay=ReplayBuffer(2,game.spec);reservoir=ReservoirBuffer(2,game.spec)
        episode=JointEpisode(game,None,b,w,episode_id=0)
        event=episode.step(replay,reservoir,np.random.default_rng(1),LearnerSchedule(100,reservoir_open=0,handoff_end=2),record=0)
        self.assertEqual(event['acting_policies'],['q','q']);self.assertEqual(reservoir.seen,0)
        self.assertFalse(replay.arrays['mode'].any())

    def test_supported_coverage_forced_root_and_all_suffix_labels_excluded(self):
        game=DelayedGame(horizons=(3,),equivalent_actions=4,codebook_seed=9)
        snapshot=FrozenPolicies(CompactQ(game.encoding.features,1,4,8),CompactPolicy(game.encoding.features,1,4,8),version=0,average_ready=False)
        b,w=self.actors(snapshot)
        episode=JointEpisode(game,game.root(3),b,w,episode_id=7,supplemental=True,forced_root=(BLACK,1))
        replay=ReplayBuffer(16,game.spec);reservoir=ReservoirBuffer(16,game.spec)
        rng=np.random.default_rng(1);schedule=LearnerSchedule(100,reservoir_open=0,handoff_end=2)
        while not game.terminal(episode.state):episode.step(replay,reservoir,rng,schedule,record=2*episode.steps)
        self.assertEqual(episode.log[0]['actions'][0],1);self.assertEqual(episode.log[0]['probabilities'][0],1.)
        self.assertEqual(reservoir.seen,0);self.assertTrue(replay.arrays['supplemental'][:6].all())
        self.assertFalse(replay.arrays['actor_mask'][:6].any());self.assertTrue(replay.arrays['value_mask'][:6].all())
        sequence=replay.sample_sequences(64,4,rng)
        for row in range(64):
            count=int(sequence['valid'][row].sum())
            self.assertEqual(len(set(sequence['stream'][row,:count])),1)
            self.assertTrue(sequence['terminated'][row,count-1])
            self.assertEqual(float(sequence['reward'][row,:count-1].sum()),0.)

    def test_mid_episode_restore_keeps_frozen_policies_and_private_rng(self):
        game=DelayedGame(horizons=(3,),equivalent_actions=4,codebook_seed=9)
        snapshot=FrozenPolicies(CompactQ(game.encoding.features,1,4,8),CompactPolicy(game.encoding.features,1,4,8),version=6,average_ready=False)
        b,w=self.actors(snapshot);episode=JointEpisode(game,game.root(3),b,w,episode_id=9)
        replay=ReplayBuffer(16,game.spec);reservoir=ReservoirBuffer(16,game.spec)
        rng=np.random.default_rng(14);schedule=LearnerSchedule(100,reservoir_open=0,handoff_end=2)
        episode.step(replay,reservoir,rng,schedule,record=0)
        state=deepcopy(episode.state_dict(asdict))
        restored=JointEpisode.restore(game,state,lambda x:DelayedState(**x),{6:snapshot})
        other_replay=deepcopy(replay);other_reservoir=deepcopy(reservoir)
        other_rng=np.random.default_rng();other_rng.bit_generator.state=deepcopy(rng.bit_generator.state)
        while not game.terminal(episode.state):
            record=2*episode.steps
            a=episode.step(replay,reservoir,rng,schedule,record=record)
            b=restored.step(other_replay,other_reservoir,other_rng,schedule,record=record)
            self.assertEqual(a,b)
        self.assertEqual(episode.state,restored.state)
        for name,array in replay.arrays.items():np.testing.assert_array_equal(array,other_replay.arrays[name])
        for name,array in reservoir.arrays.items():np.testing.assert_array_equal(array,other_reservoir.arrays[name])
        self.assertEqual(rng.bit_generator.state,other_rng.bit_generator.state)

    def test_board_adapter_uses_both_perspectives_and_resolves_actual_round(self):
        from graygo.models import QNet,PolicyNet,ModelConfig
        game=BoardGame();config=ModelConfig(width=8,blocks=1,norm_groups=2)
        snapshot=FrozenPolicies(QNet(config),PolicyNet(config),version=0,average_ready=False)
        b,w=self.actors(snapshot)
        episode=JointEpisode(game,State(Rules(3,2)),b,w,episode_id=1)
        spec=ObservationSpec.board(3);replay=ReplayBuffer(8,spec);reservoir=ReservoirBuffer(8,spec,opening_record=5_000_000)
        event=episode.step(replay,reservoir,np.random.default_rng(8),LearnerSchedule.a2(),record=0)
        self.assertEqual(replay.size,2);self.assertEqual(episode.state.rounds_played,1)
        self.assertEqual(replay.arrays['opponent_version'][0],replay.arrays['policy_version'][1])
        self.assertFalse(event['terminated']);self.assertEqual(event['reward_black'],0.)

    def test_invalid_mode_force_or_shared_action_rng_rejected(self):
        snapshot=FrozenPolicies(Constant([0.,0.]),Constant([0.,0.]),version=0,average_ready=False)
        with self.assertRaises(ValueError):EpisodeActor(snapshot,choice(acting='average'),np.random.default_rng())
        actor=EpisodeActor(snapshot,choice(),np.random.default_rng())
        with self.assertRaises(ValueError):actor.commit(SimultaneousMatrix.observation(None,BLACK),forced_action=0)
        b,w=self.actors(snapshot);w.rng=b.rng
        with self.assertRaises(ValueError):JointEpisode(SimultaneousMatrix(),None,b,w,episode_id=0)

if __name__=='__main__':unittest.main()

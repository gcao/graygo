"""Stationary production-path diagnostic of supplemental opponent intervention."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import argparse
from pathlib import Path
import json
import time
import numpy as np
import torch
from graygo.buffers import ReplayBuffer,ReservoirBuffer
from graygo.engine import BLACK,WHITE
from graygo.manifest import source_digest,make_manifest,write_report
from graygo.nfsp import OneStepBatch,one_step_targets,update_q
from graygo.nfsp_collection import FrozenPolicies,EpisodeActor,JointEpisode
from graygo.schedules import LearnerSchedule
from graygo.storage import digest
from graygo.tasks import DelayedGame

EXPECTED_SOURCE='ba5646f14d4aad47dbd9e7db33b9fad2c1cc1115df0e989c55522218054a5f6e'


class ExactQ(torch.nn.Module):
    def __init__(self,game):
        super().__init__();values=torch.zeros(game.spec.spatial_shape[0],game.spec.actions)
        for (key,role),index in game.encoding.ids.items():
            horizon,controller,depth,branch=key
            if depth==0:
                if role==controller:values[index,0]=1.;values[index,1]=-1.
                else:values[index,0]=-1.
            else:values[index,:]=(1. if branch==0 else -1.)*(1 if role==controller else -1)
        self.values=torch.nn.Parameter(values)
    def forward(self,spatial,globals):return spatial@self.values


class ExactIndices:
    def __init__(self,indices):self.indices=np.asarray(indices,dtype=np.int64)
    def integers(self,low,high,size):
        assert low==0 and size==len(self.indices) and (self.indices<high).all()
        return self.indices.copy()


def diagnose(root):
    root=Path(root).resolve()
    if root.exists():raise FileExistsError('diagnostic already exists; inspect its original output')
    if source_digest()!=EXPECTED_SOURCE:raise ValueError('use the exact original frozen 9fefeee source on PYTHONPATH')
    torch.set_num_threads(1);root.mkdir(parents=True)
    plan={'purpose':'stationary forensic loss/collector diagnostic','source_sha256':source_digest(),'driver_sha256':digest(Path(__file__)),
          'horizon':50,'equivalent_actions':40,'coverage_block':[8,1,1],'roles':[BLACK,WHITE],
          'maximum_collected_records':44,'oracle_initialization':True,'learner_selection_trial':False,
          'scope':'two exact-Q stationary fixtures through original collector/replay/target/loss; not learned convergence or a repaired implementation'}
    write_report(root/'plan.json',plan);write_report(root/'run.json',make_manifest(seed=719,command={'forced_opponent_diagnostic':plan}))
    started=time.monotonic();results=[]
    for controller in (BLACK,WHITE):
        game=DelayedGame(horizons=(50,),equivalent_actions=40,codebook_seed=719)
        q=ExactQ(game);target=deepcopy(q);snapshot=FrozenPolicies(q,deepcopy(q),version=0,average_ready=False)
        replay=ReplayBuffer(64,game.spec);reservoir=ReservoirBuffer(64,game.spec,opening_record=5000000)
        schedule=LearnerSchedule.a2();rng=np.random.default_rng(719+controller);last=None
        for episode_id,forced in enumerate([None]*8+[0,1]):
            actors=[]
            for role in (BLACK,WHITE):
                choice=replace(schedule.choose_episode(replay.insertions,rng,average_ready=False),epsilon=0.)
                actors.append(EpisodeActor(snapshot,choice,np.random.default_rng(1000+100*controller+10*episode_id+role)))
            last=JointEpisode(game,game.root(50,controller),*actors,episode_id=episode_id,
                              supplemental=forced is not None,forced_root=None if forced is None else (controller,forced))
            last.step(replay,reservoir,rng,schedule,record=replay.insertions)
        passive=3-controller;indices=np.arange(1 if passive==WHITE else 0,20,2)
        data=replay.sample(10,ExactIndices(indices));batch=OneStepBatch.from_replay(data,'cpu')
        labels,stats=one_step_targets(q,target,batch,0.)
        state=game.root(50,controller)
        while not game.terminal(state):state=game.transition(state,0,0)
        native=float(game.utility(state))*(1 if passive==BLACK else -1)
        assert native==-1. and np.all(data['value_mask']) and np.array_equal(data['supplemental'],[False]*8+[True,True])
        assert torch.equal(labels,torch.tensor([-1.]*9+[1.]))
        mean=float(labels.mean());ordinary=float(labels[~torch.tensor(data['supplemental'])].mean())
        assert abs(mean+.8)<1e-6 and ordinary==native
        own_indices=np.arange(0 if controller==BLACK else 1,20,2)
        own=replay.sample(10,ExactIndices(own_indices));own_labels,_=one_step_targets(q,target,OneStepBatch.from_replay(own,'cpu'),0.)
        conditional={str(action):float(own_labels[torch.tensor(own['action']==action)].mean()) for action in (0,1)}
        assert conditional=={'0':1.,'1':-1.}
        token=game.encoding.ids[(50,controller,0,-1),passive]
        before=float(q.values[token,0].detach())
        optimizer=torch.optim.SGD(q.parameters(),lr=.1)
        update=update_q(q,target,optimizer,batch,epsilon=0.,max_grad_norm=1.)
        after=float(q.values[token,0].detach());assert before==native and after>before
        last.step(replay,reservoir,rng,schedule,record=replay.insertions)
        assert replay.insertions==22 and replay.arrays['value_mask'][20:22].all() and reservoir.seen==0
        results.append({'controller':controller,'passive_role':passive,'native_root_value':native,'mixed_root_target_mean':mean,
                        'ordinary_only_root_target_mean':ordinary,'controller_conditional_targets':conditional,
                        'passive_supplemental_root_value_masks':data['value_mask'][-2:].tolist(),
                        'root_value_before_update':before,'root_value_after_update':after,'loss_targets':update['targets'],
                        'q_loss':update['loss'],'suffix_both_roles_value_eligible':True,'fresh_fixture_records':22})
    eta=Fraction(1,10);epsilon=Fraction(1,100);good=1-eta+eta*(1-epsilon/2)
    native=1-2*good;mixed=Fraction(4,5)*native+Fraction(1,10)*(-1)+Fraction(1,10)*(1)
    assert mixed-native==Fraction(999,5000)
    result={'verified':True,'defect_reproduced':True,'purpose':plan['purpose'],'source_sha256':source_digest(),'driver_sha256':plan['driver_sha256'],
            'elapsed_seconds':time.monotonic()-started,'fixtures':results,'total_fixture_records':44,
            'exact_primary_floor_example':{'eta':str(eta),'epsilon':str(epsilon),'average_good_action_probability':'1',
              'native_good_probability':str(good),'native_passive_root_value':str(native),'coverage_mixture_value':str(mixed),'systematic_bias':str(mixed-native)},
            'scope':'opponent-intervened passive root transitions bias the original native-opponent Bellman objective; own forced-action conditional targets and normal suffixes remain valid; this neither explains every long-horizon error nor implements a correction',
            'live_comparisons_modified':False,'new_large_training_selected':False,'a2_complete':False}
    write_report(root/'report.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_dir');args=parser.parse_args()
    print(json.dumps(diagnose(args.run_dir),sort_keys=True))

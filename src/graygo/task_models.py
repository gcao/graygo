"""Compact public-input networks for solved-game controls; no opponent metadata."""
import torch
from torch import nn


class CompactQ(nn.Module):
    def __init__(self,features,globals,actions,hidden=64):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(features+globals,hidden),nn.ReLU(),
                                  nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,actions),nn.Tanh())

    def forward(self,spatial,globals):return self.layers(torch.cat((spatial.float().flatten(1),globals.float()),-1))


class CompactPolicy(nn.Module):
    def __init__(self,features,globals,actions,hidden=64):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(features+globals,hidden),nn.ReLU(),
                                  nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,actions))

    def forward(self,spatial,globals):return self.layers(torch.cat((spatial.float().flatten(1),globals.float()),-1))


class TabularQ(nn.Module):
    """Independent direct estimates for opaque one-hot states, projected each step."""
    def __init__(self,states,actions):
        super().__init__();self.values=nn.Parameter(torch.zeros(states,actions))

    def forward(self,spatial,globals):
        if spatial.ndim!=2 or spatial.shape[1]!=len(self.values):raise ValueError('tabular Q requires state one-hot rows')
        if not bool(((spatial==0)|(spatial==1)).all()) or not bool((spatial.sum(-1)==1).all()):
            raise ValueError('tabular state must have exactly one active ID')
        return self.values[spatial.argmax(-1)]

    @torch.no_grad()
    def project_(self):self.values.clamp_(-1,1)

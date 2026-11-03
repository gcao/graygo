"""Frozen raw-policy baseline from a trusted graygo-old checkpoint.

This adapter uses the current authoritative resolver and independent private
actor RNGs. It does not import or run the old training/MCTS/gating loop.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
from graygo.confidence import canonical_sha
from graygo.engine import BLACK,WHITE,GRAY,EMPTY
from graygo.evaluation import seed_for
from graygo.sampling import draw_actions
from graygo.storage import digest as _digest

def digest(path):return _digest(Path(path))


class LegacyPolicy:
    def __init__(self,repo,checkpoint,*,label='old-champion-0081'):
        self.repo=Path(repo).resolve();self.checkpoint=Path(checkpoint).resolve()
        data=torch.load(self.checkpoint,weights_only=True,map_location='cpu')
        config=data['metadata']['config'];self.size=config['board_size']
        source=self.repo/'model.py';name='graygo_legacy_model_'+digest(source)[:12]
        previous=list(sys.path)
        try:
            sys.path.insert(0,str(self.repo))
            spec=importlib.util.spec_from_file_location(name,source)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            with torch.random.fork_rng(devices=[]),torch.device('cpu'):
                model=module.GrayGoNet(board_size=self.size,blocks=config['blocks'],filters=config['filters'])
            model.load_state_dict(data['state_dict'],strict=True)
        finally:sys.path[:]=previous
        self.model=model.eval().requires_grad_(False);self.weights=self._weights()
        self.identity={'kind':'legacy-graygo-raw-policy','label':label,'checkpoint_sha256':digest(self.checkpoint),
            'model_source_sha256':digest(source),'weights_sha256':self.weights,'board_size':self.size,
            'representation':'six-plane relative; legal softmax including pass; independent private sampling'}
        self.sha256=canonical_sha(self.identity)

    def _weights(self):
        h=hashlib.sha256()
        for name,value in sorted(self.model.state_dict().items()):
            a=value.detach().cpu().contiguous().numpy()
            h.update(name.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
        return h.hexdigest()

    def assert_frozen(self):
        if canonical_sha(self.identity)!=self.sha256 or self._weights()!=self.weights:raise ValueError('legacy policy changed')

    def _action_weights(self,state,player):
        if state.rules.size!=self.size or state.done:raise ValueError('legacy policy board/terminal mismatch')
        other=WHITE if player==BLACK else BLACK
        board=np.asarray(state.board).reshape(self.size,self.size)
        planes=np.zeros((6,self.size,self.size),dtype=np.float32)
        for channel,color in enumerate((player,other,GRAY,EMPTY)):planes[channel]=board==color
        for channel,role in ((4,player),(5,other)):
            for point in state.forbidden(role):planes[channel,point//self.size,point%self.size]=1
        with torch.no_grad():
            logits=self.model(torch.from_numpy(planes[None]))[0][0].double().numpy()
        legal=state.legal_actions(player);scores=logits[list(legal)]
        if not np.isfinite(scores).all():raise ValueError('nonfinite legacy logits')
        weights=np.zeros(self.size*self.size+1);p=np.exp(scores-scores.max());weights[list(legal)]=p/p.sum()
        return weights

    def begin_episode(self,seed):return _Actor(self,seed)


class _Actor:
    def __init__(self,policy,seed):self.policy=policy;self.rng=np.random.default_rng(seed);self.component=None
    def action(self,state,player):
        move,_=draw_actions(self.policy._action_weights(state,player)[None],self.rng)
        return int(move[0])

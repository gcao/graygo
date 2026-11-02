"""Shared PPO/GAE machinery for matrix and board rollout adapters.

Rollout collection owns old-policy snapshots and opponent selection. This module
updates the supplied actor/critic using explicit actor/value eligibility masks.
"""
from dataclasses import dataclass
import torch
from .models import masked_log_probs


@dataclass(frozen=True)
class PPOConfig:
    clip: float = .2
    epochs: int = 4
    minibatch: int = 128
    value_weight: float = .5
    entropy_weight: float = .01
    max_grad_norm: float = .5
    target_kl: float = .02

    def __post_init__(self):
        if not 0 < self.clip < 1 or self.epochs < 1 or self.minibatch < 1:
            raise ValueError("invalid PPO batch/clipping settings")
        if min(self.value_weight,self.entropy_weight) < 0 or min(self.max_grad_norm,self.target_kl) <= 0:
            raise ValueError("invalid PPO loss/stop settings")


@dataclass
class PPOBatch:
    inputs: dict[str,torch.Tensor]
    actions: torch.Tensor
    legal: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    actor_mask: torch.Tensor
    value_mask: torch.Tensor

    def validate(self):
        n=len(self.actions)
        if n<1 or self.actions.dtype != torch.long or self.legal.dtype != torch.bool:
            raise ValueError("invalid PPO action batch")
        if self.legal.ndim != 2 or self.legal.shape[0]!=n or not bool(self.legal.any(-1).all()):
            raise ValueError("PPO source observations must be nonterminal")
        if bool(((self.actions<0)|(self.actions>=self.legal.shape[1])).any()):
            raise ValueError("action outside policy head")
        if not bool(self.legal.gather(1,self.actions[:,None]).all()):
            raise ValueError("PPO batch contains an illegal action")
        for x in (self.old_log_probs,self.advantages,self.returns,self.actor_mask,self.value_mask):
            if x.shape!=(n,):raise ValueError("PPO field dimensions disagree")
        if self.actor_mask.dtype!=torch.bool or self.value_mask.dtype!=torch.bool:
            raise ValueError("eligibility masks must be bool")
        if not all(bool(torch.isfinite(x).all()) for x in (self.old_log_probs,self.advantages,self.returns)):
            raise ValueError("nonfinite PPO targets")
        if not self.inputs or any(x.shape[0]!=n for x in self.inputs.values()):
            raise ValueError("PPO model-input batch dimensions disagree")


def generalized_advantages(rewards,values,terminated,bootstrap,*,gamma=1.,lam=.95):
    if rewards.shape!=values.shape or rewards.shape!=terminated.shape or rewards.ndim!=2:
        raise ValueError("GAE requires [time,batch] rewards/values/done")
    if terminated.dtype!=torch.bool or bootstrap.shape!=rewards.shape[1:]:
        raise ValueError("invalid GAE terminal/bootstrap dimensions")
    if rewards.shape[0]<1 or not 0<=gamma<=1 or not 0<=lam<=1:
        raise ValueError("invalid GAE settings")
    with torch.no_grad():
        advantages=torch.zeros_like(values)
        tail=torch.zeros_like(bootstrap)
        next_value=bootstrap
        for t in reversed(range(rewards.shape[0])):
            alive=(~terminated[t]).to(values.dtype)
            delta=rewards[t]+gamma*alive*next_value-values[t]
            tail=delta+gamma*lam*alive*tail
            advantages[t]=tail
            next_value=values[t]
        return advantages,advantages+values


def update_ppo(model,optimizer,batch,config=PPOConfig(),*,shuffle_generator=None):
    batch.validate()
    n=len(batch.actions)
    advantages=batch.advantages.detach().clone()
    if bool(batch.actor_mask.any()):
        valid=advantages[batch.actor_mask]
        advantages[batch.actor_mask]=(valid-valid.mean())/valid.std(unbiased=False).clamp_min(1e-8)
    stats={"updates":0,"stopped_for_kl":False,"stop_kl":None,"policy_loss":0.,"value_loss":0.,"entropy":0.,"kl":0.}
    def finish():
        if stats["updates"]:
            for key in ("policy_loss","value_loss","entropy","kl"):stats[key]/=stats["updates"]
        return stats
    for _ in range(config.epochs):
        permutation=torch.randperm(n,generator=shuffle_generator)
        for start in range(0,n,config.minibatch):
            idx=permutation[start:start+config.minibatch].to(batch.actions.device)
            actor,value=batch.actor_mask[idx],batch.value_mask[idx]
            if not bool((actor|value).any()):
                continue
            logits,values=model(**{name:data[idx] for name,data in batch.inputs.items()})
            legal=batch.legal[idx]
            log_probs=masked_log_probs(logits,legal)
            selected=log_probs.gather(1,batch.actions[idx,None]).squeeze(1)
            zero=values.sum()*0.
            policy_loss,entropy,kl=zero,zero,zero
            if bool(actor.any()):
                log_ratio=selected[actor]-batch.old_log_probs[idx][actor].detach()
                ratio=log_ratio.exp()
                kl=((ratio-1)-log_ratio).mean()
                if kl.detach().item()>config.target_kl:
                    stats["stopped_for_kl"]=True
                    stats["stop_kl"]=kl.detach().item()
                    return finish()
                a=advantages[idx][actor]
                policy_loss=-torch.minimum(ratio*a,ratio.clamp(1-config.clip,1+config.clip)*a).mean()
                safe_log=log_probs.masked_fill(~legal,0.)
                h=-(log_probs.exp()*safe_log).sum(-1)
                count=legal.sum(-1)
                normalized=torch.where(count>1,h/count.clamp_min(2).float().log(),torch.zeros_like(h))
                entropy=normalized[actor].mean()
            value_loss=(values[value]-batch.returns[idx][value].detach()).square().mean() if bool(value.any()) else zero
            loss=policy_loss+config.value_weight*value_loss-config.entropy_weight*entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm,error_if_nonfinite=True)
            optimizer.step()
            stats["updates"]+=1
            for name,metric in (("policy_loss",policy_loss),("value_loss",value_loss),("entropy",entropy),("kl",kl)):
                stats[name]+=metric.detach().item()
    return finish()

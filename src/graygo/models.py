"""The design's circular 64-channel/eight-block policy, critic and Q networks."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .encoding import SPATIAL_CHANNELS, GLOBAL_FEATURES


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64
    blocks: int = 8
    norm_groups: int = 8

    def __post_init__(self):
        if self.width < 1 or self.blocks < 1 or self.norm_groups < 1 or self.width % self.norm_groups:
            raise ValueError("positive width/blocks and a normalization group divisor required")


class ResidualBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        c, g = config.width, config.norm_groups
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, padding_mode="circular")
        self.norm1 = nn.GroupNorm(g, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, padding_mode="circular")
        self.norm2 = nn.GroupNorm(g, c)

    def forward(self, x):
        return F.relu(x + self.norm2(self.conv2(F.relu(self.norm1(self.conv1(x))))))


class Encoder(nn.Module):
    def __init__(self, config, spatial_channels=SPATIAL_CHANNELS):
        super().__init__()
        if spatial_channels not in (30,32):raise ValueError("supported board input channels required")
        self.spatial_channels=spatial_channels
        self.stem = nn.Conv2d(spatial_channels + GLOBAL_FEATURES, config.width, 3,
                              padding=1, padding_mode="circular")
        self.norm = nn.GroupNorm(config.norm_groups, config.width)
        self.blocks = nn.Sequential(*(ResidualBlock(config) for _ in range(config.blocks)))

    def forward(self, spatial, globals):
        if spatial.ndim != 4 or spatial.shape[1] != self.spatial_channels or spatial.shape[-1] != spatial.shape[-2]:
            raise ValueError("expected spatial [batch,30,N,N]")
        if globals.shape != (spatial.shape[0], GLOBAL_FEATURES):
            raise ValueError("expected globals [batch,14]")
        g = globals.float()[:, :, None, None].expand(-1, -1, spatial.shape[-2], spatial.shape[-1])
        return self.blocks(F.relu(self.norm(self.stem(torch.cat((spatial.float(), g), dim=1)))))


class ActionHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.placements = nn.Sequential(nn.Conv2d(width, 32, 1), nn.ReLU(), nn.Conv2d(32, 1, 1))
        self.pass_head = nn.Sequential(nn.Linear(width + GLOBAL_FEATURES, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, features, globals):
        pooled = torch.cat((features.mean(dim=(-2,-1)), globals.float()), dim=1)
        return torch.cat((self.placements(features).flatten(1), self.pass_head(pooled)), dim=1)


class PolicyValueNet(nn.Module):
    def __init__(self, config=ModelConfig(), *, critic_buckets=0):
        super().__init__()
        if critic_buckets < 0:
            raise ValueError("negative critic bucket count")
        self.config, self.critic_buckets = config, critic_buckets
        self.encoder = Encoder(config)
        self.actor = ActionHead(config.width)
        # Training-only bucket input enters after pooling; never in actor/encoder.
        self.critic = nn.Sequential(nn.Linear(config.width + GLOBAL_FEATURES + critic_buckets, 64),
                                    nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, spatial, globals, buckets=None):
        features = self.encoder(spatial, globals)
        logits = self.actor(features, globals)
        inputs = [features.mean(dim=(-2,-1)), globals.float()]
        if self.critic_buckets:
            if buckets is None or buckets.shape != (spatial.shape[0], self.critic_buckets):
                raise ValueError("critic bucket one-hot matrix required")
            inputs.append(buckets.float())
        elif buckets is not None:
            raise ValueError("this model has no critic bucket inputs")
        return logits, self.critic(torch.cat(inputs, dim=1)).squeeze(-1)

    def policy_logits(self, spatial, globals):
        return self.actor(self.encoder(spatial, globals), globals)


class PolicyNet(nn.Module):
    """Independent public-input average policy for the NFSP branch."""
    def __init__(self, config=ModelConfig(), *, spatial_channels=30):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config,spatial_channels)
        self.head = ActionHead(config.width)

    def forward(self, spatial, globals):
        return self.head(self.encoder(spatial, globals), globals)


class QNet(nn.Module):
    def __init__(self, config=ModelConfig(), *, spatial_channels=30):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config,spatial_channels)
        self.head = ActionHead(config.width)

    def forward(self, spatial, globals):
        return torch.tanh(self.head(self.encoder(spatial, globals), globals))


def masked_log_probs(logits, legal):
    if logits.shape != legal.shape or legal.dtype != torch.bool:
        raise ValueError("legal bool mask must match action logits")
    if not bool(legal.any(dim=-1).all()):
        raise ValueError("cannot sample terminal/all-masked action rows")
    return torch.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=-1)


def greedy_distribution(q, legal):
    if q.shape != legal.shape or legal.dtype != torch.bool or not bool(legal.any(dim=-1).all()):
        raise ValueError("invalid legal action rows")
    best = q.masked_fill(~legal, -torch.inf).max(dim=-1, keepdim=True).values
    tied = legal & (q == best)
    return tied.float() / tied.sum(dim=-1, keepdim=True)


def epsilon_greedy_distribution(q, legal, epsilon):
    """Marginal action probabilities, including exploration that hits a maximizer.

Replay must log this probability, not the probability of whichever internal
exploration coin branch happened to be used when drawing the action.
"""
    if q.shape!=legal.shape or legal.dtype!=torch.bool or not torch.is_floating_point(q):
        raise ValueError("floating Q values and matching boolean masks required")
    if not torch.isfinite(q[legal]).all():
        raise ValueError("finite legal Q values required")
    eps=torch.as_tensor(epsilon,dtype=q.dtype,device=q.device)
    if eps.shape==q.shape[:-1]:eps=eps.unsqueeze(-1)
    if eps.ndim and eps.shape!=q.shape[:-1]+(1,):
        raise ValueError("epsilon must be scalar or one per action row")
    if not bool(torch.isfinite(eps).all()) or bool(((eps<0)|(eps>1)).any()):
        raise ValueError("epsilon must lie in [0,1]")
    greedy=greedy_distribution(q,legal)
    uniform=legal.to(q.dtype)/legal.sum(-1,keepdim=True)
    return (1-eps)*greedy+eps*uniform


def transfer_board_weights(model, weights):
    """Preserve old spatial/globals coefficients and zero only new hint inputs."""
    copied=dict(weights);key='encoder.stem.weight';old=weights[key];new=model.state_dict()[key]
    if old.shape != new.shape:
        if old.shape[1]!=44 or new.shape[1]!=46 or old.shape[0]!=new.shape[0]:raise ValueError('unsupported input-channel transfer')
        value=torch.zeros_like(new);value[:,:30]=old[:,:30];value[:,32:]=old[:,30:];copied[key]=value
    model.load_state_dict(copied,strict=True)

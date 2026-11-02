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
    def __init__(self, config):
        super().__init__()
        self.stem = nn.Conv2d(SPATIAL_CHANNELS + GLOBAL_FEATURES, config.width, 3,
                              padding=1, padding_mode="circular")
        self.norm = nn.GroupNorm(config.norm_groups, config.width)
        self.blocks = nn.Sequential(*(ResidualBlock(config) for _ in range(config.blocks)))

    def forward(self, spatial, globals):
        if spatial.ndim != 4 or spatial.shape[1] != SPATIAL_CHANNELS or spatial.shape[-1] != spatial.shape[-2]:
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


class QNet(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
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

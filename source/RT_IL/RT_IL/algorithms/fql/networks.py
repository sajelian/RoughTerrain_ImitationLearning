"""Networks for FQL: behavior flow, one-step policy, and Q critic."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dims: list[int], activation: str = "elu") -> nn.Sequential:
    act_cls = {"elu": nn.ELU, "relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[activation.lower()]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), act_cls()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding. `t` shape (B,), output (B, dim)."""
    assert dim % 2 == 0, "time embedding dim must be even"
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class FlowMLP(nn.Module):
    """Conditional flow-matching velocity field v_theta(a_t, t, s).

    Linear-interpolant FM: a_t = (1 - t) * z + t * a, z ~ N(0, I).
    Velocity target during training: a - z. Inference is iterative Euler integration
    from z (at t=0) to a (at t=1).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] = (512, 512, 512),
        time_embed_dim: int = 64,
        activation: str = "silu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.time_embed_dim = time_embed_dim
        in_dim = obs_dim + action_dim + time_embed_dim
        dims = [in_dim, *hidden_dims, action_dim]
        self.net = _mlp(dims, activation=activation)

    def forward(self, a_t: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        t_embed = sinusoidal_time_embedding(t, self.time_embed_dim)
        x = torch.cat([a_t, t_embed, obs], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def integrate(self, obs: torch.Tensor, z: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """Euler-integrate the flow from z (t=0) to a (t=1) conditioned on obs."""
        a = z
        dt = 1.0 / num_steps
        for k in range(num_steps):
            t = torch.full((a.shape[0],), k * dt, device=a.device)
            a = a + dt * self.forward(a, t, obs)
        return a


class OneStepFlowPolicy(nn.Module):
    """Noise-conditioned one-step policy π_φ(s, z) → a.

    Distilled from the iterative flow π_β so deployment doesn't require integration.
    The noise input `z` preserves multimodality: different z values can route to
    different modes of the action distribution.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] = (512, 512, 512),
        activation: str = "silu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        dims = [obs_dim + action_dim, *hidden_dims, action_dim]
        self.net = _mlp(dims, activation=activation)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, z], dim=-1))

    @torch.no_grad()
    def sample(self, obs: torch.Tensor) -> torch.Tensor:
        z = torch.randn(obs.shape[0], self.action_dim, device=obs.device)
        return self.forward(obs, z)


class QNetwork(nn.Module):
    """Standard Q(s, a) MLP. Used as a clipped-double-Q pair in FQLAgent."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] = (512, 512, 512),
        activation: str = "elu",
    ):
        super().__init__()
        dims = [obs_dim + action_dim, *hidden_dims, 1]
        self.net = _mlp(dims, activation=activation)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)

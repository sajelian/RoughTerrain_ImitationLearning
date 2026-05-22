"""Replay buffer for FQL offline training.

Loads one or more (s, a, r, s', done) .pt files produced by
`scripts/rsl_rl/collect_fql_dataset.py`, optionally standardizes observations,
and serves random minibatches.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class FQLBatch:
    obs: torch.Tensor          # (B, obs_dim) normalized
    action: torch.Tensor       # (B, action_dim)
    reward: torch.Tensor       # (B,)
    next_obs: torch.Tensor     # (B, obs_dim) normalized
    done: torch.Tensor         # (B,) float (0/1)


class FQLReplayBuffer:
    """Aggregates rollout files into a single in-memory replay buffer.

    Storage stays on CPU (pinned). Each `sample()` call ships a minibatch to the
    target device. Suitable for datasets up to ~32 GB of CPU RAM; for larger
    datasets switch to memory-mapped tensors or HDF5.
    """

    def __init__(
        self,
        paths: str | list[str],
        device: str | torch.device = "cuda",
        normalize_obs: bool = True,
        obs_clip: float = 5.0,
    ):
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            raise ValueError("FQLReplayBuffer needs at least one rollout file.")

        obs_chunks, act_chunks, rew_chunks, next_obs_chunks, done_chunks = [], [], [], [], []
        action_dim = obs_dim = None
        for p in paths:
            assert os.path.isfile(p), f"Rollout file not found: {p}"
            data = torch.load(p, weights_only=False, map_location="cpu")
            obs_chunks.append(data["observations"])
            act_chunks.append(data["actions"])
            rew_chunks.append(data["rewards"])
            next_obs_chunks.append(data["next_observations"])
            done_chunks.append(data["dones"])
            md = data.get("metadata", {})
            if obs_dim is None:
                obs_dim = md.get("student_obs_dim", obs_chunks[0].shape[-1])
                action_dim = md.get("action_dim", act_chunks[0].shape[-1])
            else:
                if md.get("student_obs_dim", obs_chunks[-1].shape[-1]) != obs_dim:
                    raise ValueError(f"obs_dim mismatch across files at {p}")
                if md.get("action_dim", act_chunks[-1].shape[-1]) != action_dim:
                    raise ValueError(f"action_dim mismatch across files at {p}")

        self.observations = torch.cat(obs_chunks, dim=0)
        self.actions = torch.cat(act_chunks, dim=0)
        self.rewards = torch.cat(rew_chunks, dim=0).float()
        self.next_observations = torch.cat(next_obs_chunks, dim=0)
        self.dones = torch.cat(done_chunks, dim=0).float()
        self.obs_dim = self.observations.shape[-1]
        self.action_dim = self.actions.shape[-1]
        self.n = self.observations.shape[0]

        self.device = torch.device(device)
        self.obs_clip = float(obs_clip)

        if normalize_obs:
            self.obs_mean = self.observations.mean(dim=0)
            self.obs_std = self.observations.std(dim=0).clamp(min=1e-3)
        else:
            self.obs_mean = torch.zeros(self.obs_dim)
            self.obs_std = torch.ones(self.obs_dim)
        # Move stats to target device once.
        self.obs_mean_dev = self.obs_mean.to(self.device)
        self.obs_std_dev = self.obs_std.to(self.device)

        # Reward stats (just for logging; not applied automatically).
        self.reward_mean = float(self.rewards.mean().item())
        self.reward_std = float(self.rewards.std().item())

        print(f"[FQL] Loaded {self.n:,} transitions from {len(paths)} file(s). "
              f"obs={self.obs_dim} action={self.action_dim}")
        print(f"      obs:    mean=[{self.obs_mean.min().item():+.2f}, {self.obs_mean.max().item():+.2f}]  "
              f"std=[{self.obs_std.min().item():.2f}, {self.obs_std.max().item():.2f}]")
        print(f"      reward: mean={self.reward_mean:+.4f}  std={self.reward_std:.4f}  "
              f"min={self.rewards.min().item():+.4f}  max={self.rewards.max().item():+.4f}")
        print(f"      done_rate={self.dones.mean().item():.4f}")

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return ((obs - self.obs_mean_dev) / self.obs_std_dev).clamp(-self.obs_clip, self.obs_clip)

    def denormalize_obs(self, obs_n: torch.Tensor) -> torch.Tensor:
        return obs_n * self.obs_std_dev + self.obs_mean_dev

    def sample(self, batch_size: int) -> FQLBatch:
        idx = torch.randint(0, self.n, (batch_size,))
        # Pull raw tensors on CPU, then ship to device.
        obs = self.observations[idx].to(self.device, non_blocking=True)
        act = self.actions[idx].to(self.device, non_blocking=True)
        rew = self.rewards[idx].to(self.device, non_blocking=True)
        next_obs = self.next_observations[idx].to(self.device, non_blocking=True)
        done = self.dones[idx].to(self.device, non_blocking=True)
        return FQLBatch(
            obs=self.normalize_obs(obs),
            action=act,
            reward=rew,
            next_obs=self.normalize_obs(next_obs),
            done=done,
        )

"""Replay buffer for FQL offline training.

Loads one or more (s, a, r, s', done) .pt files produced by
`scripts/rsl_rl/collect_fql_dataset.py` and serves random minibatches.

Strategy: source files are memory-mapped (zero RAM at open), then copied
chunk-by-chunk into RAM-resident float16 tensors. float16 halves the footprint
(obs are normalized NN inputs — 3 decimal digits is plenty) and the chunked
copy keeps peak memory at `final_size + one_chunk` rather than `2 x file_size`.
Sampling then indexes RAM tensors directly (microseconds, no disk).

Footprint: ~obs_dim * 4 bytes/transition  (obs + next_obs in fp16).
e.g. 4M transitions @ obs_dim=988 -> ~16 GB RAM.  For datasets too large for
RAM, pass `keep_on_disk=True` to sample straight from the mmap (slower).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class FQLBatch:
    obs: torch.Tensor          # (B, obs_dim) normalized float32
    action: torch.Tensor       # (B, action_dim) float32
    reward: torch.Tensor       # (B,) float32
    next_obs: torch.Tensor     # (B, obs_dim) normalized float32
    done: torch.Tensor         # (B,) float32 (0/1)


class FQLReplayBuffer:
    def __init__(
        self,
        paths: str | list[str],
        device: str | torch.device = "cuda",
        normalize_obs: bool = True,
        obs_clip: float = 5.0,
        action_clip_pct: float = 0.5,
        chunk: int = 250_000,
        stat_samples: int = 1_000_000,
    ):
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            raise ValueError("FQLReplayBuffer needs at least one rollout file.")

        # First pass: mmap each file, sum sizes, validate dims.
        mmaps: list[dict] = []
        sizes: list[int] = []
        for p in paths:
            assert os.path.isfile(p), f"Rollout file not found: {p}"
            data = torch.load(p, weights_only=False, map_location="cpu", mmap=True)
            mmaps.append(data)
            sizes.append(int(data["observations"].shape[0]))
        self.obs_dim = int(mmaps[0]["observations"].shape[-1])
        self.action_dim = int(mmaps[0]["actions"].shape[-1])
        for fi, data in enumerate(mmaps):
            if data["observations"].shape[-1] != self.obs_dim:
                raise ValueError(f"obs_dim mismatch in {paths[fi]}")
            if data["actions"].shape[-1] != self.action_dim:
                raise ValueError(f"action_dim mismatch in {paths[fi]}")
        self.n = sum(sizes)

        # Preallocate RAM-resident tensors (fp16 for the big ones).
        self.observations = torch.empty((self.n, self.obs_dim), dtype=torch.float16)
        self.next_observations = torch.empty((self.n, self.obs_dim), dtype=torch.float16)
        self.actions = torch.empty((self.n, self.action_dim), dtype=torch.float16)
        self.rewards = torch.empty((self.n,), dtype=torch.float32)
        self.dones = torch.empty((self.n,), dtype=torch.float32)

        # Chunked copy mmap -> RAM (bounds peak memory to one chunk of fp32).
        # The reward comes straight from whichever env produced this file — for
        # velocity data that's the velocity-tracking reward; for skill data
        # that's the motion-tracking reward. Either way it's the natural
        # supervision signal for that data source's Q-learning objective.
        write = 0
        for data, sz in zip(mmaps, sizes):
            for lo in range(0, sz, chunk):
                hi = min(lo + chunk, sz)
                dst = slice(write + lo, write + hi)
                self.observations[dst] = data["observations"][lo:hi].half()
                self.next_observations[dst] = data["next_observations"][lo:hi].half()
                self.actions[dst] = data["actions"][lo:hi].half()
                self.rewards[dst] = data["rewards"][lo:hi].float()
                self.dones[dst] = data["dones"][lo:hi].float()
            write += sz
        del mmaps  # release the memory maps

        self.device = torch.device(device)
        self.obs_clip = float(obs_clip)

        # --- Clip the action tail per joint ---
        # Expert clips contain rare reactive transients (RSI settling, contact
        # impacts) up to ~25 sigma. They are not reproducible skill actions and
        # poison both normalization and the flow fit. Clip per-joint at the
        # [action_clip_pct, 100-action_clip_pct] percentiles before computing
        # any stats. Genuine sustained pushes live in the bulk and survive.
        si = torch.randint(0, self.n, (min(self.n, stat_samples),))
        self.action_clip_pct = float(action_clip_pct)
        if action_clip_pct > 0.0:
            qs = self.actions[si].float()
            lo_q = torch.quantile(qs, action_clip_pct / 100.0, dim=0)
            hi_q = torch.quantile(qs, 1.0 - action_clip_pct / 100.0, dim=0)
            del qs
            n_clipped = ((self.actions.float() < lo_q) | (self.actions.float() > hi_q)).any(dim=-1).sum()
            self.actions = torch.minimum(torch.maximum(self.actions, lo_q.half()), hi_q.half())
            self.action_clip_lo = lo_q
            self.action_clip_hi = hi_q
            print(f"[FQL] Action tail clipped at [{action_clip_pct}%, {100-action_clip_pct}%] per joint: "
                  f"{n_clipped.item():,} / {self.n:,} transitions touched "
                  f"({100*n_clipped.item()/self.n:.2f}%)")
        else:
            self.action_clip_lo = None
            self.action_clip_hi = None

        # Normalization stats reuse the subset indices `si` drawn above.
        if normalize_obs:
            obs_sample = self.observations[si].float()
            self.obs_mean = obs_sample.mean(dim=0)
            self.obs_std = obs_sample.std(dim=0).clamp(min=1e-3)
            del obs_sample
        else:
            self.obs_mean = torch.zeros(self.obs_dim)
            self.obs_std = torch.ones(self.obs_dim)
        self.obs_mean_dev = self.obs_mean.to(self.device)
        self.obs_std_dev = self.obs_std.to(self.device)

        # Action normalization is NOT optional — flow matching assumes actions on
        # a scale comparable to the z~N(0,I) prior. Raw joint actions are O(±8).
        act_sample = self.actions[si].float()
        self.action_mean = act_sample.mean(dim=0)
        self.action_std = act_sample.std(dim=0).clamp(min=1e-3)
        del act_sample
        self.action_mean_dev = self.action_mean.to(self.device)
        self.action_std_dev = self.action_std.to(self.device)

        self.reward_mean = float(self.rewards.mean())
        self.reward_std = float(self.rewards.std())
        done_rate = float(self.dones.mean())

        gb = (self.observations.element_size() * self.observations.nelement() * 2
              + self.actions.element_size() * self.actions.nelement()) / 1e9
        print(f"[FQL] Loaded {self.n:,} transitions from {len(paths)} file(s) into RAM (fp16). "
              f"obs={self.obs_dim} action={self.action_dim}  (~{gb:.1f} GB RAM)")
        print(f"      obs:    mean=[{self.obs_mean.min().item():+.2f}, {self.obs_mean.max().item():+.2f}]  "
              f"std=[{self.obs_std.min().item():.2f}, {self.obs_std.max().item():.2f}]")
        print(f"      action: mean=[{self.action_mean.min().item():+.2f}, {self.action_mean.max().item():+.2f}]  "
              f"std=[{self.action_std.min().item():.2f}, {self.action_std.max().item():.2f}]  (standardized)")
        print(f"      reward: mean={self.reward_mean:+.4f}  std={self.reward_std:.4f}  "
              f"done_rate={done_rate:.4f}")

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return ((obs - self.obs_mean_dev) / self.obs_std_dev).clamp(-self.obs_clip, self.obs_clip)

    def denormalize_obs(self, obs_n: torch.Tensor) -> torch.Tensor:
        return obs_n * self.obs_std_dev + self.obs_mean_dev

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean_dev) / self.action_std_dev

    def denormalize_action(self, action_n: torch.Tensor) -> torch.Tensor:
        """Map a policy output in standardized space back to raw env action units."""
        return action_n * self.action_std_dev + self.action_mean_dev

    def sample(self, batch_size: int) -> FQLBatch:
        idx = torch.randint(0, self.n, (batch_size,))
        obs = self.observations[idx].to(self.device, non_blocking=True).float()
        act = self.actions[idx].to(self.device, non_blocking=True).float()
        rew = self.rewards[idx].to(self.device, non_blocking=True)
        next_obs = self.next_observations[idx].to(self.device, non_blocking=True).float()
        done = self.dones[idx].to(self.device, non_blocking=True)
        return FQLBatch(
            obs=self.normalize_obs(obs),
            action=self.normalize_action(act),
            reward=rew,
            next_obs=self.normalize_obs(next_obs),
            done=done,
        )

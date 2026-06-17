"""Replay buffer for FQL offline training.

Loads one or more (s, a, r, s', done) .pt files produced by
`scripts/rsl_rl/collect_fql_dataset.py` and serves random minibatches.

Two storage modes:

* **In-RAM (default).** Source files are memory-mapped (zero RAM at open), then
  copied chunk-by-chunk into RAM-resident float16 tensors. float16 halves the
  footprint (obs are normalized NN inputs — 3 decimal digits is plenty) and the
  chunked copy keeps peak memory at `final_size + one_chunk` rather than
  `2 x file_size`. Sampling then indexes RAM tensors directly (microseconds).

  Footprint: ~obs_dim * 4 bytes/transition (obs + next_obs in fp16). e.g. 4M
  transitions @ obs_dim=988 -> ~16 GB RAM.

* **Disk-backed (`keep_on_disk=True`).** The files stay memory-mapped and every
  batch is gathered straight from the mmap — RAM stays near zero regardless of
  dataset size. Use this when the union of datasets does not fit in RAM, or to
  keep RAM free for an online / DAgger buffer alongside the offline data. Per
  batch this costs `batch_size` random page reads per obs tensor; on an SSD that
  is a few ms (vs microseconds in-RAM), so training is I/O-bound but unbounded
  in dataset size. fp16 source files (see collect_fql_dataset.py) halve those
  reads.
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
        max_total_transitions: int | None = None,
        keep_on_disk: bool = False,
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
        total = sum(sizes)

        # If a RAM cap is set and total exceeds it, subsample each file
        # proportionally so the union fits. Per-file uniform-random subset.
        # (Disk-backed mode ignores this — disk has no such ceiling.)
        target_sizes: list[int]
        per_file_idx: list[torch.Tensor | None]
        if (not keep_on_disk) and max_total_transitions is not None and total > max_total_transitions:
            frac = max_total_transitions / total
            target_sizes = [max(1, int(sz * frac)) for sz in sizes]
            per_file_idx = []
            for sz, tsz in zip(sizes, target_sizes):
                if tsz == sz:
                    per_file_idx.append(None)
                else:
                    # Random subset, then sort for sequential mmap reads
                    idx = torch.randperm(sz)[:tsz]
                    idx, _ = torch.sort(idx)
                    per_file_idx.append(idx)
            print(f"[FQL] Subsampling: {total:,} → {sum(target_sizes):,} transitions "
                  f"({100*frac:.1f}% per file) to fit max_total_transitions={max_total_transitions:,}")
        else:
            target_sizes = sizes
            per_file_idx = [None] * len(sizes)
        self.n = sum(target_sizes)

        self.keep_on_disk = bool(keep_on_disk)
        self.device = torch.device(device)
        self.obs_clip = float(obs_clip)
        self.action_clip_pct = float(action_clip_pct)

        # State shared by both modes for index → file/row mapping.
        self._sizes = sizes
        self._target_sizes = target_sizes
        self._file_idx = per_file_idx

        if self.keep_on_disk:
            # Disk-backed: keep the mmaps; per-batch sampling reads from them.
            self._mmaps = mmaps
            probs = torch.tensor(target_sizes, dtype=torch.float64)
            self._probs = probs / probs.sum()
            print(f"[FQL] Disk-backed buffer: {self.n:,} transitions across {len(paths)} "
                  f"mmap'd file(s), ~0 GB RAM (sampling reads from disk).")
        else:
            # In-RAM: preallocate fp16 tensors and chunk-copy from the mmaps.
            self.observations = torch.empty((self.n, self.obs_dim), dtype=torch.float16)
            self.next_observations = torch.empty((self.n, self.obs_dim), dtype=torch.float16)
            self.actions = torch.empty((self.n, self.action_dim), dtype=torch.float16)
            self.rewards = torch.empty((self.n,), dtype=torch.float32)
            self.dones = torch.empty((self.n,), dtype=torch.float32)
            write = 0
            for data, tsz, idx in zip(mmaps, target_sizes, per_file_idx):
                if idx is None:
                    for lo in range(0, tsz, chunk):
                        hi = min(lo + chunk, tsz)
                        dst = slice(write + lo, write + hi)
                        self.observations[dst] = data["observations"][lo:hi].half()
                        self.next_observations[dst] = data["next_observations"][lo:hi].half()
                        self.actions[dst] = data["actions"][lo:hi].half()
                        self.rewards[dst] = data["rewards"][lo:hi].float()
                        self.dones[dst] = data["dones"][lo:hi].float()
                else:
                    for lo in range(0, tsz, chunk):
                        hi = min(lo + chunk, tsz)
                        chunk_idx = idx[lo:hi]
                        dst = slice(write + lo, write + hi)
                        self.observations[dst] = data["observations"][chunk_idx].half()
                        self.next_observations[dst] = data["next_observations"][chunk_idx].half()
                        self.actions[dst] = data["actions"][chunk_idx].half()
                        self.rewards[dst] = data["rewards"][chunk_idx].float()
                        self.dones[dst] = data["dones"][chunk_idx].float()
                write += tsz
            del mmaps  # release the memory maps

        # --- Stats sample (used for action-clip percentiles + normalization) ---
        # Drawn once the same way in both modes (one disk gather in disk mode).
        stat_n = min(self.n, stat_samples)
        s_obs, _, s_act, s_rew, s_done = self._raw_sample(stat_n)

        # --- Clip the action tail per joint ---
        # Expert clips contain rare reactive transients (RSI settling, contact
        # impacts) up to ~25 sigma. They are not reproducible skill actions and
        # poison both normalization and the flow fit. Clip per-joint at the
        # [action_clip_pct, 100-action_clip_pct] percentiles. Genuine sustained
        # pushes live in the bulk and survive. Clip is applied at sample time
        # (and once in place to the RAM tensors so diagnostics see clipped data).
        if action_clip_pct > 0.0:
            lo_q = torch.quantile(s_act, action_clip_pct / 100.0, dim=0)
            hi_q = torch.quantile(s_act, 1.0 - action_clip_pct / 100.0, dim=0)
            self.action_clip_lo = lo_q
            self.action_clip_hi = hi_q
            touched = ((s_act < lo_q) | (s_act > hi_q)).any(dim=-1).float().mean()
            s_act = torch.minimum(torch.maximum(s_act, lo_q), hi_q)
            if not self.keep_on_disk:
                self.actions = torch.minimum(torch.maximum(self.actions, lo_q.half()), hi_q.half())
            print(f"[FQL] Action tail clipped at [{action_clip_pct}%, {100-action_clip_pct}%] per joint: "
                  f"~{100*touched.item():.2f}% of transitions touched (estimated on {stat_n:,} samples).")
        else:
            self.action_clip_lo = None
            self.action_clip_hi = None

        # --- Observation normalization stats ---
        if normalize_obs:
            self.obs_mean = s_obs.mean(dim=0)
            self.obs_std = s_obs.std(dim=0).clamp(min=1e-3)
        else:
            self.obs_mean = torch.zeros(self.obs_dim)
            self.obs_std = torch.ones(self.obs_dim)
        self.obs_mean_dev = self.obs_mean.to(self.device)
        self.obs_std_dev = self.obs_std.to(self.device)

        # Action normalization is NOT optional — flow matching assumes actions on
        # a scale comparable to the z~N(0,I) prior. Raw joint actions are O(±8).
        self.action_mean = s_act.mean(dim=0)
        self.action_std = s_act.std(dim=0).clamp(min=1e-3)
        self.action_mean_dev = self.action_mean.to(self.device)
        self.action_std_dev = self.action_std.to(self.device)

        # Device-side clip bounds (applied per batch in both modes).
        if self.action_clip_lo is not None:
            self._clip_lo_dev = self.action_clip_lo.to(self.device)
            self._clip_hi_dev = self.action_clip_hi.to(self.device)
        else:
            self._clip_lo_dev = self._clip_hi_dev = None

        # Reward stats (computed on the same sample to avoid a full-disk pass).
        self.reward_mean = float(s_rew.mean())
        self.reward_std = float(s_rew.std())
        done_rate = float(s_done.mean())

        print(f"[FQL] Loaded {self.n:,} transitions from {len(paths)} file(s) "
              f"({'disk-backed' if self.keep_on_disk else 'in-RAM fp16'}). "
              f"obs={self.obs_dim} action={self.action_dim}")
        print(f"      obs:    mean=[{self.obs_mean.min().item():+.2f}, {self.obs_mean.max().item():+.2f}]  "
              f"std=[{self.obs_std.min().item():.2f}, {self.obs_std.max().item():.2f}]")
        print(f"      action: mean=[{self.action_mean.min().item():+.2f}, {self.action_mean.max().item():+.2f}]  "
              f"std=[{self.action_std.min().item():.2f}, {self.action_std.max().item():.2f}]  (standardized)")
        print(f"      reward: mean={self.reward_mean:+.4f}  std={self.reward_std:.4f}  "
              f"done_rate={done_rate:.4f}  (stats on {stat_n:,} samples)")

    # ------------------------------------------------------------------
    # Raw sampling — returns unnormalized, unclipped fp32 CPU tensors.
    # ------------------------------------------------------------------
    def _raw_sample(self, n: int):
        if self.keep_on_disk:
            return self._gather_disk(n)
        idx = torch.randint(0, self.n, (n,))
        return (
            self.observations[idx].float(),
            self.next_observations[idx].float(),
            self.actions[idx].float(),
            self.rewards[idx],
            self.dones[idx],
        )

    def _gather_disk(self, n: int):
        """Gather n random transitions straight from the mmap'd files.

        Each sample is assigned to a file with probability proportional to that
        file's (target) size, then a uniform random row within the file is read.
        Rows are sorted per file so disk reads are mostly sequential.
        """
        file_choice = torch.multinomial(self._probs, n, replacement=True)
        obs_p, next_p, act_p, rew_p, done_p = [], [], [], [], []
        for fi in torch.unique(file_choice).tolist():
            k = int((file_choice == fi).sum())
            allowed = self._file_idx[fi]
            if allowed is None:
                rows = torch.randint(0, self._sizes[fi], (k,))
            else:
                rows = allowed[torch.randint(0, allowed.numel(), (k,))]
            rows, _ = torch.sort(rows)
            d = self._mmaps[fi]
            obs_p.append(d["observations"][rows].float())
            next_p.append(d["next_observations"][rows].float())
            act_p.append(d["actions"][rows].float())
            rew_p.append(d["rewards"][rows].float())
            done_p.append(d["dones"][rows].float())
        return (
            torch.cat(obs_p), torch.cat(next_p), torch.cat(act_p),
            torch.cat(rew_p), torch.cat(done_p),
        )

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
        obs, next_obs, act, rew, done = self._raw_sample(batch_size)
        obs = obs.to(self.device, non_blocking=True)
        next_obs = next_obs.to(self.device, non_blocking=True)
        act = act.to(self.device, non_blocking=True)
        rew = rew.to(self.device, non_blocking=True)
        done = done.to(self.device, non_blocking=True)
        if self._clip_lo_dev is not None:
            act = torch.minimum(torch.maximum(act, self._clip_lo_dev), self._clip_hi_dev)
        return FQLBatch(
            obs=self.normalize_obs(obs),
            action=self.normalize_action(act),
            reward=rew,
            next_obs=self.normalize_obs(next_obs),
            done=done,
        )

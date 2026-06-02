"""Events specific to the velocity-tracking task variants.

Provides `reset_root_to_motion_first_frame` — spawns the robot at the motion
clip's first-frame anchor pose (no RSI, no joint alignment). The motion clip
and obstacle mesh are spatially aligned at recording time, so this puts the
robot at a consistent pose where the box is in the expected relative location
for the velocity PPO to drive around. Used by `Velocity-G1-Obstacle-v0`.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Cache the per-anchor-body root pose at the requested frame.
# Keyed by (motion_file_path, anchor_body, frame_idx) -> dict of CPU tensors.
_motion_frame_cache: dict[tuple[str, str, int], dict[str, torch.Tensor]] = {}


def _read_motion_frame(motion_file: str, anchor_body: str, frame_idx: int) -> dict[str, torch.Tensor]:
    """Read the motion clip's anchor-body pose at `frame_idx`.

    Returns:
      - 'root_pos'   (3,) anchor body world position at frame_idx
      - 'root_quat'  (4,) anchor body world quat (wxyz) at frame_idx
    """
    key = (motion_file, anchor_body, frame_idx)
    if key in _motion_frame_cache:
        return _motion_frame_cache[key]
    data = np.load(motion_file, allow_pickle=True)
    if "body_names" not in data.files:
        raise ValueError(f"Motion file '{motion_file}' has no 'body_names' metadata.")
    body_names = list(data["body_names"])
    if anchor_body not in body_names:
        raise ValueError(
            f"anchor_body '{anchor_body}' not in motion '{motion_file}'.  "
            f"Available: {body_names}"
        )
    idx = body_names.index(anchor_body)
    T = data["body_pos_w"].shape[0]
    if frame_idx >= T or frame_idx < 0:
        raise ValueError(f"frame_idx {frame_idx} out of range [0, {T}) for '{motion_file}'.")
    frame = {
        "root_pos":  torch.from_numpy(data["body_pos_w"][frame_idx, idx].astype(np.float32)),
        "root_quat": torch.from_numpy(data["body_quat_w"][frame_idx, idx].astype(np.float32)),
    }
    _motion_frame_cache[key] = frame
    return frame


def reset_root_to_motion_first_frame(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    motion_file: str,
    anchor_body: str = "pelvis",
    frame_idx: int = 10,
    pose_range: dict | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Write only the robot's root pose (position + orientation) from the motion
    clip at `frame_idx`. Joints and root velocity are left to other reset events.

    `frame_idx` defaults to 5 — the first few frames of retargeted motion clips
    can be corrupted (zero/garbage values), so we skip a few to land on a clean
    pose. Matches the tracking env's `FixedStartMotionCommand` spirit (which
    uses 10), just with a tighter offset.

    Args:
        motion_file: Absolute path to a motion .npz.
        anchor_body: Root-link body name (default "pelvis").
        frame_idx: Time-step index in the motion clip (default 5; first frames often corrupted).
        pose_range: Per-dim uniform noise on root pose. Keys: 'x','y','z','roll','pitch','yaw'.
        asset_cfg: Robot asset cfg (default: SceneEntityCfg("robot")).
    """
    frame = _read_motion_frame(motion_file, anchor_body, frame_idx)

    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    n = len(env_ids)

    # Move cached tensors to asset device once.
    if frame["root_pos"].device != device:
        frame = {k: v.to(device) for k, v in frame.items()}
        _motion_frame_cache[(motion_file, anchor_body, frame_idx)] = frame

    base_pos = frame["root_pos"].expand(n, 3).clone()    # (n, 3)
    base_quat = frame["root_quat"].expand(n, 4).clone()  # (n, 4) wxyz

    # One-shot debug print so you can verify exactly what's being written.
    if not getattr(reset_root_to_motion_first_frame, "_logged", False):
        p, q = frame["root_pos"].cpu().tolist(), frame["root_quat"].cpu().tolist()
        yaw = float(np.arctan2(2 * (q[0]*q[3] + q[1]*q[2]), 1 - 2 * (q[2]**2 + q[3]**2)))
        print(f"[reset_root_to_motion_first_frame] motion='{motion_file}' anchor='{anchor_body}' frame={frame_idx}")
        print(f"  pos (xyz)       = ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
        print(f"  quat (wxyz)     = ({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}, {q[3]:+.3f})  |q|={sum(c*c for c in q)**0.5:.4f}")
        print(f"  derived yaw     = {np.rad2deg(yaw):+.1f}°  (heading: {np.cos(yaw):+.3f}, {np.sin(yaw):+.3f})")
        reset_root_to_motion_first_frame._logged = True

    if pose_range:
        ranges = torch.tensor(
            [pose_range.get(k, (0.0, 0.0))
             for k in ("x", "y", "z", "roll", "pitch", "yaw")],
            device=device, dtype=torch.float32,
        )
        rand = sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=device)
        base_pos = base_pos + rand[:, :3]
        delta_q = quat_from_euler_xyz(rand[:, 3], rand[:, 4], rand[:, 5])
        base_quat = quat_mul(delta_q, base_quat)

    # Offset by per-env scene origin so envs spawned across the grid don't collide.
    base_pos = base_pos + env.scene.env_origins[env_ids]

    pose = torch.cat([base_pos, base_quat], dim=-1)
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)

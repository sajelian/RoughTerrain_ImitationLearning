from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from RT_IL.tasks.tracking.mdp.commands import MotionCommand
from RT_IL.tasks.tracking.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str,
                   threshold_xy: float, threshold_z: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    diff = command.anchor_pos_w - command.robot_anchor_pos_w

    # xy: allow more drift for terrain-reactive foot placement
    xy_bad = torch.norm(diff[:, :2], dim=1) > threshold_xy

    # z: keep tight — robot should stay at roughly correct height
    z_bad = torch.abs(diff[:, 2]) > threshold_z

    return xy_bad | z_bad

def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_xy_z(
    env: ManagerBasedRLEnv, command_name: str, threshold_xy: float, threshold_z: float,
    body_names: list[str] | None = None,
    curriculum_name: str | None = None,
) -> torch.Tensor:
    """Terminate when any tracked body deviates too far in XY or Z.

    If ``curriculum_name`` is set, the per-env XY threshold stored by
    :class:`decrease_helper_box_height` overrides ``threshold_xy``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    diff = command.body_pos_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]

    # --- resolve per-env XY threshold from curriculum if available ---
    if curriculum_name is not None:
        idx = env.curriculum_manager._term_names.index(curriculum_name)
        curric = env.curriculum_manager._term_cfgs[idx].func
        if curric.ee_feet_threshold_xy is not None:
            # shape (num_envs,) → (num_envs, 1) for broadcast over bodies
            xy_thresh = curric.ee_feet_threshold_xy.unsqueeze(-1)
        else:
            xy_thresh = threshold_xy
    else:
        xy_thresh = threshold_xy

    xy_bad = torch.norm(diff[..., :2], dim=-1) > xy_thresh
    z_bad = torch.abs(diff[..., 2]) > threshold_z

    return torch.any(xy_bad | z_bad, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)

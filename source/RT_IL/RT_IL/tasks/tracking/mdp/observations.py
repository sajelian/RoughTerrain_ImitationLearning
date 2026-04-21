from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms
from isaaclab.managers import SceneEntityCfg

from RT_IL.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def noised_height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from a noised ray caster sensor w.r.t. the sensor's frame.

    Works with any sensor exposing a `.noisy_data` property (NoisedRayCaster or
    NoisedMultiMeshRayCaster). The offset (default 0.5 m) is subtracted so that
    flat-ground returns are near zero.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.noisy_data.pos_w[:, 2].unsqueeze(1) - sensor.noisy_data.ray_hits_w[..., 2] - offset


def _safe_quat(q: torch.Tensor) -> torch.Tensor:
    """Replace zero quaternions with identity [1,0,0,0] to avoid quat_inv errors."""
    norms = q.pow(2).sum(dim=-1, keepdim=True)
    identity = torch.zeros_like(q)
    identity[..., 0] = 1.0
    return torch.where(norms < 1e-7, identity, q)


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(_safe_quat(command.robot_anchor_quat_w))
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    anchor_quat = _safe_quat(command.robot_anchor_quat_w)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        anchor_quat[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        _safe_quat(command.robot_body_quat_w),
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    anchor_quat = _safe_quat(command.robot_anchor_quat_w)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        anchor_quat[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        _safe_quat(command.robot_body_quat_w),
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        _safe_quat(command.robot_anchor_quat_w),
        command.anchor_pos_w,
        _safe_quat(command.anchor_quat_w),
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        _safe_quat(command.robot_anchor_quat_w),
        command.anchor_pos_w,
        _safe_quat(command.anchor_quat_w),
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference body positions (absolute world frame) expressed in the robot anchor frame.

    Gives the policy/critic explicit per-body targets in a frame it can act on,
    consistent with the absolute world-frame position rewards.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    anchor_quat = _safe_quat(command.robot_anchor_quat_w)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        anchor_quat[:, None, :].repeat(1, num_bodies, 1),
        command.body_pos_w,
        _safe_quat(command.body_quat_w),
    )
    return pos_b.view(env.num_envs, -1)


def motion_body_pos_error_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Absolute world-frame body position error expressed in the robot anchor frame.

    Returns (ref_pos_w - actual_pos_w) rotated into the robot's body frame
    for all tracked bodies, giving the policy the absolute correction vectors
    it needs to reach terrain-locked targets.
    """
    from isaaclab.utils.math import quat_rotate_inverse

    command: MotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    # world-frame error
    error_w = command.body_pos_w - command.robot_body_pos_w
    # rotate into robot anchor frame
    anchor_quat = _safe_quat(command.robot_anchor_quat_w)
    error_b = quat_rotate_inverse(
        anchor_quat[:, None, :].expand(-1, num_bodies, -1).reshape(-1, 4),
        error_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)
    return error_b.view(env.num_envs, -1)


def motion_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference body orientations (absolute world frame) expressed in the robot anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    anchor_quat = _safe_quat(command.robot_anchor_quat_w)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        anchor_quat[:, None, :].repeat(1, num_bodies, 1),
        command.body_pos_w,
        _safe_quat(command.body_quat_w),
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)

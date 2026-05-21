from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms, yaw_quat
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
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    # world-frame error
    error_w = command.body_pos_w - command.robot_body_pos_w
    # rotate into robot anchor frame
    anchor_quat = _safe_quat(command.robot_anchor_quat_w)
    error_b = quat_apply_inverse(
        anchor_quat[:, None, :].expand(-1, num_bodies, -1).reshape(-1, 4),
        error_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)
    return error_b.view(env.num_envs, -1)


def _quantize_joystick(
    speed_ema: torch.Tensor,
    wz: torch.Tensor,
    forward_speed: float,
    turn_slow: float,
    turn_fast: float,
) -> torch.Tensor:
    """Snap smoothed planar speed + continuous wz to the discrete command space.

    speed_ema is an EMA-smoothed norm(vx, vy) so brief torso stillness mid-skill
    does not flip the label back to stop.
    vx_out ∈ {0, forward_speed},  vy_out = 0,  wz ∈ {±turn_fast, ±turn_slow, 0}.
    """
    vx_q = torch.where(speed_ema >= 0.1,
                        torch.full_like(speed_ema, forward_speed),
                        torch.zeros_like(speed_ema))

    slow_thresh = turn_slow * 0.5
    fast_thresh = (turn_slow + turn_fast) * 0.5
    wz_q = torch.where(
        wz > slow_thresh,
        torch.where(wz > fast_thresh, torch.full_like(wz, turn_fast), torch.full_like(wz, turn_slow)),
        torch.where(
            wz < -slow_thresh,
            torch.where(wz < -fast_thresh, torch.full_like(wz, -turn_fast), torch.full_like(wz, -turn_slow)),
            torch.zeros_like(wz),
        ),
    )

    return torch.stack([vx_q, torch.zeros_like(vx_q), wz_q], dim=-1)


def motion_joystick_command(
    env: ManagerBasedEnv,
    command_name: str,
    horizon_s: float = 1.5,
    forward_speed: float = 0.8,
    turn_slow: float = 0.785,   # π/4 rad/s ≈ 45°/s
    turn_fast: float = 1.571,   # π/2 rad/s ≈ 90°/s
    speed_ema_tau_s: float = 1.0,
) -> torch.Tensor:
    """Discrete joystick command (vx, vy=0, omega_yaw) derived from the motion reference.

    Average velocity is (pos[t+H] - pos[t]) / H — temporal integral cancels gait
    oscillation. Planar speed norm is smoothed with an exponential moving average
    (tau=1s) before thresholding so brief torso stillness mid-skill (e.g. leg
    repositioning during box_up) does not create a spurious stop label.
    Quantised to {0, forward_speed} × {±turn_fast, ±turn_slow, 0}.
    Returns shape (num_envs, 3).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion = command.motion

    control_dt = env.cfg.decimation * env.cfg.sim.dt
    H = max(1, int(horizon_s / control_dt))
    T = motion.time_step_total
    raw_anchor_idx = int(motion._body_indexes[command.motion_anchor_body_index].item())

    t = command.time_steps                           # (N,) int64
    future_t = (t + H).clamp(max=T - 1)             # (N,) int64
    actual_dt = (future_t - t).float() * control_dt  # (N,) shorter at clip end

    pos_now = motion._body_pos_w[t, raw_anchor_idx]        # (N, 3)
    pos_fut = motion._body_pos_w[future_t, raw_anchor_idx]  # (N, 3)
    avg_vel_w = (pos_fut - pos_now) / actual_dt.clamp(min=1e-6).unsqueeze(-1)

    q_now = motion._body_quat_w[t, raw_anchor_idx]
    q_fut = motion._body_quat_w[future_t, raw_anchor_idx]
    yaw_now = torch.atan2(
        2.0 * (q_now[:, 0] * q_now[:, 3] + q_now[:, 1] * q_now[:, 2]),
        1.0 - 2.0 * (q_now[:, 2] ** 2 + q_now[:, 3] ** 2),
    )
    yaw_fut = torch.atan2(
        2.0 * (q_fut[:, 0] * q_fut[:, 3] + q_fut[:, 1] * q_fut[:, 2]),
        1.0 - 2.0 * (q_fut[:, 2] ** 2 + q_fut[:, 3] ** 2),
    )
    d_yaw = (yaw_fut - yaw_now + torch.pi) % (2 * torch.pi) - torch.pi
    omega_z = d_yaw / actual_dt.clamp(min=1e-6)

    heading_quat = yaw_quat(_safe_quat(command.robot_anchor_quat_w))
    lin_vel_h = quat_apply_inverse(heading_quat, avg_vel_w)
    speed = (lin_vel_h[:, 0] ** 2 + lin_vel_h[:, 1] ** 2).sqrt()

    # EMA smoothing: bridges brief torso stillness within a skill
    alpha = float(torch.exp(torch.tensor(-control_dt / speed_ema_tau_s)).item())
    if not hasattr(command, '_joystick_speed_ema'):
        command._joystick_speed_ema = speed.clone()
    command._joystick_speed_ema = alpha * command._joystick_speed_ema + (1.0 - alpha) * speed

    return _quantize_joystick(command._joystick_speed_ema, omega_z, forward_speed, turn_slow, turn_fast)


def keyboard_joystick_command(
    env: ManagerBasedEnv,
    forward_speed: float = 0.8,
    turn_slow: float = 0.785,   # π/4 rad/s ≈ 45°/s
    turn_fast: float = 1.571,   # π/2 rad/s ≈ 90°/s
) -> torch.Tensor:
    """Discrete joystick command from keyboard — same command space as training.

    W / Arrow-Up     : forward
    A / Arrow-Left   : gentle left turn  (+turn_slow)
    D / Arrow-Right  : gentle right turn (-turn_slow)
    Q / Z            : sharp left turn   (+turn_fast)
    E / X            : sharp right turn  (-turn_fast)
    L                : stop
    W + A/D/Q/E      : forward + turn simultaneously

    vy = 0 always. Returns shape (num_envs, 3).
    """
    if not hasattr(env, "_teleop_keyboard"):
        import numpy as np
        from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

        kb = Se2Keyboard(Se2KeyboardCfg(
            v_x_sensitivity=1.0,
            v_y_sensitivity=0.0,
            omega_z_sensitivity=1.0,
            sim_device=env.device,
        ))
        # Use raw[0] as forward indicator, raw[2] as turn indicator.
        # Two turn levels: slow (1.0) and fast (2.0) so we can threshold between them.
        kb._INPUT_KEY_MAPPING.update({
            "W":        np.array([1.0, 0.0, 0.0]),
            "UP":       np.array([1.0, 0.0, 0.0]),
            "NUMPAD_8": np.array([1.0, 0.0, 0.0]),
            "A":        np.array([0.0, 0.0, 1.0]),   # slow left
            "LEFT":     np.array([0.0, 0.0, 1.0]),
            "NUMPAD_4": np.array([0.0, 0.0, 1.0]),
            "D":        np.array([0.0, 0.0, -1.0]),  # slow right
            "RIGHT":    np.array([0.0, 0.0, -1.0]),
            "NUMPAD_6": np.array([0.0, 0.0, -1.0]),
            "Z":        np.array([0.0, 0.0, 2.0]),   # fast left
            "NUMPAD_7": np.array([0.0, 0.0, 2.0]),
            "X":        np.array([0.0, 0.0, -2.0]),  # fast right
            "NUMPAD_9": np.array([0.0, 0.0, -2.0]),
        })
        for k in ("S", "DOWN", "NUMPAD_2"):
            kb._INPUT_KEY_MAPPING.pop(k, None)

        env._teleop_keyboard = kb
        print(f"[joystick] W=forward({forward_speed}m/s)  A/D=±{turn_slow:.2f}rad/s  Z/X=±{turn_fast:.2f}rad/s")

    raw = env._teleop_keyboard.advance()  # (3,)
    vx = forward_speed if raw[0].item() > 0.5 else 0.0
    r = raw[2].item()
    if r > 1.5:
        wz = turn_fast
    elif r > 0.5:
        wz = turn_slow
    elif r < -1.5:
        wz = -turn_fast
    elif r < -0.5:
        wz = -turn_slow
    else:
        wz = 0.0
    cmd = torch.tensor([vx, 0.0, wz], dtype=torch.float32, device=env.device)

    env._teleop_step = getattr(env, "_teleop_step", 0) + 1
    if env._teleop_step % 50 == 0:
        print(f"[joystick] raw={raw.tolist()}  →  vx={vx:.2f}  wz={wz:.2f}")

    return cmd.unsqueeze(0).expand(env.num_envs, -1)


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

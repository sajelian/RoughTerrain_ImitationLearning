from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from RT_IL.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_terminal_anchor_goal(
    env: ManagerBasedRLEnv, command_name: str, std: float, final_fraction: float = 0.1
) -> torch.Tensor:
    """Sparse-in-time "success" reward: match the absolute anchor pose in the goal window.

    Fires only during the final ``final_fraction`` of the clip, where the reference is
    spatially meaningful (e.g. standing on the box / past the obstacle). Because the
    error is absolute world-frame xyz, this rewards *arriving at the goal location* even
    if the robot needed an extra adjustment step and is slightly behind schedule — for
    "stand still at the end" clips the target stays put while the robot catches up. This
    is the term the robustness curriculum ramps up to pay for the adjustment step.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    window_start = int((1.0 - final_fraction) * command.motion.time_step_total)
    in_window = (command.time_steps >= window_start).float()
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2) * in_window


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Reward based on absolute world-frame body position error.

    Unlike the relative reward, no yaw/xy anchor transform is applied — the motion
    body positions are compared directly in world space.  This is appropriate for
    foot-hold targets where the exact ground contact location matters.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def foot_contact_schedule(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    std_normalized: float = 0.5,  # in units of body-weight-force (m*g), NOT raw Newtons
    robot_mass: float = 35.0,  # G1 mass in kg
    false_contact_weight: float = 0.7,  # weight for false contact penalty (swing but grounded)
    missed_contact_weight: float = 0.3,  # weight for missed contact penalty (stance but airborne)
) -> torch.Tensor:
    """Reward for matching the reference foot contact schedule.

    Uses net contact force magnitude as a continuous proxy for binary contact state.
    Asymmetric weighting penalizes false contacts (foot down during swing) harder
    than missed contacts (foot up during stance), reflecting their different physical
    consequences for gait stability.

    Forces are normalized by body-weight-force (m*g) before the Gaussian, making
    std_normalized dimensionless and consistent across motion types (walking, stairs,
    long strides etc.). This works because peak contact forces in legged locomotion
    scale with body weight regardless of motion type — so BW is the natural unit.

    std_normalized=0.5 means the reward curve transitions around 0.5 * m*g ≈ 170N
    for the G1. At exactly that force:
        exp(-(0.5)^2 / (0.5)^2) = exp(-1) ≈ 0.37  → sits in the informative middle

    The two reward components:
      - Contact expected  (ref=True):  1 - exp(-F_norm²/std²)  → 0 when no force, 1 when high force
      - No-contact expected (ref=False): exp(-F_norm²/std²)    → 1 when no force, 0 when high force

    Args:
        command_name:          Name of the MotionCommand term.
        sensor_cfg:            ContactSensor config with body_ids for [left_foot, right_foot].
        std_normalized:        Gaussian width in body-weight units. 0.5 = transition at ~half BW.
                               Increase for more leniency, decrease for stricter contact matching.
        robot_mass:            Robot mass in kg, used to compute body-weight-force for normalization.
        false_contact_weight:  Relative penalty for foot grounded during swing phase.
                               Higher = agent punished harder for not lifting feet.
        missed_contact_weight: Relative penalty for foot airborne during stance phase.
                               Lower = agent gets more slack on landing timing.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    schedule = command.foot_contact_schedule
    if schedule is None:
        return torch.ones(env.num_envs, device=env.device)

    ref_left, ref_right, valid_mask = schedule

    # --- Contact force magnitude ---
    # Max over history window for robustness against single-frame noise
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_mag = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)  # (num_envs, history, 2)
        .max(dim=1)[0]  # (num_envs, 2) — max over history
    )

    # --- Normalize forces by body-weight-force ---
    # This makes the reward scale-invariant across motion types:
    # walking peaks ~1.3 BW, stairs descent ~2.0 BW — both handled by the same std_normalized
    GRAVITY = 9.81  # m/s²
    body_weight_force = robot_mass * GRAVITY  # G1: 35 * 9.81 ≈ 343 N
    force_normalized = force_mag / body_weight_force  # dimensionless, typically 0.0–2.0

    # --- Gaussian reward components (now operating on normalized force) ---
    # exp_term → 1 when force≈0 (airborne), → 0 when force is large (grounded)
    exp_term = torch.exp(-force_normalized.square() / (std_normalized**2))

    contact_reward = 1.0 - exp_term  # high reward when foot IS on ground
    no_contact_reward = exp_term  # high reward when foot IS in air

    # ref=True means stance phase (contact expected), ref=False means swing (no contact)
    ref = torch.stack([ref_left, ref_right], dim=-1).float()  # (num_envs, 2)

    # --- Asymmetric penalty scaling ---
    # Base reward: 1 when matching reference, 0 when not
    correct_reward = ref * contact_reward + (1 - ref) * no_contact_reward

    # Scale the penalty asymmetrically: different cost for different failure modes
    penalty = 1.0 - correct_reward
    penalty_weight = ref * missed_contact_weight + (1 - ref) * false_contact_weight
    reward_per_foot = 1.0 - penalty_weight * penalty

    # --- Average across feet, mask invalid frames ---
    reward = reward_per_foot.mean(dim=-1)
    # Invalid frames (no footlock data for this motion clip) get neutral reward 1.0
    reward = torch.where(valid_mask, reward, torch.ones_like(reward))

    return reward


def foot_contact_schedule_sparse(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    std_normalized: float = 0.5,
    robot_mass: float = 35.0,
) -> torch.Tensor:
    """Sparse reward that only fires when feet match the reference contact schedule.

    Returns 0 by default; the agent earns reward only for correct contacts.
    With weight=4.0: both correct → 4.0, one correct → 2.0, both wrong → 0.0.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    schedule = command.foot_contact_schedule
    if schedule is None:
        return torch.zeros(env.num_envs, device=env.device)

    ref_left, ref_right, valid_mask = schedule

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1)
    # max over history confirms a foot landed (peak force); min confirms it's truly airborne
    force_mag_max = force_history.max(dim=1)[0]
    force_mag_min = force_history.min(dim=1)[0]

    GRAVITY = 9.81
    body_weight_force = robot_mass * GRAVITY
    force_norm_max = force_mag_max / body_weight_force
    force_norm_min = force_mag_min / body_weight_force

    # contact_reward uses max: a single peak is enough to confirm ground contact
    # no_contact_reward uses min: foot must be light throughout the window to earn air reward
    contact_reward = 1.0 - torch.exp(-force_norm_max.square() / (std_normalized**2))
    no_contact_reward = torch.exp(-force_norm_min.square() / (std_normalized**2))

    ref = torch.stack([ref_left, ref_right], dim=-1).float()
    correct_reward = ref * contact_reward + (1 - ref) * no_contact_reward

    reward = correct_reward.mean(dim=-1)
    reward = torch.where(valid_mask, reward, torch.zeros_like(reward))

    return reward


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

"""G1 velocity-tracking env, joystick-distillation compatible.

Inherits IsaacLab's standard `G1RoughEnvCfg` (rough terrain curriculum,
locomotion rewards, push perturbations, etc.) and overrides only what we need
for compatibility with our existing FQL collection pipeline:

  1. **Robot**: G1_CYLINDER_CFG with the hybrid action scale, so collected
     action distributions match what the box_up / idle teachers produced.
  2. **Height scanner**: size [1.2, 0.6] @ 0.05 resolution, matching the
     tracking env's MultiMeshRayCasterCfg byte-for-byte. Without this the
     student-obs `height_scan` dim differs and the FQL replay buffer refuses
     to union the datasets.
  3. **`student` obs group**: the joystick-distillation student obs (same
     term order and history lengths as `ObservationsCfg_JOYSTICK_DISTILLATION
     .StudentPolicyCfg` in the tracking package). The joystick term reads
     `velocity_command_joystick` (snaps the env's continuous velocity command
     onto the discrete keyboard grid).
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
    G1RoughEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp_std  # for standard obs funcs
import RT_IL.tasks.velocity.mdp as mdp                                  # for velocity_command_joystick
from RT_IL.tasks.tracking.mdp.observations import skill_mode_constant   # 1-bit toggle (shared)

from RT_IL.robots.g1 import G1_CYLINDER_CFG, G1_HYBRID_ACTION_SCALE


# Mirror the joystick-distillation student history lengths exactly.
PROPIO_HISTORY_SHORT = 5
PROPIO_HISTORY_LONG = 8


@configclass
class StudentPolicyCfg(ObsGroup):
    """Byte-for-byte identical to ObservationsCfg_JOYSTICK_DISTILLATION.StudentPolicyCfg
    in the tracking package, except the joystick comes from the velocity command."""

    joystick = ObsTerm(
        func=mdp.velocity_command_joystick,
        params={"command_name": "base_velocity"},
        history_length=PROPIO_HISTORY_SHORT,
    )
    skill_mode = ObsTerm(
        func=skill_mode_constant,
        params={"value": 0.0},   # velocity-policy data: toggle off
    )
    base_ang_vel = ObsTerm(
        func=mdp_std.base_ang_vel,
        scale=0.2,
        noise=Unoise(n_min=-0.2, n_max=0.2),
        history_length=PROPIO_HISTORY_LONG,
    )
    projected_gravity = ObsTerm(
        func=mdp_std.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
        history_length=PROPIO_HISTORY_SHORT,
    )
    joint_pos = ObsTerm(
        func=mdp_std.joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        history_length=PROPIO_HISTORY_LONG,
    )
    joint_vel = ObsTerm(
        func=mdp_std.joint_vel_rel,
        noise=Unoise(n_min=-0.5, n_max=0.5),
        scale=0.05,
        history_length=PROPIO_HISTORY_LONG,
    )
    actions = ObsTerm(func=mdp_std.last_action, history_length=PROPIO_HISTORY_SHORT)
    height_scan = ObsTerm(
        func=mdp_std.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.02, n_max=0.02),
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class ObservationsCfg_Distillation(ObservationsCfg):
    """Extend the standard ObservationsCfg with the student group."""
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

    # observation groups
    critic: CriticCfg = CriticCfg()
    student: StudentPolicyCfg = StudentPolicyCfg()


@configclass
class G1VelocityDistillationEnvCfg(G1RoughEnvCfg):
    """Training-ready velocity-tracking env with the student obs group attached."""

    observations: ObservationsCfg_Distillation = ObservationsCfg_Distillation()

    def __post_init__(self):
        super().__post_init__()

        # --- Robot: use cylinder-collision G1 + hybrid action scale so collected
        # actions match the box_up / idle teachers' distribution.
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_HYBRID_ACTION_SCALE

        # --- Height scanner: replace standard 1.6x1.0 @ 0.1 res with the
        # tracking env's 1.2x0.6 @ 0.05 res so student-obs dims line up.
        self.scene.height_scanner = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            history_length=0,
            offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.2, 0.6]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
            update_period=self.decimation * self.sim.dt,
        )

        # --- Fix inherited G1Rewards joint references for our G1_CYLINDER_CFG.
        # IsaacLab's G1_MINIMAL_CFG has split elbow_pitch/elbow_roll joints,
        # fingers, and a `torso_joint`. Ours has a single `*_elbow_joint`, no
        # fingers, and waist_{yaw,roll,pitch}_joint instead. Without these
        # overrides the reward manager errors out on joint-name regex match.
        self.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
        )
        self.rewards.joint_deviation_fingers = None  # type: ignore[assignment]
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        )


@configclass
class G1VelocityDistillationEnvCfg_PLAY(G1RoughEnvCfg_PLAY):
    """PLAY variant — fewer envs, terrain curriculum off, viz robot if desired."""

    observations: ObservationsCfg_Distillation = ObservationsCfg_Distillation()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_HYBRID_ACTION_SCALE
        self.scene.height_scanner = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            history_length=0,
            offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.2, 0.6]),
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
            update_period=self.decimation * self.sim.dt,
        )

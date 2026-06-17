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

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
    G1Rewards,
    G1RoughEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg, TerminationsCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp_std  # for standard obs funcs
import RT_IL.tasks.velocity.mdp as mdp                                  # for velocity_command_joystick
from RT_IL.tasks.tracking.mdp.observations import skill_mode_constant   # 1-bit toggle (shared)

from RT_IL.robots.g1 import G1_CYLINDER_CFG, G1_HYBRID_ACTION_SCALE


# Mirror the joystick-distillation student history lengths exactly.
PROPIO_HISTORY_SHORT = 5
PROPIO_HISTORY_LONG = 8


@configclass
class CriticCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
    actions = ObsTerm(func=mdp.last_action)
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )


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
    # observation groups
    critic: CriticCfg = CriticCfg()
    student: StudentPolicyCfg = StudentPolicyCfg()


@configclass
class RewardsCfg_TRAIN(G1Rewards):
    """Reward terms for the MDP.

    Overrides joint-name regexes in G1Rewards to match our G1_CYLINDER_CFG /
    main.urdf. IsaacLab's G1_MINIMAL_CFG has split `*_elbow_{pitch,roll}_joint`,
    finger joints, and a single `torso_joint`. Ours has a single `*_elbow_joint`,
    `*_wrist_{roll,pitch,yaw}_joint` instead of fingers, and
    `waist_{yaw,roll,pitch}_joint` instead of `torso_joint`. Without these
    overrides the reward manager errors out on joint-name regex match.
    """

    # Replace fingers reward (no fingers on our robot) with wrist deviation.
    joint_deviation_fingers = None
    joint_deviation_wrist = RewTerm(
        func=mdp_std.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
            )
        },
    )

    feet_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            },
    )
    
    stand_still = RewTerm(
        func=mdp.joint_pos_stand_still, weight=-1.0, params={"command_name": "base_velocity", "threshold": 0.05}
    )

    def __post_init__(self):
        self.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
        )
        self.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        )

        self.lin_vel_z_l2.weight = -0.1
        self.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*")
        self.feet_air_time = None


@configclass
class TerminationsCfg_TRAIN(TerminationsCfg):
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.80})
    fall_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "pelvis",
                    "torso_link",
                    ".*_hip_pitch_link",
                    ".*_hip_roll_link",
                    ".*_hip_yaw_link",
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class G1VelocityDistillationEnvCfg(G1RoughEnvCfg):
    """Training-ready velocity-tracking env with the student obs group attached."""

    observations: ObservationsCfg_Distillation = ObservationsCfg_Distillation()
    rewards: RewardsCfg_TRAIN = RewardsCfg_TRAIN()
    terminations: TerminationsCfg_TRAIN = TerminationsCfg_TRAIN()

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
        self.terminations.base_contact = None

        # change here to collect different command distributions
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class G1VelocityObstacleEnvCfg(G1VelocityDistillationEnvCfg):
    """Velocity-tracking env with an obstacle prim — used to collect velocity-PPO
    rollouts on the *same spatial layout* the skill experts were trained on.

    Key differences from the parent:
      * Flat terrain (replaces ROUGH_TERRAINS_CFG). The skill experts were
        trained on flat ground + obstacle; matching their distribution at
        collection time means the student gets coherent obs across both
        skill_mode=0 (this env) and skill_mode=1 (tracking env) data sources.
      * Adds an `obstacle` AssetBaseCfg prim with the same swap mechanism as
        the tracking env's G1MeshEnvCfg (play_fql.py / collect scripts use
        `--usd_name` to set the USD at runtime).
      * Height scanner now ray-casts against the obstacle too, so the obstacle
        shows up in the student's heightmap obs.
      * Reset event swapped: instead of random base pose, robot spawns at the
        motion clip's first-frame anchor pose. The collect script wires the
        motion file into `events.reset_base.params['motion_file']` from the
        same `--registry_name` it uses for the tracking env's MotionCommand.
    """

    obstacle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/obstacle",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",  # filled at runtime via --usd_name
            scale=(1.0, 1.0, 1.0),
            visual_material=sim_utils.materials.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.2, 0.2),
                emissive_color=(0.0, 0.0, 0.0),
                roughness=0.5,
                metallic=0.5,
                opacity=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        collision_group=0,
    )

    def __post_init__(self):
        super().__post_init__()

        # --- Replace rough terrain with flat (match skill expert's training) ---
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                size=(100.0, 100.0),
                num_rows=1,
                num_cols=1,
                border_width=0.0,
                sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg()},
            ),
            use_terrain_origins=False,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
        )

        # --- Attach obstacle + extend height scanner to ray-cast it ---
        self.scene.obstacle = self.obstacle
        self.scene.height_scanner = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            history_length=0,
            offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.2, 0.6]),
            debug_vis=False,
            mesh_prim_paths=[
                "/World/ground",
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/obstacle",
                    is_shared=True,
                    track_mesh_transforms=False,
                ),
            ],
            update_period=self.decimation * self.sim.dt,
        )

        # --- Swap reset event: write only the root pose from motion frame 0 ---
        # The parent's `reset_robot_joints` event still runs and sets joints to
        # the default standing pose with noise (untouched).
        from dataclasses import MISSING as _MISSING
        self.events.reset_base = EventTerm(
            func=mdp.reset_root_to_motion_first_frame,
            mode="reset",
            params={
                "motion_file": _MISSING,
                "anchor_body": "pelvis",  # root link, NOT torso
                "pose_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (0.0, 0.02),
                    "roll": (-0.05, 0.05),
                    "pitch": (-0.05, 0.05),
                    "yaw": (-0.1, 0.1),
                },
            },
        )

        # --- Disable curriculum that requires rough-terrain levels ---
        # The parent inherits LocomotionVelocityRoughEnvCfg.CurriculumCfg which
        # has terrain_levels = CurrTerm(func=mdp.terrain_levels_vel). Our flat
        # terrain has no terrain_levels attribute, so the curriculum would crash
        # at the first reset.
        if hasattr(self, "curriculum") and self.curriculum is not None:
            self.curriculum.terrain_levels = None  # type: ignore[attr-defined]

        self.scene.env_spacing = 3.5
        self.episode_length_s = 2.0

@configclass
class G1VelocityDistillationEnvCfg_PLAY(G1RoughEnvCfg_PLAY):
    """PLAY variant — fewer envs, terrain curriculum off, viz robot if desired."""

    rewards: RewardsCfg_TRAIN = RewardsCfg_TRAIN()
    observations: ObservationsCfg_Distillation = ObservationsCfg_Distillation()
    terminations: TerminationsCfg_TRAIN = TerminationsCfg_TRAIN()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_HYBRID_ACTION_SCALE
        self.scene.height_scanner = MultiMeshRayCasterCfg(  # TODO replace this with noised multi ray down the road
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            history_length=0,
            offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.2, 0.6]),
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
            update_period=self.decimation * self.sim.dt,
        )
        self.terminations.base_contact = None

        # self.commands.base_velocity.ranges.lin_vel_x = (-0.0, 0.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        # TODO collect 1e6 standing still
        # TODO collect 5e6 random walking
        # TODO train fql student
        # TODO evaluate behaviour (should stop when no command given, walk when command given and not fall over (espcially not on flat terrain))
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Pre-defined configs
##
from RT_IL.robots.g1 import (
    G1_ACTION_SCALE,
    G1_CYLINDER_CFG,
    G1_HYBRID_ACTION_SCALE,
)

from isaaclab.sensors import RayCasterCfg
import RT_IL.tasks.velocity.mdp as mdp
from isaaclab.sensors import patterns
from RT_IL.mbrl.sensors import NoisedRayCasterCfg


@configclass
class ObservationsCfg_TRAIN(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

    # observation groups
    critic: CriticCfg = CriticCfg()


@configclass
class ObservationsCfg_DISTILLATION(ObservationsCfg_TRAIN):
    """Observation specifications for the MDP."""

    @configclass
    class StudentPolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # LOW NOISE SETUP: this should recover the teacher (deployable in sim, but not in real)
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=5)
        # projected_gravity = ObsTerm(
        #     func=mdp.projected_gravity,
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        #     history_length=5,
        # )
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, history_length=5)
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=5)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05, history_length=5)
        # actions = ObsTerm(func=mdp.last_action, history_length=5)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        #     history_length=1,
        # )
        # NOISED DISTILL SETUP (deployable in sim and real)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.25, n_max=0.25),
            history_length=5,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=5,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=5,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.02, n_max=0.02),
            history_length=5,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-2.0, n_max=2.0),
            scale=0.05,
            history_length=5,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=5)
        height_scan = ObsTerm(
            func=mdp.noised_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            history_length=1,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    student: StudentPolicyCfg = StudentPolicyCfg()


@configclass
class RewardsCfg_TRAIN(RewardsCfg):
    """Reward terms for the MDP."""

    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"],
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
        func=mdp.joint_pos_stand_still,
        weight=-2.0,
        params={"command_name": "base_velocity", "threshold": 0.05},
    )

    def __post_init__(self):
        self.lin_vel_z_l2.weight = -0.1
        self.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names="(?!.*ankle.*).*"
        )
        self.feet_air_time = None


@configclass
class TerminationsCfg_TRAIN(TerminationsCfg):
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})

    def __post_init__(self):
        self.base_contact = None


@configclass
class UnitreeG1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):

    observations: ObservationsCfg_TRAIN = ObservationsCfg_TRAIN()
    rewards: RewardsCfg_TRAIN = RewardsCfg_TRAIN()
    terminations: TerminationsCfg_TRAIN = TerminationsCfg_TRAIN()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        self.observations.policy.base_ang_vel.scale = 0.2
        self.observations.policy.joint_vel.scale = 0.05
        self.actions.joint_pos.scale = (
            0.25  # this needs to match g1_rsl deployment config yaml
        )
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.15  # 15% of envs train with zero command

        # override events
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.base_com = None
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.3, 0.3),
            "z": (-0.2, 0.2),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.2, 0.2),
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (-1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = (
            "torso_link"
        )
        self.events.push_robot.interval_range_s = (5.0, 5.0)


@configclass
class UnitreeG1RoughEnvCfg_PLAY(UnitreeG1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class UnitreeG1RoughEnvCfg_DISTILLATION(UnitreeG1RoughEnvCfg):

    observations: ObservationsCfg_DISTILLATION = ObservationsCfg_DISTILLATION()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.height_scanner = NoisedRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )


@configclass
class UnitreeG1RoughEnvCfg_DISTILLATIONPLAY(UnitreeG1RoughEnvCfg_DISTILLATION):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        #self.commands.base_velocity.ranges.lin_vel_x = (0, 0)
        #self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        #self.commands.base_velocity.ranges.heading = (0, 0)

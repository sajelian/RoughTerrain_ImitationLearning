from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
import isaaclab.terrains as terrain_gen
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab.sensors.ray_caster import RayCasterCfg, MultiMeshRayCasterCfg, patterns
from RT_IL.mbrl.sensors import NoisedRayCasterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import numpy as np

import RT_IL.tasks.tracking.mdp as mdp
from RT_IL.robots.g1 import G1_FOOT_LINKS, G1_HAND_LINKS

from isaaclab.terrains.trimesh.utils import make_plane

GROUND_HEIGHT = -0.25


def lowered_flat_terrain(difficulty, cfg):
    """Flat terrain at GROUND_HEIGHT instead of z=0."""
    origin = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, GROUND_HEIGHT)
    plane_mesh = make_plane(cfg.size, GROUND_HEIGHT, center_zero=False)
    return [plane_mesh], np.array(origin)


VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

# Smaller push range: absolute tracking means large pushes cause irrecoverable
# xy drift from the reference path with no reset, so keep perturbations small.
PUSH_VELOCITY_RANGE = {
    "x": (-0.1, 0.1),
    "y": (-0.1, 0.1),
    "z": (-0.05, 0.05),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.15, 0.15),
}


RESET_VELOCITY_RANGE = {
    "x": (-0.2, 0.2),
    "y": (-0.15, 0.15),   # lateral more harmful for xy drift, keep tighter
    "z": (-0.1, 0.1),
    "roll": (-0.2, 0.2),
    "pitch": (-0.2, 0.2),
    "yaw": (-0.3, 0.3),
}


PROPIO_HISTORY_SHORT = 5
PROPIO_HISTORY_LONG = 8


##
# Scene definition
##
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(100.0, 100.0),
            num_rows=1,
            num_cols=1,
            border_width=0.0,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(function=lowered_flat_terrain),
            },
        ),
        use_terrain_origins=False,
        env_spacing=2.5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    viz_robot: ArticulationCfg | None = None
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )

    obstacle: AssetBaseCfg = MISSING

    height_scanner = MultiMeshRayCasterCfg(
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
    )


##
# MDP settings
##
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.00, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        },
        velocity_range=RESET_VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        start_xy_noise_std=0.1,  # keep this < 0.3 or episode will terminate immediately
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=True
    )


@configclass
class ObservationsCfg_TRAIN:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "motion"}
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.1, n_max=0.1))
        actions = ObsTerm(func=mdp.last_action)

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "motion"}
        )
        root_pos = ObsTerm(
            func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        root_quat = ObsTerm(
            func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
        )
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        motion_body_pos_b = ObsTerm(
            func=mdp.motion_body_pos_b, params={"command_name": "motion"}
        )
        motion_body_ori_b = ObsTerm(
            func=mdp.motion_body_ori_b, params={"command_name": "motion"}
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        ankle_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=G1_FOOT_LINKS,
                )
            },
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ObservationsCfg_DEPLOY:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "motion"}
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=PROPIO_HISTORY_SHORT,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=PROPIO_HISTORY_LONG,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=PROPIO_HISTORY_SHORT,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=PROPIO_HISTORY_LONG,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            scale=0.05,
            history_length=PROPIO_HISTORY_LONG,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=PROPIO_HISTORY_SHORT)
        height_scan = ObsTerm(
            func=mdp.noised_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0)
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "motion"}
        )
        root_pos = ObsTerm(
            func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        root_quat = ObsTerm(
            func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
        )
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        motion_body_pos_b = ObsTerm(
            func=mdp.motion_body_pos_b, params={"command_name": "motion"}
        )
        motion_body_ori_b = ObsTerm(
            func=mdp.motion_body_ori_b, params={"command_name": "motion"}
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        ankle_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=G1_FOOT_LINKS,
                )
            },
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class ObservationsCfg_DISTILLATION(ObservationsCfg_TRAIN):
    """Observation specifications for the MDP."""

    @configclass
    class StudentPolicyCfg(ObsGroup):
        """Observations for policy group."""

        command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "motion"}
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=PROPIO_HISTORY_SHORT,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=PROPIO_HISTORY_LONG,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=PROPIO_HISTORY_SHORT,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=PROPIO_HISTORY_LONG,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            scale=0.05,
            history_length=PROPIO_HISTORY_LONG,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=PROPIO_HISTORY_SHORT)
        height_scan = ObsTerm(
            func=mdp.noised_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0)
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    student: StudentPolicyCfg = StudentPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.9),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": PUSH_VELOCITY_RANGE},
    )

    foot_perturbation: EventTerm | None = None


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    foot_contact_schedule = RewTerm(
        func=mdp.foot_contact_schedule_sparse,
        weight=2.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=G1_FOOT_LINKS,
            ),
            "std_normalized": 0.25,
            "robot_mass": 35.0,
        },
    )

    motion_global_feet_body_pos = RewTerm(
        func=mdp.motion_global_body_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.3,
            "body_names": G1_FOOT_LINKS,
        },
    )

    motion_global_hand_body_pos = RewTerm(
        func=mdp.motion_global_body_position_error_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            "std": 0.25,
            "body_names": G1_HAND_LINKS,
        },
    )

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=2.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 3.14},
    )
    # TODO check if this helps remove the shaky behavior
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.2)
    l1_joint_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos,
        params={
            "command_name": "motion",
            "threshold_xy": 0.35,
            "threshold_z": 0.22,
        },  # full 3D norm: catches xy drift under absolute tracking
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": 0.8,
        },
    )
    ee_feet_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_xy_z,
        params={
            "command_name": "motion",
            "threshold_xy": 0.35,
            "threshold_z": 0.35,
            "body_names": G1_FOOT_LINKS,
        },
    )
    ee_hand_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.8,
            "body_names": G1_HAND_LINKS,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##
@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg_TRAIN = ObservationsCfg_TRAIN()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 4.0  # this should be <= clip length
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**17
        # depends on mesh -> needs to scale with num_envs
        self.sim.physx.gpu_collision_stack_size = (
            400425504  #  2.5 * 2**15 * self.scene.num_envs
        )
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

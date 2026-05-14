import torch

from RT_IL.mbrl.sensors import NoisedMultiMeshRayCasterCfg
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors.ray_caster.patterns import patterns
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm

from RT_IL.robots.g1 import G1_CYLINDER_CFG, G1_HYBRID_ACTION_SCALE, G1_VIZ_CFG
from RT_IL.tasks.tracking.tracking_env_cfg import ObservationsCfg_DISTILLATION, ObservationsCfg_DEPLOY, TrackingEnvCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, MultiMeshRayCasterCfg


from isaaclab.assets import AssetBaseCfg

import isaaclab.sim as sim_utils
from RT_IL.tasks.tracking.mdp.commands import MotionCommand
import RT_IL.tasks.tracking.mdp as mdp


class FixedStartMotionCommand(MotionCommand):
    """Motion command variant that always restarts trajectories from the first frame."""

    def _adaptive_sampling(self, env_ids):
        if len(env_ids) == 0:
            return
        self.time_steps[env_ids] = 0


@configclass
class G1MeshEnvCfg(TrackingEnvCfg):

    obstacle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/obstacle",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            scale=(1.0, 1.0, 1.0),
            visual_material=sim_utils.materials.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.2, 0.2),
                emissive_color=(0.0, 0.0, 0.0),
                roughness=0.5,
                metallic=0.5,
                opacity=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        collision_group=0,
    )

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_HYBRID_ACTION_SCALE
        self.scene.env_spacing = 7.5
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

        self.scene.obstacle = self.obstacle
        # self.commands.motion.class_type = FixedStartMotionCommand


@configclass
class G1MimicMeshWoStateEstimationEnvCfg(G1MeshEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1MimicMeshEnvCfg_PLAY(G1MeshEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.viz_robot = G1_VIZ_CFG.replace(prim_path="{ENV_REGEX_NS}/VizRobot")
        self.scene.height_scanner.debug_vis = True
        # Viewer options for filming whole motions externally
        #self.viewer.eye = (0.9, 0.5, 2.4)
        #self.viewer.lookat = (0.0, -4.7, 0.0)
        #self.viewer.origin_type = "asset_root"
        self.viewer.eye = (2.5, 2.5, 2.5)
        self.viewer.resolution = (1920, 1080)
        # always start from beginning of motion
        self.commands.motion.class_type = FixedStartMotionCommand
        # reduce termination thresholds to show failures more clearly
        #self.terminations.anchor_pos = None
        # self.terminations.anchor_ori = None
        #self.terminations.ee_feet_pos = None
        #self.terminations.ee_hand_pos = None
        #self.events.push_robot = None
        # self.commands.motion.start_xy_noise_std = 0.0
        # self.episode_length_s = 7.0  # some motions are longer than default 10s


@configclass
class G1MeshEnvDeployCfg(G1MeshEnvCfg):
    """Environtment config for deployment of G1 mesh tracking policy on real robot.
    
    Removes observations that require state estimation and adds sim-to-real randomization.

    Args:
        G1MeshEnvCfg (_type_): _description_
    """
    observations: ObservationsCfg_DEPLOY = ObservationsCfg_DEPLOY()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.height_scanner = NoisedMultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.8]), # !!! flipped dimension for sideways beam!
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

        DEPLOY_PUSH_VELOCITY_RANGE = {
            "x": (-0.10, 0.10),
            "y": (-0.10, 0.10),
            "z": (-0.03, 0.05),
            "roll": (-0.08, 0.08),
            "pitch": (-0.08, 0.08),
            "yaw": (-0.15, 0.15),
        }

        self.events.push_robot.params["velocity_range"] = DEPLOY_PUSH_VELOCITY_RANGE
        self.events.push_robot.interval_range_s = (1.0, 7.0)

        self.events.add_joint_default_pos.params["pos_distribution_params"] = (-0.05, 0.05)

        self.events.physics_material.params["static_friction_range"] = (0.15, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.10, 1.0)

        self.events.foot_perturbation = EventTerm(
            func=mdp.RandomFootWrench,
            mode="interval",
            interval_range_s=(0.02, 0.02),  # evaluated every RL step
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                ),
                "force_range": {
                    "x": (-60.0, 60.0),
                    "y": (-60.0, 60.0),
                    "z": (-10.0, 30.0),
                },
                "torque_range": {
                    "x": (-6.0, 6.0),
                    "y": (-6.0, 6.0),
                    "z": (-6.0, 6.0),
                },
                "event_interval_range_s": (1.0, 3.0),
                "duration_range_s": (0.05, 0.20),
            },
        )

        self.events.actuator_gains_startup = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "operation": "scale",
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.5, 2.0),
                "distribution": "uniform",
            },
        )

        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link*"),
                "mass_distribution_params": (-2.5, 2.5),
                "operation": "add",
            },
        )

        self.commands.motion.pose_range = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.00, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.45, 0.45),
        }

        self.commands.motion.start_xy_noise_std = 0.125


@configclass
class G1MimicMeshEnvDeployCfg_PLAY(G1MeshEnvDeployCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.viz_robot = G1_VIZ_CFG.replace(prim_path="{ENV_REGEX_NS}/VizRobot")
        self.scene.height_scanner.debug_vis = True
        # Viewer options for filming whole motions externally
        #self.viewer.eye = (0.9, 0.5, 2.4)
        #self.viewer.lookat = (0.0, -4.7, 0.0)
        #self.viewer.origin_type = "asset_root"
        self.viewer.eye = (2.5, 2.5, 2.5)
        self.viewer.resolution = (1920, 1080)
        # always start from beginning of motion
        self.commands.motion.class_type = FixedStartMotionCommand
        # reduce termination thresholds to show failures more clearly
        # self.terminations.anchor_pos = None
        self.events.foot_perturbation = None
        # self.terminations.anchor_ori = None
        self.terminations.ee_feet_pos = None
        self.terminations.ee_hand_pos = None
        self.events.push_robot = None
        # self.commands.motion.start_xy_noise_std = 0.0
        self.episode_length_s = 12.0  # some motions are longer than default 10s

@configclass
class G1MeshEnvCfg_DISTILLATION(G1MeshEnvCfg):

    observations: ObservationsCfg_DISTILLATION = ObservationsCfg_DISTILLATION()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.height_scanner = NoisedMultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
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


        # TODO: consider relaxing termination thresholds for student training
        # self.terminations.anchor_pos.params["threshold_xy"] = 1.0
        # self.terminations.anchor_pos.params["threshold_z"] = 0.4


@configclass
class G1MeshEnvCfg_DISTILLATIONPLAY(G1MeshEnvCfg_DISTILLATION):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.commands.motion.debug_vis = False
        self.scene.viz_robot = G1_VIZ_CFG.replace(prim_path="{ENV_REGEX_NS}/VizRobot")

        # Viewer options for filming whole motions externally
        #self.viewer.eye = (0.9, 0.5, 2.4)
        #self.viewer.lookat = (0.0, -4.7, 0.0)
        #self.viewer.origin_type = "asset_root"
        self.viewer.eye = (2.5, 2.5, 2.5)
        self.viewer.resolution = (1920, 1080)
        # always start from beginning of motion
        self.commands.motion.class_type = FixedStartMotionCommand
        self.commands.motion.start_xy_noise_std = 0.0
        # reduce termination thresholds to show failures more clearly
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_feet_pos = None
        self.terminations.ee_hand_pos = None
        self.events.push_robot = None
import torch

from RT_IL.mbrl.sensors import NoisedMultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns import patterns
from isaaclab.utils import configclass

from RT_IL.robots.g1 import G1_CYLINDER_CFG, G1_HYBRID_ACTION_SCALE, G1_VIZ_CFG
from RT_IL.tasks.tracking.tracking_env_cfg import ObservationsCfg_DISTILLATION, TrackingEnvCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, MultiMeshRayCasterCfg, patterns


from isaaclab.assets import AssetBaseCfg

import isaaclab.sim as sim_utils
from RT_IL.tasks.tracking.mdp.commands import MotionCommand
import RT_IL.tasks.tracking.mdp as mdp


class FixedStartMotionCommand(MotionCommand):
    """Motion command variant that always restarts trajectories from the first frame."""

    def _adaptive_sampling(self, env_ids):
        if len(env_ids) == 0:
            return
        self.time_steps[env_ids] = 10


class BiasedStartMotionCommand(MotionCommand):
    """Motion command variant with fixed biased sampling: 50% of clips start in the first
    25% of frames, 50% start in the remaining 75%."""

    def _adaptive_sampling(self, env_ids):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        total = max(self.motion.time_step_total - 1, 1)
        n_early = (n + 1) // 2  # ceil(n/2) go to first 25%
        n_late = n - n_early
        early = (torch.rand(n_early, device=self.device) * 0.25 * total).long()
        late = (0.25 * total + torch.rand(n_late, device=self.device) * 0.75 * total).long()
        self.time_steps[env_ids] = torch.cat([early, late]).clamp(0, total)


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
        self.episode_length_s = 7.0  # some motions are longer than default 10s


@configclass
class G1MeshEnvCfg_DISTILLATION(G1MeshEnvCfg):

    observations: ObservationsCfg_DISTILLATION = ObservationsCfg_DISTILLATION()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.commands.motion.class_type = BiasedStartMotionCommand

        self.scene.height_scanner = NoisedMultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=True,
            mesh_prim_paths=[
                "/World/ground",
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/obstacle",
                    is_shared=True,
                    track_mesh_transforms=False,
                ),
            ],
        )


@configclass
class G1MeshEnvCfg_DISTILLATIONPLAY(G1MeshEnvCfg_DISTILLATION):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.commands.motion.debug_vis = False
        self.scene.viz_robot = None # G1_VIZ_CFG.replace(prim_path="{ENV_REGEX_NS}/VizRobot")

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
        self.episode_length_s = 7.0  # some motions are longer than default 10s

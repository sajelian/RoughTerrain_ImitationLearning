from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG, G1_VIZ_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

from isaaclab.assets import AssetBaseCfg

import isaaclab.sim as sim_utils
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand


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
                diffuse_color=(0.4, 0.4, 0.4),
                emissive_color=(0.0, 0.0, 0.0),
                roughness=0.5,
                metallic=0.5,
                opacity=1.0
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        collision_group=0,  # not really important
    )

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.scene.env_spacing = 5.0
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


@configclass
class G1MimicMeshEnvCfg_PLAY(G1MeshEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.debug_vis = False
        self.scene.viz_robot = G1_VIZ_CFG.replace(prim_path="{ENV_REGEX_NS}/VizRobot")
        # self.viewer.eye = (6.0, 6.0, 6.0)
        self.commands.motion.class_type = FixedStartMotionCommand
        self.terminations = None

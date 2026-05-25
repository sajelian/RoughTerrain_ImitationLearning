"""G1 deploy env — robot + obstacle + matching action scale + viz robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from RT_IL.robots.g1 import G1_CYLINDER_CFG, G1_HYBRID_ACTION_SCALE, G1_VIZ_CFG
from RT_IL.tasks.deploy.deploy_env_cfg import DeployEnvCfg


@configclass
class G1DeployEnvCfg(DeployEnvCfg):
    """Deploy env wired for G1 + obstacle USD swap (play_fql.py uses --usd_name).

    Obstacle prim defaults to an empty USD; play_fql.py overwrites the
    `usd_path` at runtime via env_cfg.scene.obstacle.replace(...). Same
    pattern as the tracking env collection script.
    """

    obstacle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/obstacle",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",  # filled at runtime
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
        self.scene.obstacle = self.obstacle

        # Reasonable viewer pose for single-env play.
        self.scene.env_spacing = 7.5
        self.viewer.eye = (2.5, 2.5, 2.5)
        self.viewer.resolution = (1920, 1080)
        # Viz robot is optional — comment out if you don't want the ghost.
        self.scene.viz_robot = G1_VIZ_CFG.replace(prim_path="{ENV_REGEX_NS}/VizRobot")

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from RT_IL.tasks.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg_PLAY

from isaaclab.assets import AssetBaseCfg

import isaaclab.sim as sim_utils


@configclass
class G1MeshTerrainCfg_PLAY(G1RoughEnvCfg_PLAY):

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
        # post init of parent
        super().__post_init__()

        self.scene.obstacle = self.obstacle

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 7.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # self.scene.robot.init_state.pos = (-3.5, 0, 0.76)
        self.curriculum = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.75, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)  # this will not be used if heading_command is True
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)  # walk toward x direction
        self.commands.base_velocity.rel_standing_envs = 0.0

        self.events.reset_base.params = {  # robot should always spawn determinstically, otherwise alignment with terrain is hard
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

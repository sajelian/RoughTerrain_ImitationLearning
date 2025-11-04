# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from whole_body_tracking.tasks.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg_PLAY

from isaaclab.assets import AssetBaseCfg

from isaaclab.sensors import RayCasterCfg, MultiMeshRayCasterCfg, patterns

import isaaclab.sim as sim_utils


@configclass
class G1MeshTerrainCfg(G1RoughEnvCfg_PLAY):

    obstacle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/obstacle",  # TODO make this cli arg
        spawn=sim_utils.UsdFileCfg(
            usd_path="meshes/step.usd",
            scale=(1.0, 3.0, 1.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.25, -0.25, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        collision_group=0
    )

    height_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=True,
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                target_prim_expr="{ENV_REGEX_NS}/obstacle",
                is_shared=True,
                track_mesh_transforms=False,
            ),
        ],
    )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.obstacle = self.obstacle
        self.scene.height_scanner = self.height_scanner

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

"""Helper script to visualize terrains in Isaac Sim and adapt positions according to robot. Mesh must be in USD format.
Example usage: python scripts/terrain_viz_usd_multi_ray.py --usd_file meshes/step.usd --num_envs=2
"""

"""Launch Isaac Sim Simulator first."""
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument(
    "--usd_file",
    default=None,
    type=str,
    help="The name of the usd file.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sensors import RayCasterCfg, MultiMeshRayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg

from isaaclab.assets import ArticulationCfg
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class TerrainVisualizationSceneCfg(InteractiveSceneCfg):
    """Configuration for a terrain visualization scene."""

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
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

    obstacle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/obstacle",
        spawn=sim_utils.UsdFileCfg(usd_path="meshes/step.usd", scale=(1.0, 3.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.2, 0.0, 0.0)),
        collision_group=0,
    )

    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Simulation loop
    while simulation_app.is_running():
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)
    scene_cfg = TerrainVisualizationSceneCfg(
        num_envs=args_cli.num_envs, env_spacing=5.0
    )
    scene = InteractiveScene(scene_cfg)
    # Run the simulator
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

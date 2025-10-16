""" Helper script to visualize terrains in Isaac Sim and adapt positions according to robot.
    Example usage: python scripts/terrain_viz.py --num_envs 1 --mesh_file meshes/crater.stl
"""

"""Launch Isaac Sim Simulator first."""
import argparse

from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--mesh_file", default="meshes/crater.stl", type=str, help="The name of the mesh file.")

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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from isaaclab.assets import ArticulationCfg, Articulation
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

import trimesh

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

    ground = TerrainImporterCfg(
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

    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


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

    scene_cfg = TerrainVisualizationSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    # Meshinger
    meshes = [trimesh.load_mesh(args_cli.mesh_file) for _ in range(scene_cfg.num_envs)]
    for i, mesh in enumerate(meshes):  # seems pretty slow, but only done once
        mesh.apply_scale(0.010)
        t = scene.env_origins[i].cpu().numpy()
        # TODO these numbers are mesh specific. how will we handle this with the real data? Do we care about centering?
        # probably not because the motion will include the robot going to the correct starting point either way
        t[0] += 0.8
        t[1] -= 0.215
        t[2] = -0.075
        mesh.apply_translation(t)
        scene.terrain.import_mesh(f"mesh_{i}", mesh)  # the crater index is not matching the env index, do we care?

    # Run the simulator
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

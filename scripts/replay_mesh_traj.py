"""Script to replay motion loaded from a wandb registry with mesh terrain.
Example usage:
python scripts/replay_mesh_traj.py --registry_name=sajelian-eth-zurich-org/wandb-registry-Motions/step_traversal --usd_name sequence_07
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch
import yaml

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")
parser.add_argument(
    "--usd_name", type=str, default=None, help="Name of the USD mesh."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# USD import and config
usd_folder = "source/RT_IL/RT_IL/assets/usd/"
usd_path = usd_folder + f"{args_cli.usd_name}/" + f"{args_cli.usd_name}" + ".usd"
usd_cfg_path = usd_folder + f"{args_cli.usd_name}/" + "config.yaml"


# Fetch config from yaml
with open(usd_cfg_path, "r") as f:
    usd_cfg: dict = yaml.load(f, Loader=yaml.FullLoader)

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from RT_IL.robots.g1 import G1_CYLINDER_CFG
from RT_IL.tasks.tracking.mdp import MotionLoader
from isaaclab.terrains import TerrainImporterCfg


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

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

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    registry_name = args_cli.registry_name
    if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
        registry_name += ":latest"
    import pathlib

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Simulation loop

    sim.set_camera_view((-3.5, 0, 2.0), (0, -1, 0.1))
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        #pos_lookat = root_states[0, :3].cpu().numpy()
        #sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)

    scene_cfg.obstacle = scene_cfg.obstacle.replace(
        spawn=scene_cfg.obstacle.spawn.replace(
            usd_path=usd_path,
            scale=tuple(usd_cfg.get("scale", (1.0, 1.0, 1.0))),
        ),
        init_state=scene_cfg.obstacle.init_state.replace(
            pos=tuple(usd_cfg.get("translation", (0.0, 0.0, 0.0))),
            rot=tuple(usd_cfg.get("rotation", (1.0, 0.0, 0.0, 0.0))),
        ),
    )

    scene = InteractiveScene(scene_cfg)

    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

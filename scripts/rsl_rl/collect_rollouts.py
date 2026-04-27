"""Collect rollouts with learned Mimic Policy and noise on actions (to gather corrective actions)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

import yaml

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Tracking-Mesh-G1-Play-v0", help="Name of the task.")
parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")
parser.add_argument(
    "--usd_name", type=str, default=None, help="Name of the USD mesh."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# USD import and config
usd_folder = "data/terrains/prepared/"
usd_path = usd_folder + f"{args_cli.usd_name}/" + f"{args_cli.usd_name}" + ".usd"
usd_cfg_path = usd_folder + f"{args_cli.usd_name}/" + "config.yaml"

# Fetch config from yaml
with open(usd_cfg_path, "r") as f:
    usd_cfg: dict = yaml.load(f, Loader=yaml.FullLoader)


"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import pathlib
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
import importlib.metadata as metadata

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

installed_version = metadata.version("rsl-rl-lib")

# Import extensions to set up environment tasks
import RT_IL.tasks  # noqa: F401
from RT_IL.tasks.tracking.mdp.commands import MotionCommand
from RT_IL.utils.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
    infer_export_observation_group,
    select_export_normalizer,
)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    import wandb
    api = wandb.Api()
    # load the motion file from the wandb registry
    registry_name = args_cli.registry_name
    if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
        registry_name += ":latest"
    artifact = api.artifact(registry_name)
    env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    env_cfg.scene.obstacle = env_cfg.scene.obstacle.replace(
        spawn=env_cfg.scene.obstacle.spawn.replace(
            usd_path=usd_path,
            scale=tuple(usd_cfg.get("scale", (1.0, 1.0, 1.0))),
        ),
        init_state=env_cfg.scene.obstacle.init_state.replace(
            pos=tuple(usd_cfg.get("translation", (0.0, 0.0, 0.0))),
            rot=tuple(usd_cfg.get("rotation", (1.0, 0.0, 0.0, 0.0))),
        ),
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # strip deprecated fields from model configs for the installed rsl-rl version
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        ppo_runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    obs_group = infer_export_observation_group(env.unwrapped)
    normalizer = select_export_normalizer(policy_nn, obs_group)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        policy_nn,
        normalizer=normalizer,
        path=export_model_dir,
        filename="policy.onnx",
        prefer_student=(obs_group == "student"),
    )
    attach_onnx_metadata(
        env.unwrapped,
        args_cli.wandb_path if args_cli.wandb_path else "none",
        export_model_dir,
        obs_group=obs_group,
    )
    # reset environment
    obs = env.get_observations()

    scene = env.unwrapped.scene
    robot = scene["robot"]
    data_joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    data_joint_indices = robot.find_joints(data_joint_names, preserve_order=True)[0]
    data_joint_indices_tensor = torch.as_tensor(
        data_joint_indices, dtype=torch.long, device=robot.data.joint_pos.device
    )
    # states
    root_pos_rows: list[np.ndarray] = []
    root_quat_rows: list[np.ndarray] = []
    joint_pos_rows: list[np.ndarray] = []
    height_scan_rows: list[np.ndarray] = []
    # actions
    actions_rows: list[np.ndarray] = []
    output_dir = os.path.join("data", "motions", "inference")
    os.makedirs(output_dir, exist_ok=True)
    output_name = args_cli.usd_name or "rollout"
    output_path = os.path.join(output_dir, f"{output_name}_rollout.npz")
    try:
        viz_robot = scene["viz_robot"]
    except KeyError:
        viz_robot = None
    motion_term: MotionCommand = env.unwrapped.command_manager.get_term("motion")  # type: ignore
    apply_noise = True  # need this later to collect data for the diffusion model
    target_steps = int(motion_term.motion.time_step_total)
    attempt_steps = 0
    attempt_count = 0
    completed = False
    height_scanner = scene.sensors.get("height_scanner")
    if height_scanner is None:
        raise RuntimeError("Height scanner not found in scene; enable scene.height_scanner in the env config.")
    height_scanner_cfg = env.unwrapped.cfg.scene.height_scanner
    pattern_cfg = height_scanner_cfg.pattern_cfg
    height_scan_resolution = float(pattern_cfg.resolution)
    height_scan_size = (float(pattern_cfg.size[0]), float(pattern_cfg.size[1]))
    height_scan_ordering = getattr(pattern_cfg, "ordering", "xy")
    height_scan_x = np.arange(
        -height_scan_size[0] / 2,
        height_scan_size[0] / 2 + 1.0e-9,
        height_scan_resolution,
        dtype=np.float32,
    )
    height_scan_y = np.arange(
        -height_scan_size[1] / 2,
        height_scan_size[1] / 2 + 1.0e-9,
        height_scan_resolution,
        dtype=np.float32,
    )
    height_scan_count_x = int(height_scan_x.size)
    height_scan_count_y = int(height_scan_y.size)
    height_scan_num_rays = int(height_scanner.data.ray_hits_w.shape[1])
    if height_scan_count_x * height_scan_count_y != height_scan_num_rays:
        raise RuntimeError(
            "Height scan shape mismatch: expected"
            f" {height_scan_count_y}x{height_scan_count_x}={height_scan_count_x * height_scan_count_y},"
            f" got {height_scan_num_rays} rays."
        )
    if height_scan_ordering == "xy":
        height_scan_shape = (height_scan_count_y, height_scan_count_x)
    else:
        height_scan_shape = (height_scan_count_x, height_scan_count_y)
    height_scan_offset_z = float(height_scanner_cfg.offset.pos[2])
    height_scan_reference = "parent_frame"
    def _update_viz_robot():
        """Mirror the current motion command on the visualization robot."""
        if viz_robot is None:
            return
        root_states = viz_robot.data.default_root_state.clone()
        root_states[:, :3] = motion_term.body_pos_w[:, 0]
        root_states[:, 3:7] = motion_term.body_quat_w[:, 0]
        root_states[:, 7:10] = motion_term.body_lin_vel_w[:, 0]
        root_states[:, 10:] = motion_term.body_ang_vel_w[:, 0]
        viz_robot.write_root_state_to_sim(root_states)
        viz_robot.write_joint_state_to_sim(motion_term.joint_pos.clone(), motion_term.joint_vel.clone())
        scene.write_data_to_sim()

    # simulate environment
    while simulation_app.is_running() and not completed:
        root_pos = robot.data.root_pos_w.clone()
        root_pos[:, :2] -= scene.env_origins[:, :2]
        root_quat = robot.data.root_quat_w
        root_quat_xyzw = root_quat[:, [1, 2, 3, 0]]
        joint_pos = torch.index_select(robot.data.joint_pos, 1, data_joint_indices_tensor)
        root_pos_rows.append(root_pos.cpu().numpy().copy())
        root_quat_rows.append(root_quat_xyzw.cpu().numpy().copy())
        joint_pos_rows.append(joint_pos.cpu().numpy().copy())
        height_scan = height_scanner.data.pos_w[:, 2].unsqueeze(1) - height_scanner.data.ray_hits_w[..., 2]
        height_scan_rows.append(height_scan.cpu().numpy().copy())
        attempt_steps += 1
        if attempt_steps >= target_steps:
            completed = True
            break

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            actions_rows.append(actions.cpu().numpy().copy())  # store corrective actions without noise
            # apply action noise if specified
            if apply_noise:
                # add noise to the actions, sample from normal distribution with mean at the actual action
                actions = torch.normal(mean=actions, std=0.125 * torch.ones_like(actions))

            # env stepping
            obs, _, dones, _ = env.step(actions)  # apply noisy actions
            _update_viz_robot()
        if torch.any(dones > 0).item():
            attempt_count += 1
            root_pos_rows.clear()
            root_quat_rows.clear()
            joint_pos_rows.clear()
            height_scan_rows.clear()
            actions_rows.clear()
            attempt_steps = 0
            print(f"[INFO]: Rollout terminated early; retrying (attempt {attempt_count}).")

    # close the simulator
    env.close()

    if completed and root_pos_rows:
        root_pos = np.asarray(root_pos_rows, dtype=np.float32)
        root_quat = np.asarray(root_quat_rows, dtype=np.float32)
        joint_pos = np.asarray(joint_pos_rows, dtype=np.float32)
        height_scan = np.asarray(height_scan_rows, dtype=np.float32)
        actions = np.asarray(actions_rows, dtype=np.float32)
        np.savez(
            output_path,
            root_pos=root_pos,
            root_quat=root_quat,
            joint_pos=joint_pos,
            joint_names=np.array(data_joint_names),
            height_scan=height_scan,
            height_scan_shape=np.array(height_scan_shape, dtype=np.int32),
            height_scan_resolution=np.array([height_scan_resolution], dtype=np.float32),
            height_scan_size=np.array(height_scan_size, dtype=np.float32),
            height_scan_ordering=np.array(height_scan_ordering),
            height_scan_offset_z=np.array([height_scan_offset_z], dtype=np.float32),
            height_scan_reference=np.array(height_scan_reference),
            actions=actions,
            dt=np.array([env.unwrapped.step_dt], dtype=np.float32),
        )
        print(f"[INFO]: Saved rollout to: {output_path}")
    else:
        print("[WARN]: Rollout did not complete; nothing saved.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

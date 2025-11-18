"""Use velocity task policy to traverse mesh terrain and collect data implement data collection."""
""" Example usage:
python scripts/rsl_rl/record_mesh_trajectory.py --load_run 2025-10-31_01-13-53 --checkpoint model_6000.pt --motion_length 200 --usd_name step --headless
"""
"""Launch Isaac Sim Simulator first."""

import argparse

import yaml

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--motion_length",
    type=int,
    default=200,
    help="Length of the desired motion (in steps).",
)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Mesh-G1-Play-v0", help="Name of the task.")
parser.add_argument(
    "--usd_name", type=str, default=None, help="Name of the USD mesh."
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

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

import os
import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import RT_IL.tasks  # noqa: F401


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli
    )

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

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
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

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx"
    )

    scene = env.unwrapped.scene

    # prepare logging
    log = {
        "body_pos_w": [],
        "body_quat_w": [],
        "joint_pos": [],
    }

    # reset environment
    obs = env.get_observations()
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

    timestep = 0

    # simulate environment
    while simulation_app.is_running() and timestep < args_cli.motion_length:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            body_pos = robot.data.body_pos_w[0]
            body_quat = robot.data.body_quat_w[0]
            if body_pos.ndim > 1:
                body_pos = body_pos[0]
            if body_quat.ndim > 1:
                body_quat = body_quat[0]
            log["body_pos_w"].append(body_pos.cpu().numpy().copy())
            log["body_quat_w"].append(body_quat.cpu().numpy().copy())
            joint_pos_tensor = torch.index_select(
                robot.data.joint_pos[0], 0, data_joint_indices_tensor
            )
            log["joint_pos"].append(joint_pos_tensor.cpu().numpy().copy())

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        timestep += 1

    # close the simulator
    env.close()

    if log["body_pos_w"]:
        body_pos = np.asarray(log["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(log["body_quat_w"], dtype=np.float64)
        joint_pos = np.asarray(log["joint_pos"], dtype=np.float64)

        if body_quat.shape[1] == 4:
            body_quat = body_quat[:, [1, 2, 3, 0]]

        trajectory = np.hstack((body_pos, body_quat, joint_pos))

        trajectory_path = os.path.join("trajectories/mesh", f"{args_cli.usd_name}_trajectory.csv")
        os.makedirs(os.path.dirname(trajectory_path), exist_ok=True)
        np.savetxt(trajectory_path, trajectory, delimiter=",", fmt="%.17f")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

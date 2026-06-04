"""Evaluate learned Mimic Policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from usd_utils import load_usd_config  # isort: skip

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
parser.add_argument(
    "--difficulty",
    type=float,
    default=None,
    help="Freeze the robustness curriculum at this difficulty in [0, 1] for eval "
    "(1.0 = full spawn randomization + loosened terminations the policy trained against).",
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

usd_path, usd_cfg = load_usd_config(args_cli.usd_name)


"""Rest everything follows."""

import gymnasium as gym
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

    # override the robustness curriculum difficulty for eval (recover-from-randomization tests)
    if args_cli.difficulty is not None:
        robustness = getattr(getattr(env_cfg, "curriculum", None), "robustness", None)
        if robustness is not None:
            robustness.params["start_difficulty"] = args_cli.difficulty
            robustness.params["freeze"] = True
            print(f"[INFO] Robustness curriculum frozen at difficulty={args_cli.difficulty}")
        else:
            print("[WARN] --difficulty given but this env has no robustness curriculum term; ignoring.")

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

        # if args_cli.motion_file is not None:
        #     print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        #     env_cfg.commands.motion.motion_file = args_cli.motion_file

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
        try:
            # version 2.2 and below
            policy_nn = ppo_runner.alg.actor_critic
        except AttributeError:
            # current rsl_rl (actor/critic split, use get_policy())
            policy_nn = ppo_runner.alg.get_policy()

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
    timestep = 0

    scene = env.unwrapped.scene
    try:
        viz_robot = scene["viz_robot"]
    except KeyError:
        viz_robot = None
    motion_term: MotionCommand = env.unwrapped.command_manager.get_term("motion")  # type: ignore
    apply_noise = False  # need this later to collect data for the diffusion model

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
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # apply action noise if specified
            if apply_noise:
                # add noise to the actions, sample from normal distribution with mean at the actual action
                actions = torch.normal(mean=actions, std=0.5 * torch.ones_like(actions))

            # env stepping
            obs, _, _, _ = env.step(actions)
            _update_viz_robot()
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

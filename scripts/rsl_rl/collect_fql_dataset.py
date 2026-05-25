"""Collect a Flow-Q-Learning dataset by rolling out an expert policy.

Loads a trained motion-tracking expert (teacher) and rolls it out in the
joystick-distillation env so both observation groups exist simultaneously:
  - obs["policy"]  : teacher's view (motion command + proprio + scan)
  - obs["student"] : student's view (joystick + proprio + scan)

At each step the teacher acts on obs["policy"], the env transitions, and we log
the (s_student, a, r, s'_student, done) tuple. The output is a single .pt file
per expert that can be aggregated later into a multi-skill FQL replay buffer.

Example:
    python scripts/rsl_rl/collect_fql_dataset.py \\
        --task Tracking-Mesh-G1-Joystick-Distillation-v0 \\
        --registry_name box_up \\
        --wandb_path <wandb_run_path> \\
        --usd_name box_up \\
        --num_envs 4096 \\
        --num_transitions 2000000 \\
        --action_noise_std 0.05 \\
        --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from usd_utils import load_usd_config  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=4096,
                    help="Number of parallel envs for rollout (higher = faster collection).")
parser.add_argument("--task", type=str, default="Tracking-Mesh-G1-Joystick-Distillation-v0",
                    help="Env task id. Must expose both 'policy' and 'student' obs groups.")
parser.add_argument("--teacher_cfg", type=str, default="ppo", choices=["ppo", "ppo_deploy"],
                    help="Which PPO runner cfg matches the teacher's training. "
                         "'ppo' = G1MeshPPORunnerCfg ([512,256,128]); "
                         "'ppo_deploy' = G1MeshPPODeployRunnerCfg ([1024,512,256,128]).")
parser.add_argument("--registry_name", type=str, default=None,
                    help="W&B motion artifact name (required for motion-tracking envs; "
                         "omit for velocity-tracking envs that have no motion command).")
parser.add_argument("--usd_name", type=str, default=None,
                    help="Name of the obstacle USD mesh (matches the clip). Omit for "
                         "envs without an obstacle prim (e.g. the velocity env).")
parser.add_argument("--num_transitions", type=int, default=2_000_000,
                    help="Target number of (s, a, r, s', done) tuples to collect.")
parser.add_argument("--action_noise_std", type=float, default=0.0,
                    help="Optional Gaussian noise on the expert's action for state coverage.")
parser.add_argument("--output_dir", type=str, default="data/fql_rollouts",
                    help="Directory to write the dataset file into.")
parser.add_argument("--output_name", type=str, default=None,
                    help="Output file stem (defaults to <registry>_<usd>).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if args_cli.usd_name:
    usd_path, usd_cfg = load_usd_config(args_cli.usd_name)
else:
    usd_path, usd_cfg = None, {}


import gymnasium as gym
import importlib.metadata as metadata
import pathlib
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import RT_IL.tasks  # noqa: F401

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg_unused):
    # The hydra-loaded cfg here is the DISTILLATION runner cfg (because the joystick task
    # is registered with that). That's wrong for loading a PPO teacher checkpoint —
    # the teacher reads obs["policy"] (motion command), not obs["student"] (joystick).
    # Override with the PPO cfg so the runner builds a network with obs_groups
    # {"actor": ["policy"], "critic": ["critic"]} matching the teacher's training.
    from RT_IL.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import (
        G1MeshPPODeployRunnerCfg,
        G1MeshPPORunnerCfg,
    )
    if args_cli.teacher_cfg == "ppo_deploy":
        agent_cfg = G1MeshPPODeployRunnerCfg()
    else:
        agent_cfg = G1MeshPPORunnerCfg()
    # Honour the standard CLI overrides (resume path, run name, etc.) the same way
    # parse_rsl_rl_cfg would for the PPO task.
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    print(f"[INFO] Runner cfg: {type(agent_cfg).__name__}  "
          f"obs_groups={getattr(agent_cfg, 'obs_groups', '?')}")

    # Resolve teacher checkpoint (wandb_path takes precedence, mirrors collect_rollouts.py).
    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path
        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        files = [f.name for f in wandb_run.files() if "model" in f.name]
        if "model" in args_cli.wandb_path:
            ckpt_file = args_cli.wandb_path.split("/")[-1]
        else:
            ckpt_file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        wandb_run.file(ckpt_file).download("./logs/rsl_rl/temp", replace=True)
        resume_path = f"./logs/rsl_rl/temp/{ckpt_file}"
        print(f"[INFO] Teacher checkpoint: {run_path}/{ckpt_file}")
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Teacher checkpoint: {resume_path}")

    # Resolve motion file from W&B registry — only if this env has a motion command.
    if args_cli.registry_name is not None and hasattr(env_cfg.commands, "motion"):
        import wandb
        api = wandb.Api()
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        artifact = api.artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    elif hasattr(env_cfg.commands, "motion"):
        raise ValueError(
            "Env has a `motion` command term but --registry_name was not provided."
        )

    # Substitute obstacle USD — only if both the env has an obstacle prim and a USD was given.
    if usd_path is not None and getattr(env_cfg.scene, "obstacle", None) is not None:
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

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    # Robot handle for body-frame velocity (used by the command-tracking reward).
    robot = env.unwrapped.scene["robot"]
    from isaaclab.utils.math import yaw_quat, quat_apply_inverse

    # Load teacher checkpoint into an OnPolicyRunner so we get its inference path.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Probe obs/action dims by stepping once.
    obs = env.get_observations()
    if not isinstance(obs, dict) and not hasattr(obs, "keys"):
        raise RuntimeError(
            f"Env returned a single-tensor obs; expected a dict with 'policy' and 'student' groups. "
            f"Use a task that exposes both groups (e.g. Tracking-Mesh-G1-Joystick-Distillation-v0)."
        )
    if "policy" not in obs or "student" not in obs:
        raise RuntimeError(
            f"Env obs groups are {list(obs.keys())}; need both 'policy' and 'student'."
        )

    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    student_dim = obs["student"].shape[-1]
    action_dim = env.unwrapped.action_manager.total_action_dim
    print(f"[INFO] num_envs={num_envs}  student_obs_dim={student_dim}  action_dim={action_dim}")

    # Pre-allocate CPU buffers (GPU is reserved for sim + policy).
    target = int(args_cli.num_transitions)
    # Round up to a multiple of num_envs so we don't waste a partial step.
    num_steps = (target + num_envs - 1) // num_envs
    capacity = num_steps * num_envs
    print(f"[INFO] Collecting {capacity:,} transitions over {num_steps} env steps "
          f"(target was {target:,}).")

    # Plain CPU tensors (no pin_memory) — pinned pages exhaust the CUDA-pinned pool
    # at multi-GB buffer sizes while sim is still running on the GPU.
    obs_buf = torch.empty((capacity, student_dim), dtype=torch.float32, device="cpu")
    next_obs_buf = torch.empty((capacity, student_dim), dtype=torch.float32, device="cpu")
    action_buf = torch.empty((capacity, action_dim), dtype=torch.float32, device="cpu")
    reward_buf = torch.empty((capacity,), dtype=torch.float32, device="cpu")
    reward_cmd_buf = torch.empty((capacity,), dtype=torch.float32, device="cpu")
    done_buf = torch.empty((capacity,), dtype=torch.bool, device="cpu")

    apply_noise = args_cli.action_noise_std > 0.0
    if apply_noise:
        print(f"[INFO] Action noise std={args_cli.action_noise_std}")

    t0 = time.time()
    write = 0
    with torch.inference_mode():
        for step in range(num_steps):
            s_student = obs["student"]
            action_clean = policy(obs)
            if apply_noise:
                action = action_clean + args_cli.action_noise_std * torch.randn_like(action_clean)
            else:
                action = action_clean

            obs, reward, done, _ = env.step(action)

            # ---- Command-tracking reward (computed in next-obs context) ----
            # The "current" joystick label sits in the first 3 dims of obs["student"]
            # because joystick is the first ObsTerm in StudentPolicyCfg and IsaacLab
            # stacks history newest-first.  Robot velocity in the yaw-only heading
            # frame is the natural frame for forward-speed tracking.
            yaw_q = yaw_quat(robot.data.root_quat_w)
            v_h = quat_apply_inverse(yaw_q, robot.data.root_lin_vel_w)
            vx_robot = v_h[:, 0]
            wz_robot = robot.data.root_ang_vel_w[:, 2]  # yaw rate is yaw-frame invariant
            joy = obs["student"][:, 0:3]
            vx_cmd, wz_cmd = joy[:, 0], joy[:, 2]
            err = (vx_cmd - vx_robot).square() + (wz_cmd - wz_robot).square()
            r_cmd = torch.exp(-err / (0.5 ** 2))  # sigma = 0.5

            # Store the action that was actually executed so (s, a, r, s') is consistent
            # for Q learning. With action_noise_std=0 this is just the clean expert action.
            sl = slice(write, write + num_envs)
            obs_buf[sl].copy_(s_student, non_blocking=True)
            next_obs_buf[sl].copy_(obs["student"], non_blocking=True)
            action_buf[sl].copy_(action, non_blocking=True)
            reward_buf[sl].copy_(reward.float(), non_blocking=True)
            reward_cmd_buf[sl].copy_(r_cmd.float(), non_blocking=True)
            done_buf[sl].copy_(done.bool(), non_blocking=True)
            write += num_envs

            if (step + 1) % 50 == 0 or step == num_steps - 1:
                elapsed = time.time() - t0
                fps = (step + 1) * num_envs / elapsed
                print(f"  [{step+1:>5d}/{num_steps}]  {write:,} transitions  "
                      f"{fps:,.0f} env-steps/s  done={done.float().mean().item():.3f}  "
                      f"r_motion={reward.float().mean().item():+.3f}  "
                      f"r_cmd={r_cmd.mean().item():.3f}")

    env.close()

    # Save.
    output_name = args_cli.output_name or f"{args_cli.registry_name.split('/')[-1].replace(':', '_')}"
    if args_cli.usd_name:
        output_name = f"{output_name}_{args_cli.usd_name}"
    os.makedirs(args_cli.output_dir, exist_ok=True)
    output_path = os.path.join(args_cli.output_dir, f"{output_name}.pt")

    torch.save({
        "observations": obs_buf,
        "actions": action_buf,
        "rewards": reward_buf,
        "rewards_cmd": reward_cmd_buf,   # command-velocity tracking reward
        "next_observations": next_obs_buf,
        "dones": done_buf,
        "metadata": {
            "task": args_cli.task,
            "registry_name": args_cli.registry_name,
            "usd_name": args_cli.usd_name,
            "checkpoint": resume_path,
            "num_envs": num_envs,
            "num_steps": num_steps,
            "student_obs_dim": student_dim,
            "action_dim": action_dim,
            "action_noise_std": args_cli.action_noise_std,
            "reward_cmd_sigma": 0.5,
            "step_dt": float(env.unwrapped.step_dt),
        },
    }, output_path)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\n[INFO] Saved {capacity:,} transitions to {output_path}  ({size_mb:,.1f} MB)")
    print(f"[INFO] Total wall time: {elapsed:.1f}s ({capacity/elapsed:,.0f} transitions/s)")


if __name__ == "__main__":
    main()
    simulation_app.close()

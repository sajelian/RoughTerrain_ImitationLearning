"""Smoke test for the joystick distillation env.

Launches IsaacLab headless, constructs Tracking-Mesh-G1-Joystick-Distillation-v0
with num_envs=2, steps the env once, and prints observation-group shapes so we
can confirm:
  - the joystick obs is being produced
  - the student group dropped `command` and `motion_anchor_ori_b`
  - the teacher (policy) group is unchanged

Example:
    python scripts/smoke_test_joystick_env.py --headless \\
        --motion_file artifacts/beam_along:v1/motion.npz \\
        --usd_name beam_along
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usd_utils import load_usd_config  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion_file", default="artifacts/beam_along:v1/motion.npz")
parser.add_argument("--usd_name", default="beam_along")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--task", default="Tracking-Mesh-G1-Joystick-Distillation-v0")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

usd_path, usd_cfg = load_usd_config(args.usd_name)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import RT_IL.tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands.motion.motion_file = args.motion_file
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

    env = gym.make(args.task, cfg=env_cfg)

    print("\n=== observation_space (raw env) ===")
    for name, space in env.observation_space.items():
        print(f"  {name:>10s}  shape={space.shape}")

    obs, _ = env.reset()
    print("\n=== reset obs shapes ===")
    for name, tensor in obs.items():
        print(f"  {name:>10s}  shape={tuple(tensor.shape)}  dtype={tensor.dtype}")

    print("\n=== first 'joystick' value per env (should match motion start velocity) ===")
    # extract joystick: it's the first 3 dims * PROPIO_HISTORY_SHORT of student
    # but since obs is concatenated, easier to call the obs func directly
    from RT_IL.tasks.tracking.mdp.observations import motion_joystick_command
    joystick = motion_joystick_command(env.unwrapped, command_name="motion", horizon_s=1.5)
    for i, j in enumerate(joystick):
        print(f"  env {i}: vx={j[0]:+.3f}  vy={j[1]:+.3f}  wz={j[2]:+.3f}")

    actions = torch.zeros((args.num_envs, env.unwrapped.action_manager.total_action_dim),
                          device=env.unwrapped.device)
    obs, reward, terminated, truncated, info = env.step(actions)
    print("\n=== one env.step() succeeded ===")
    print(f"  reward shape: {tuple(reward.shape)}  mean={reward.mean().item():+.3f}")
    print(f"  terminated: {terminated.sum().item()}/{args.num_envs}  truncated: {truncated.sum().item()}/{args.num_envs}")

    env.close()
    print("\n[OK] joystick distillation env smoke test passed.")


if __name__ == "__main__":
    main()
    simulation_app.close()

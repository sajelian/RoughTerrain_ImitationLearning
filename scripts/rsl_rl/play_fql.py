"""Play a trained FQL policy in the joystick-keyboard env.

Loads `agent_*.pt` + `normalizer.pt` produced by `train_fql.py` and runs the
one-step policy μ_ω (or the iterative flow μ_θ for comparison) inside
`Tracking-Mesh-G1-Joystick-Distillation-Play-v0` so the joystick obs comes
from the live keyboard (see `keyboard_joystick_command`).

Example:
    python scripts/rsl_rl/play_fql.py \\
        --checkpoint logs/fql/box_up/agent_100000.pt \\
        --registry_name box_up --usd_name box_up
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from usd_utils import load_usd_config  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to agent_*.pt produced by train_fql.py.")
parser.add_argument("--normalizer", type=str, default=None,
                    help="Path to normalizer.pt (default: <checkpoint_dir>/normalizer.pt).")
parser.add_argument("--task", type=str, default="Deploy-G1-v0",
                    help="Deploy-G1-v0 is the clean deploy env (no motion clip). "
                         "Use Tracking-Mesh-G1-Joystick-Distillation-Play-v0 only for "
                         "legacy/sanity testing with the motion-tracking infrastructure.")
parser.add_argument("--registry_name", type=str, default=None,
                    help="W&B motion artifact — required only for legacy tracking-env tasks "
                         "that have a motion command. Deploy-G1-v0 does not need it.")
parser.add_argument("--usd_name", type=str, default=None,
                    help="Obstacle USD to load at runtime. Omit for flat-ground deploy.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)

# Policy mode
parser.add_argument("--use_flow", action="store_true",
                    help="Use iterative flow integration (μ_θ) instead of one-step (μ_ω). "
                         "Slower but gives the ceiling the one-step distillation aims for.")
parser.add_argument("--deterministic_z", action="store_true",
                    help="Use z=0 instead of N(0,I) for action sampling. Removes stochasticity.")
parser.add_argument("--flow_steps", type=int, default=10,
                    help="Euler steps when --use_flow is set.")

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
import pathlib
import torch

from isaaclab.envs import (
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import RT_IL.tasks  # noqa: F401
from RT_IL.algorithms.fql import FQLAgent, FQLConfig


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs

    # Motion file — only when the env actually has a motion command term
    # (legacy tracking play envs). Deploy-G1-v0 doesn't have one.
    if args_cli.registry_name is not None and hasattr(env_cfg.commands, "motion"):
        import wandb
        import pathlib as _pathlib
        api = wandb.Api()
        reg = args_cli.registry_name if ":" in args_cli.registry_name else args_cli.registry_name + ":latest"
        artifact = api.artifact(reg)
        env_cfg.commands.motion.motion_file = str(_pathlib.Path(artifact.download()) / "motion.npz")

    # Obstacle USD swap — only if both the env has an obstacle prim and a USD was given.
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

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)
    device = env.unwrapped.device

    # --- Load checkpoint + normalizer ---
    ckpt = torch.load(args_cli.checkpoint, weights_only=False, map_location=device)
    norm_path = args_cli.normalizer or os.path.join(os.path.dirname(args_cli.checkpoint), "normalizer.pt")
    if not os.path.isfile(norm_path):
        raise FileNotFoundError(
            f"normalizer.pt not found at {norm_path}. Pass --normalizer explicitly."
        )
    normalizer = torch.load(norm_path, weights_only=False, map_location=device)

    obs_mean = normalizer["obs_mean"].to(device)
    obs_std = normalizer["obs_std"].to(device)
    obs_clip = float(normalizer["obs_clip"])
    obs_dim = int(normalizer["obs_dim"])
    action_dim = int(normalizer["action_dim"])
    # Action normalizer (policy operates in standardized action space).
    if "action_mean" not in normalizer:
        raise KeyError(
            "normalizer.pt has no action_mean/action_std — it was produced by an "
            "older train_fql.py. Retrain with the current version."
        )
    action_mean = normalizer["action_mean"].to(device)
    action_std = normalizer["action_std"].to(device)

    # Recreate the agent. The saved 'config' dict in the checkpoint preserves the exact
    # network sizes so we don't have to guess.
    saved_cfg = ckpt.get("config", {})
    cfg = FQLConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        flow_hidden_dims=tuple(saved_cfg.get("flow_hidden_dims", (512, 512, 512, 512))),
        onestep_hidden_dims=tuple(saved_cfg.get("onestep_hidden_dims", (512, 512, 512, 512))),
        q_hidden_dims=tuple(saved_cfg.get("q_hidden_dims", (512, 512, 512, 512))),
        activation=saved_cfg.get("activation", "gelu"),
        time_embed_dim=saved_cfg.get("time_embed_dim", 64),
        flow_integration_steps=args_cli.flow_steps,
        device=str(device),
    )
    agent = FQLAgent(cfg)
    agent.load_state_dict(ckpt["agent"])
    agent.flow.eval(); agent.onestep.eval(); agent.q1.eval(); agent.q2.eval()
    print(f"[FQL] Loaded checkpoint from iter {ckpt.get('iter', '?')}: {args_cli.checkpoint}")
    print(f"[FQL] obs_dim={obs_dim}  action_dim={action_dim}  "
          f"hidden={cfg.onestep_hidden_dims}  activation={cfg.activation}")
    mode = "flow (μ_θ)" if args_cli.use_flow else "one-step (μ_ω)"
    z_mode = "deterministic z=0" if args_cli.deterministic_z else "stochastic z~N(0,I)"
    print(f"[FQL] Action mode: {mode}  |  {z_mode}")

    def normalize_obs(o: torch.Tensor) -> torch.Tensor:
        return ((o - obs_mean) / obs_std).clamp(-obs_clip, obs_clip)

    def act(student_obs: torch.Tensor) -> torch.Tensor:
        obs_n = normalize_obs(student_obs)
        if args_cli.use_flow:
            action_n = agent.act_flow(obs_n, num_steps=args_cli.flow_steps)
        else:
            action_n = agent.act(obs_n, deterministic_z=args_cli.deterministic_z)
        # Policy operates in standardized action space — map back to raw env units.
        return action_n * action_std + action_mean

    # --- Run loop ---
    obs = env.get_observations()
    step = 0
    print("[FQL] Running. Use W/A/D/Z/X to drive (see keyboard_joystick_command for full mapping).")
    with torch.inference_mode():
        while simulation_app.is_running():
            student_obs = obs["student"]
            actions = act(student_obs)
            obs, _, _, _ = env.step(actions)
            step += 1
            if step % 100 == 0:
                # Joystick slice is the first 3 dims of the most-recent obs frame.
                # With concatenate_terms + history, the layout puts the newest joystick
                # at indices [0:3] (history is concatenated newest-first per ObsGroup).
                # We print raw (un-normalized) values for readability.
                js = student_obs[0, :3].cpu()
                a = actions[0].cpu()
                print(f"  [step {step:5d}] joystick=[{js[0]:+.2f}, {js[1]:+.2f}, {js[2]:+.2f}]  "
                      f"action range=[{a.min():+.3f}, {a.max():+.3f}]  |a|_mean={a.abs().mean():.3f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

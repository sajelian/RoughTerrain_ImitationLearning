"""Side-by-side PPO vs FQL action comparison at deploy.

Loads both the velocity PPO checkpoint AND the FQL student checkpoint, runs them
in the same env, and at every step logs both predicted actions. The env is
driven by *one* of them (default: PPO) so the state distribution stays in the
PPO's training regime, isolating "does FQL output what PPO would output at
these exact states?" from "does FQL's own state distribution match training?".

Three interpretations of the metric (running RMSE between predictions):
  * Near training-time RMSE (e.g. ~0.15 in raw action units for the velocity task):
    FQL faithfully matches PPO; remaining deploy failure is *not* about distillation
    precision per se but about how that small error compounds in the unstable system.
  * Much larger than training RMSE: FQL extrapolates badly at deploy states.
    Means the deploy state distribution differs from training in a way the student
    can't generalise across.
  * Same range, but specific joints/dims diverge: log the per-dim error to spot
    which joints FQL gets wrong.

Example:
    python scripts/rsl_rl/compare_ppo_fql.py \\
        --task Velocity-G1-Play-v0 \\
        --wandb_path <ppo_velocity_run> \\
        --fql_checkpoint logs/fql/<run>/agent_N.pt \\
        --num_envs 1 --num_steps 500 --drive_with ppo
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Velocity-G1-Play-v0",
                    help="Env task id. Must expose both 'policy' and 'student' obs groups.")
parser.add_argument("--fql_checkpoint", type=str, required=True,
                    help="Path to FQL agent_*.pt produced by train_fql.py.")
parser.add_argument("--normalizer", type=str, default=None,
                    help="Path to normalizer.pt (default: <fql_checkpoint_dir>/normalizer.pt).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=500,
                    help="How long to run before exiting.")
parser.add_argument("--drive_with", type=str, default="ppo", choices=["ppo", "fql", "alternate"],
                    help="Which policy actually steps the env. "
                         "'ppo' keeps the state in PPO's training distribution (recommended). "
                         "'fql' lets FQL drive — useful for watching divergence. "
                         "'alternate' swaps every step (chaos but informative).")
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--deterministic_fql_z", action="store_true",
                    help="Use z=0 for the FQL one-step policy (removes z noise).")
parser.add_argument("--use_flow", action="store_true",
                    help="Use iterative flow for FQL instead of one-step.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import importlib.metadata as metadata
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import RT_IL.tasks  # noqa: F401
from RT_IL.algorithms.fql import FQLAgent, FQLConfig

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # ---- Resolve PPO checkpoint ----
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
        ppo_resume_path = f"./logs/rsl_rl/temp/{ckpt_file}"
        print(f"[INFO] PPO checkpoint (wandb): {ppo_resume_path}")
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        ppo_resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] PPO checkpoint (local): {ppo_resume_path}")

    # ---- Env ----
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = env.unwrapped.device

    # ---- Load PPO ----
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(device))
    ppo_runner.load(ppo_resume_path)
    ppo_policy = ppo_runner.get_inference_policy(device=device)
    print("[INFO] PPO loaded.")

    # ---- Load FQL agent + normalizer ----
    fql_ckpt = torch.load(args_cli.fql_checkpoint, weights_only=False, map_location=device)
    norm_path = args_cli.normalizer or os.path.join(os.path.dirname(args_cli.fql_checkpoint), "normalizer.pt")
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
    action_mean = normalizer["action_mean"].to(device)
    action_std = normalizer["action_std"].to(device)

    saved_cfg = fql_ckpt.get("config", {})
    cfg = FQLConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        flow_hidden_dims=tuple(saved_cfg.get("flow_hidden_dims", (1024, 1024, 512, 512))),
        onestep_hidden_dims=tuple(saved_cfg.get("onestep_hidden_dims", (1024, 1024, 512, 512))),
        q_hidden_dims=tuple(saved_cfg.get("q_hidden_dims", (1024, 1024, 512, 512))),
        activation=saved_cfg.get("activation", "gelu"),
        layer_norm_q=saved_cfg.get("layer_norm_q", True),
        device=str(device),
    )
    fql_agent = FQLAgent(cfg)
    fql_agent.load_state_dict(fql_ckpt["agent"])
    fql_agent.onestep.eval(); fql_agent.flow.eval(); fql_agent.q1.eval(); fql_agent.q2.eval()
    print(f"[INFO] FQL loaded (iter {fql_ckpt.get('iter', '?')}). "
          f"obs_dim={obs_dim}  action_dim={action_dim}")

    def fql_act(student_obs: torch.Tensor) -> torch.Tensor:
        obs_n = ((student_obs - obs_mean) / obs_std).clamp(-obs_clip, obs_clip)
        if args_cli.use_flow:
            action_n = fql_agent.act_flow(obs_n)
        else:
            action_n = fql_agent.act(obs_n, deterministic_z=args_cli.deterministic_fql_z)
        return action_n * action_std + action_mean

    # ---- Run ----
    obs = env.get_observations()
    if "policy" not in obs or "student" not in obs:
        raise RuntimeError(
            f"Env obs groups are {list(obs.keys())}; need both 'policy' (PPO) and 'student' (FQL)."
        )

    # Running stats
    n = 0
    sum_rmse = 0.0
    sum_cos = 0.0
    sum_diff_per_dim = torch.zeros(action_dim, device=device)
    sum_abs_ppo = 0.0
    sum_abs_fql = 0.0

    print(f"\n[INFO] Driving env with: {args_cli.drive_with}")
    print(f"[INFO] Step  |  PPO|a|  FQL|a|  rmse_norm  rmse_raw  cos_sim  diff_max")
    print(f"          {'-'*70}")

    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            if not simulation_app.is_running():
                break

            a_ppo = ppo_policy(obs)
            a_fql = fql_act(obs["student"])

            # Normalized-space comparison (matches diagnose_fql.py scale)
            a_ppo_n = (a_ppo - action_mean) / action_std
            a_fql_n = (a_fql - action_mean) / action_std

            diff = a_ppo - a_fql
            diff_n = a_ppo_n - a_fql_n
            rmse_raw = diff.pow(2).mean().sqrt().item()
            rmse_norm = diff_n.pow(2).mean().sqrt().item()
            cos = torch.nn.functional.cosine_similarity(
                a_ppo.reshape(-1), a_fql.reshape(-1), dim=0
            ).item()
            diff_max = diff.abs().max().item()
            abs_ppo = a_ppo.abs().mean().item()
            abs_fql = a_fql.abs().mean().item()

            n += 1
            sum_rmse += rmse_norm
            sum_cos += cos
            sum_diff_per_dim += diff.abs().mean(dim=0)
            sum_abs_ppo += abs_ppo
            sum_abs_fql += abs_fql

            if step % args_cli.log_interval == 0:
                print(f"  {step:>5d}  |  {abs_ppo:6.3f}  {abs_fql:6.3f}  "
                      f"{rmse_norm:8.4f}  {rmse_raw:8.4f}  {cos:+6.3f}  {diff_max:6.3f}")

            # Decide which action drives the env
            if args_cli.drive_with == "ppo":
                action_to_apply = a_ppo
            elif args_cli.drive_with == "fql":
                action_to_apply = a_fql
            else:  # alternate
                action_to_apply = a_ppo if (step % 2 == 0) else a_fql

            obs, _, _, _ = env.step(action_to_apply)

    # Summary
    print("\n" + "=" * 72)
    print(f"Summary over {n} steps  (env driven by: {args_cli.drive_with})")
    print("=" * 72)
    print(f"  mean PPO |a|:           {sum_abs_ppo/n:.4f}")
    print(f"  mean FQL |a|:           {sum_abs_fql/n:.4f}")
    print(f"  mean rmse_normalized:   {sum_rmse/n:.4f}    "
          f"(compare to diagnose_fql.py 'rmse_vs_ref'; if much larger -> deploy OOD)")
    print(f"  mean cos_sim:           {sum_cos/n:+.4f}")
    print()
    per_dim = sum_diff_per_dim / n
    top = per_dim.argsort(descending=True)[:5]
    print(f"  Top 5 joints by mean |PPO - FQL| (raw action units):")
    for k in top:
        print(f"    dim {int(k):>3d}:  {per_dim[k].item():.4f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

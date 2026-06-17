"""Collect a DAgger augmentation dataset by rolling out the FQL student in the
training env and labelling each state with the PPO teacher's action.

This produces transitions (s_student, a_PPO, r, s'_student, done) where:
  * s_student is the state the *student* visited (not states PPO would visit)
  * a_PPO is the teacher's correct action at that state
  * r, s', done come from the env stepping with the student's action

This is the canonical DAgger augmentation — every transition has been labelled
by the expert at exactly the kind of off-distribution state the BC student
tends to drift into. Mixing this dataset with the original PPO-driven dataset
and retraining typically closes the compounding-error gap that pure offline
BC can't reach.

Output is a .pt file in the same format as `collect_fql_dataset.py`, so the
existing replay buffer + `train_fql.py --datasets <files>` integrates it with
zero changes.

Loop integration:
    Designed for iterative DAgger rounds — pass a unique `--output_name` per
    round (e.g. `dagger_round_1`, `dagger_round_2`) and accumulate them in
    the dataset directory. `train_fql.py --datasets dir/*.pt` then trains
    on the growing union.

Example (single round):
    python scripts/rsl_rl/collect_dagger_dataset.py \\
        --task Velocity-G1-v0 \\
        --wandb_path <ppo_velocity_run> \\
        --fql_checkpoint logs/fql/<run>/agent_N.pt \\
        --output_name dagger_round_1 \\
        --num_envs 2048 --num_transitions 1000000 --headless
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
parser.add_argument("--num_envs", type=int, default=2048,
                    help="Parallel envs for rollout. Higher = faster collection.")
parser.add_argument("--task", type=str, default="Velocity-G1-v0",
                    help="Env task id. Must expose both 'policy' (for PPO) and "
                         "'student' (for FQL) obs groups.")
parser.add_argument("--teacher_cfg", type=str, default="ppo", choices=["ppo", "ppo_velocity"],
                    help="Which PPO runner cfg matches the teacher's training.")
parser.add_argument("--fql_checkpoint", type=str, required=True,
                    help="Path to FQL agent_*.pt — the student that drives the env.")
parser.add_argument("--normalizer", type=str, default=None,
                    help="Path to normalizer.pt (default: <fql_checkpoint_dir>/normalizer.pt).")
parser.add_argument("--num_transitions", type=int, default=1_000_000,
                    help="Target number of DAgger (s, a_PPO, r, s', done) tuples.")
parser.add_argument("--output_dir", type=str, default="data/fql_rollouts",
                    help="Directory to write the dataset file into.")
parser.add_argument("--output_name", type=str, required=True,
                    help="Output file stem (without .pt). Use a unique name per "
                         "DAgger round, e.g. 'dagger_round_1'.")
parser.add_argument("--deterministic_fql_z", action="store_true",
                    help="Drive with z=0 instead of stochastic z~N(0,I). Stochastic "
                         "is usually closer to the real deploy distribution.")
parser.add_argument("--use_flow", action="store_true",
                    help="Drive with iterative flow integration instead of one-step.")
parser.add_argument("--reset_on_fall", action="store_true", default=True,
                    help="Rely on env's bad_orientation termination to reset fallen "
                         "robots (default). Disable only if you want fallen-robot data.")
parser.add_argument("--registry_name", type=str, default=None,
                    help="W&B motion artifact name (required for tracking envs with a "
                         "MotionCommand term; omit for velocity envs).")
parser.add_argument("--usd_name", type=str, default=None,
                    help="Obstacle USD name. Omit for envs without an obstacle prim.")
cli_args.add_difficulty_arg(parser)
cli_args.add_skill_mode_arg(parser)
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
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg_unused):
    # The hydra-loaded cfg may be the distillation cfg if the task is registered
    # that way. We override with the PPO cfg so the runner builds the right
    # network that reads obs["policy"].
    if args_cli.teacher_cfg == "ppo_velocity":
        from RT_IL.tasks.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1VelocityPPORunnerCfg
        agent_cfg = G1VelocityPPORunnerCfg()
    else:
        from RT_IL.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1MeshPPORunnerCfg
        agent_cfg = G1MeshPPORunnerCfg()
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    cli_args.apply_difficulty_override(env_cfg, args_cli.difficulty)
    cli_args.apply_skill_mode_override(env_cfg, args_cli.skill_mode)
    print(f"[INFO] Teacher cfg: {type(agent_cfg).__name__}  obs_groups={agent_cfg.obs_groups}")

    # --- Resolve PPO checkpoint (wandb_path or local) ---
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

    # --- Resolve motion file from W&B registry (tracking envs + velocity-obstacle env need it) ---
    motion_file_path: str | None = None
    if args_cli.registry_name is not None:
        import wandb
        api_motion = wandb.Api()
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        artifact = api_motion.artifact(registry_name)
        motion_file_path = str(pathlib.Path(artifact.download()) / "motion.npz")
        print(f"[INFO] Motion file resolved from W&B: {motion_file_path}")

    # Wire it into wherever the env consumes it.
    motion_command = getattr(env_cfg.commands, "motion", None)
    reset_event = getattr(env_cfg.events, "reset_base", None)
    event_needs_motion = (
        reset_event is not None
        and hasattr(reset_event, "params")
        and "motion_file" in (reset_event.params or {})
    )
    env_needs_motion = (motion_command is not None) or event_needs_motion
    if env_needs_motion and motion_file_path is None:
        raise ValueError(
            "Env requires a motion file (commands.motion or events.reset_base) "
            "but --registry_name was not provided."
        )
    if motion_command is not None and motion_file_path is not None:
        motion_command.motion_file = motion_file_path
    if event_needs_motion and motion_file_path is not None:
        reset_event.params["motion_file"] = motion_file_path

    # --- Substitute obstacle USD if the env has one and a USD was given ---
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
        print(f"[INFO] Obstacle USD: {usd_path}")

    # Spatial alignment is handled by the motion clip + reset event (see
    # collect_fql_dataset.py for the full reasoning). Use a no-offset USD.

    # --- Env ---
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = env.unwrapped.device

    # --- Load PPO teacher ---
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(device))
    ppo_runner.load(ppo_resume_path)
    ppo_policy = ppo_runner.get_inference_policy(device=device)
    print(f"[INFO] PPO teacher loaded.")

    # --- Load FQL student + normalizer ---
    fql_ckpt = torch.load(args_cli.fql_checkpoint, weights_only=False, map_location=device)
    norm_path = args_cli.normalizer or os.path.join(os.path.dirname(args_cli.fql_checkpoint), "normalizer.pt")
    if not os.path.isfile(norm_path):
        raise FileNotFoundError(f"normalizer.pt not found at {norm_path}.")
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
    print(f"[INFO] FQL student loaded (iter {fql_ckpt.get('iter', '?')}).")

    def fql_act(student_obs: torch.Tensor) -> torch.Tensor:
        obs_n = ((student_obs - obs_mean) / obs_std).clamp(-obs_clip, obs_clip)
        if args_cli.use_flow:
            action_n = fql_agent.act_flow(obs_n)
        else:
            action_n = fql_agent.act(obs_n, deterministic_z=args_cli.deterministic_fql_z)
        return action_n * action_std + action_mean

    # --- Probe obs structure ---
    obs = env.get_observations()
    if "policy" not in obs or "student" not in obs:
        raise RuntimeError(
            f"Env obs groups are {list(obs.keys())}; need both 'policy' (PPO) and 'student' (FQL)."
        )
    num_envs = env.unwrapped.num_envs
    student_dim = obs["student"].shape[-1]
    env_action_dim = env.unwrapped.action_manager.total_action_dim
    print(f"[INFO] num_envs={num_envs}  student_obs_dim={student_dim}  action_dim={env_action_dim}")

    # --- Preallocate dataset buffers (CPU, plain — pinned memory blows the
    # CUDA pool at multi-GB sizes). ---
    target = int(args_cli.num_transitions)
    num_steps = (target + num_envs - 1) // num_envs
    capacity = num_steps * num_envs
    print(f"[INFO] Collecting {capacity:,} DAgger transitions over {num_steps} env steps.")

    obs_buf = torch.empty((capacity, student_dim), dtype=torch.float32, device="cpu")
    next_obs_buf = torch.empty((capacity, student_dim), dtype=torch.float32, device="cpu")
    action_buf = torch.empty((capacity, env_action_dim), dtype=torch.float32, device="cpu")  # PPO's actions
    reward_buf = torch.empty((capacity,), dtype=torch.float32, device="cpu")
    done_buf = torch.empty((capacity,), dtype=torch.bool, device="cpu")

    # Running stats for sanity reporting
    sum_diff = 0.0
    sum_abs_ppo = 0.0
    sum_abs_fql = 0.0

    t0 = time.time()
    write = 0
    with torch.inference_mode():
        for step in range(num_steps):
            s_student = obs["student"]

            # Both policies act on the same state
            a_ppo = ppo_policy(obs)            # teacher action (logged)
            a_fql = fql_act(s_student)         # student action (drives env)

            # Env transitions under the STUDENT's action — this is what makes
            # this DAgger and not vanilla expert collection.
            obs, reward, done, _ = env.step(a_fql)

            sl = slice(write, write + num_envs)
            obs_buf[sl].copy_(s_student, non_blocking=True)
            next_obs_buf[sl].copy_(obs["student"], non_blocking=True)
            action_buf[sl].copy_(a_ppo, non_blocking=True)   # ← PPO's action, not FQL's
            reward_buf[sl].copy_(reward.float(), non_blocking=True)
            done_buf[sl].copy_(done.bool(), non_blocking=True)
            write += num_envs

            sum_diff += (a_ppo - a_fql).pow(2).mean().sqrt().item()
            sum_abs_ppo += a_ppo.abs().mean().item()
            sum_abs_fql += a_fql.abs().mean().item()

            if (step + 1) % 50 == 0 or step == num_steps - 1:
                elapsed = time.time() - t0
                fps = (step + 1) * num_envs / elapsed
                print(f"  [{step+1:>5d}/{num_steps}]  {write:,} transitions  "
                      f"{fps:,.0f} env-steps/s  done={done.float().mean().item():.3f}  "
                      f"reward={reward.float().mean().item():+.3f}  "
                      f"|a_ppo-a_fql|={sum_diff/(step+1):.3f}")

    env.close()

    # --- Save in the same format as collect_fql_dataset.py ---
    os.makedirs(args_cli.output_dir, exist_ok=True)
    output_path = os.path.join(args_cli.output_dir, f"{args_cli.output_name}.pt")

    torch.save({
        "observations": obs_buf,
        "actions": action_buf,                 # ← PPO's actions (teacher labels)
        "rewards": reward_buf,
        "next_observations": next_obs_buf,
        "dones": done_buf,
        "metadata": {
            "task": args_cli.task,
            "kind": "dagger",                   # marks this as DAgger augmentation
            "fql_checkpoint": args_cli.fql_checkpoint,
            "ppo_checkpoint": ppo_resume_path,
            "num_envs": num_envs,
            "num_steps": num_steps,
            "student_obs_dim": student_dim,
            "action_dim": env_action_dim,
            "fql_mode": "flow" if args_cli.use_flow else ("onestep_det" if args_cli.deterministic_fql_z else "onestep_stoch"),
            "step_dt": float(env.unwrapped.step_dt),
        },
    }, output_path)

    n = num_steps
    print()
    print(f"[INFO] Saved {capacity:,} DAgger transitions to {output_path}  "
          f"({os.path.getsize(output_path) / 1e6:.1f} MB)")
    print(f"[INFO] Aggregate teacher-student divergence:")
    print(f"         mean |a_PPO - a_FQL|  = {sum_diff/n:.3f}  "
          f"(closer to 0 = student already matches teacher; closer to 1+ = lots of correction signal)")
    print(f"         mean |a_PPO|          = {sum_abs_ppo/n:.3f}")
    print(f"         mean |a_FQL|          = {sum_abs_fql/n:.3f}")
    print(f"[INFO] Wall time: {(time.time() - t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
    simulation_app.close()

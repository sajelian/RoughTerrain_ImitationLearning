"""Train a Flow-Q-Learning agent on collected expert rollouts.

Purely offline — no Isaac Sim. Loads .pt files produced by
`collect_fql_dataset.py`, runs gradient updates on flow / one-step / Q,
saves checkpoints and the obs-normalization stats needed for deployment.

Example:
    python scripts/rsl_rl/train_fql.py \\
        --datasets data/fql_rollouts/box_up_box_up.pt \\
        --num_iterations 200000 \\
        --batch_size 4096 \\
        --output_dir logs/fql/box_up
"""

from __future__ import annotations

import argparse
import os
import time

import torch

# importable without Isaac Sim
from RT_IL.algorithms.fql import FQLAgent, FQLConfig, FQLReplayBuffer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", required=True,
                   help="One or more .pt files from collect_fql_dataset.py")
    p.add_argument("--num_iterations", type=int, default=1_000_000,
                   help="Paper default: 1M gradient steps.")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Paper default: 256.")
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=50_000)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--action_clip_pct", type=float, default=0.0,
                   help="Per-joint percentile to clip the action tail at "
                        "([pct, 100-pct]). Removes rare reactive transients that "
                        "poison the flow fit. 0 = no clipping.")
    p.add_argument("--reward_key", type=str, default="rewards_cmd",
                   choices=["rewards", "rewards_cmd"],
                   help="Which reward field to optimize. 'rewards'=motion tracking "
                        "(teacher's env reward, depends on hidden motion command). "
                        "'rewards_cmd'=command velocity tracking (depends only on "
                        "joystick + robot velocity; aligned with deploy objective).")

    # algorithm hyperparams (override defaults in FQLConfig)
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Paper uses 0.995 for humanoid/giant tasks; try that if 0.99 plateaus.")
    p.add_argument("--target_tau", type=float, default=0.005)
    p.add_argument("--flow_lr", type=float, default=3e-4)
    p.add_argument("--onestep_lr", type=float, default=3e-4)
    p.add_argument("--q_lr", type=float, default=3e-4)
    p.add_argument("--flow_integration_steps", type=int, default=10)
    p.add_argument("--alpha", type=float, default=3.0,
                   help="BC coefficient (the paper's single major knob). "
                        "Suggested: humanoid locomotion=30, contact-rich manipulation=300, "
                        "dexterous=3000-30000. Sweep {30, 100, 300} for new tasks.")
    p.add_argument("--q_normalize", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable batch-wise 1/mean(|Q|) normalization in the actor loss "
                        "(paper App. B recommendation for new tasks). "
                        "Use --no-q_normalize to disable. With normalization on, the paper's "
                        "recommended alpha range is {0.03, 0.1, 0.3, 1, 3, 10}; without it, "
                        "alpha must scale with the reward magnitude (paper uses 100-30000).")
    p.add_argument("--flow_hidden", type=int, nargs="+", default=[1024, 1024, 512, 512],
                   help="Wider than paper default ([512]×4) because our student obs is ~988-dim "
                        "(a bottleneck at layer-1=512 instead of expansion).")
    p.add_argument("--onestep_hidden", type=int, nargs="+", default=[1024, 1024, 512, 512])
    p.add_argument("--q_hidden", type=int, nargs="+", default=[1024, 1024, 512, 512])
    p.add_argument("--activation", type=str, default="gelu", choices=["gelu", "elu", "silu", "relu"])

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true",
                   help="Log training metrics to Weights & Biases.")
    p.add_argument("--wandb_project", type=str, default="RT_IL_FQL")
    p.add_argument("--wandb_name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- data ---
    buf = FQLReplayBuffer(paths=args.datasets, device=args.device, normalize_obs=True,
                          action_clip_pct=args.action_clip_pct, reward_key=args.reward_key)

    # --- agent ---
    cfg = FQLConfig(
        obs_dim=buf.obs_dim,
        action_dim=buf.action_dim,
        flow_hidden_dims=tuple(args.flow_hidden),
        onestep_hidden_dims=tuple(args.onestep_hidden),
        q_hidden_dims=tuple(args.q_hidden),
        activation=args.activation,
        gamma=args.gamma,
        target_tau=args.target_tau,
        flow_lr=args.flow_lr,
        onestep_lr=args.onestep_lr,
        q_lr=args.q_lr,
        flow_integration_steps=args.flow_integration_steps,
        alpha=args.alpha,
        q_weight_normalize=args.q_normalize,
        device=args.device,
    )
    agent = FQLAgent(cfg)

    # --- logging ---
    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   config={**vars(args), "obs_dim": buf.obs_dim, "action_dim": buf.action_dim})

    # --- save obs + action normalizer (needed at deploy time) ---
    torch.save({
        "obs_mean": buf.obs_mean,
        "obs_std": buf.obs_std,
        "obs_clip": buf.obs_clip,
        "obs_dim": buf.obs_dim,
        "action_dim": buf.action_dim,
        "action_mean": buf.action_mean,
        "action_std": buf.action_std,
    }, os.path.join(args.output_dir, "normalizer.pt"))

    # --- train ---
    t_start = time.time()
    running = {}
    for it in range(1, args.num_iterations + 1):
        batch = buf.sample(args.batch_size)
        info = agent.update(batch)
        for k, v in info.items():
            running[k] = running.get(k, 0.0) + v

        if it % args.log_interval == 0:
            denom = args.log_interval
            avg = {k: v / denom for k, v in running.items()}
            running.clear()
            elapsed = time.time() - t_start
            rate = it / elapsed
            msg = (f"[{it:>7d}/{args.num_iterations}]  {rate:.0f} it/s  "
                   f"flow={avg['flow_loss']:.4f}  q={avg['q_loss']:.4f}  "
                   f"bc={avg['bc_loss']:.4f}  policy_q={avg['policy_q']:+.4f}  "
                   f"q_target={avg['q_target_mean']:+.3f}")
            print(msg)
            if use_wandb:
                wandb.log({**avg, "iter": it, "it_per_sec": rate})

        if it % args.save_interval == 0 or it == args.num_iterations:
            ckpt_path = os.path.join(args.output_dir, f"agent_{it}.pt")
            torch.save({
                "agent": agent.state_dict(),
                "iter": it,
                "config": cfg.__dict__,
            }, ckpt_path)
            print(f"  [save] {ckpt_path}")

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()

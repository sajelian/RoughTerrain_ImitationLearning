"""Offline diagnostic for a trained FQL policy.

Loads agent + normalizer, samples real (s, a, r) transitions from the dataset,
and compares the agent's output to the recorded teacher action. If the agent
matches the data, training succeeded and any deploy failure is distribution
shift. If it doesn't match, training itself failed (more steps, bigger nets,
or different α).

Example:
    python scripts/rsl_rl/diagnose_fql.py \\
        --checkpoint logs/fql/box_up_alpha100/agent_100000.pt \\
        --dataset data/fql_rollouts/box_up_flat_box_convex.pt
"""

import argparse
import os

import torch

from RT_IL.algorithms.fql import FQLAgent, FQLConfig, FQLReplayBuffer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--normalizer", default=None)
    p.add_argument("--dataset", required=True, help="Same .pt used for training.")
    p.add_argument("--num_samples", type=int, default=2048)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    # --- Load checkpoint ---
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=args.device)
    norm_path = args.normalizer or os.path.join(os.path.dirname(args.checkpoint), "normalizer.pt")
    normalizer = torch.load(norm_path, weights_only=False, map_location=args.device)

    obs_mean = normalizer["obs_mean"].to(args.device)
    obs_std = normalizer["obs_std"].to(args.device)
    obs_clip = float(normalizer["obs_clip"])
    obs_dim = int(normalizer["obs_dim"])
    action_dim = int(normalizer["action_dim"])

    saved_cfg = ckpt.get("config", {})
    cfg = FQLConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        flow_hidden_dims=tuple(saved_cfg.get("flow_hidden_dims", (512, 512, 512, 512))),
        onestep_hidden_dims=tuple(saved_cfg.get("onestep_hidden_dims", (512, 512, 512, 512))),
        q_hidden_dims=tuple(saved_cfg.get("q_hidden_dims", (512, 512, 512, 512))),
        activation=saved_cfg.get("activation", "gelu"),
        device=args.device,
    )
    agent = FQLAgent(cfg)
    agent.load_state_dict(ckpt["agent"])
    agent.flow.eval(); agent.onestep.eval(); agent.q1.eval(); agent.q2.eval()
    print(f"[diag] checkpoint iter={ckpt.get('iter', '?')}  alpha={saved_cfg.get('alpha', '?')}")

    # --- Sample from the training dataset ---
    buf = FQLReplayBuffer(args.dataset, device=args.device, normalize_obs=True)
    batch = buf.sample(args.num_samples)

    # Compare policy outputs to recorded actions on the same states
    with torch.no_grad():
        # Same z, both methods
        z = torch.randn(args.num_samples, action_dim)
        a_onestep_z = agent.onestep(batch.obs, z)
        a_flow_z = agent.flow.integrate(batch.obs, z, num_steps=10)
        # Deterministic z=0
        z0 = torch.zeros(args.num_samples, action_dim)
        a_onestep_z0 = agent.onestep(batch.obs, z0)
        a_flow_z0 = agent.flow.integrate(batch.obs, z0, num_steps=10)
        # Random z for stochasticity check
        z2 = torch.randn(args.num_samples, action_dim)
        a_onestep_z2 = agent.onestep(batch.obs, z2)

        a_data = batch.action

        def stats(name, a, ref=None):
            mag = a.abs().mean().item()
            r = a.std().item()
            line = f"  {name:22s}  |a|_mean={mag:.4f}  std={r:.4f}"
            if ref is not None:
                diff = (a - ref).pow(2).mean().sqrt().item()
                cosine = torch.nn.functional.cosine_similarity(a, ref, dim=-1).mean().item()
                line += f"  rmse_vs_ref={diff:.4f}  cos_sim={cosine:+.3f}"
            print(line)

        print(f"\n[Action comparison on {args.num_samples} real (s, a) pairs from the training set]")
        stats("dataset (target)", a_data)
        stats("flow det z=0",     a_flow_z0,      a_data)
        stats("flow stochastic", a_flow_z,        a_data)
        stats("one-step det z=0", a_onestep_z0,   a_data)
        stats("one-step stoch",  a_onestep_z,     a_data)
        stats("one-step stoch'", a_onestep_z2,    a_data)
        # Stochasticity check: does z affect output?
        z_effect = (a_onestep_z - a_onestep_z2).pow(2).mean().sqrt().item()
        print(f"\n[Stochasticity] RMSE between one-step under two different z draws: {z_effect:.4f}")
        print("  (≈0 → policy ignores z; >0.01 → z affects output)")

        # Per-dimension correlation with data
        a_pred = a_onestep_z0
        per_dim_rmse = (a_pred - a_data).pow(2).mean(0).sqrt()
        worst = per_dim_rmse.argsort(descending=True)[:5]
        print(f"\n[Per-dim error] worst 5 action dims (one-step z=0 vs data):")
        for k in worst:
            print(f"  dim {int(k):3d}: rmse={per_dim_rmse[k].item():.4f}  "
                  f"data_range=[{a_data[:,k].min():+.3f}, {a_data[:,k].max():+.3f}]  "
                  f"pred_range=[{a_pred[:,k].min():+.3f}, {a_pred[:,k].max():+.3f}]")

        # Look specifically at frames where data action is large
        large_idx = a_data.abs().mean(dim=-1).argsort(descending=True)[:200]
        a_data_large = a_data[large_idx]
        a_pred_large = a_pred[large_idx]
        print(f"\n[Large-action subset: top 200 by |a|_mean]")
        stats("  data (large)",     a_data_large)
        stats("  one-step on same", a_pred_large, a_data_large)

        # Joystick analysis: split by vx label (forward vs stop)
        # joystick obs is normalized — we need to read the unnormalized to recover the label.
        # Convention: first 3 dims of student obs are the most-recent joystick frame.
        # Reverse-normalize using saved obs_mean/std for those 3 dims.
        # Recover unnormalized from the BATCH (which is already normalized + clipped)
        # → tricky if clip kicked in. Easier: re-sample raw values from the buffer.
        idx = torch.randint(0, buf.n, (args.num_samples,))
        raw_obs = buf.observations[idx]
        js_vx = raw_obs[:, 0]   # first dim of unnormalized obs
        forward_mask = js_vx > 0.4   # >0.4 → forward label (the values are 0 or 0.8)
        stop_mask = ~forward_mask
        print(f"\n[Joystick split in sampled subset] forward={forward_mask.sum().item()} stop={stop_mask.sum().item()}")
        if forward_mask.any():
            obs_fwd = buf.normalize_obs(raw_obs[forward_mask].to(args.device))
            a_fwd_pred = agent.onestep(obs_fwd, torch.zeros(forward_mask.sum().item(), action_dim))
            a_fwd_data = buf.actions[idx[forward_mask]]
            print(f"  forward subset:  pred |a|={a_fwd_pred.abs().mean():.4f}   data |a|={a_fwd_data.abs().mean():.4f}")
        if stop_mask.any():
            obs_stop = buf.normalize_obs(raw_obs[stop_mask].to(args.device))
            a_stop_pred = agent.onestep(obs_stop, torch.zeros(stop_mask.sum().item(), action_dim))
            a_stop_data = buf.actions[idx[stop_mask]]
            print(f"  stop subset:     pred |a|={a_stop_pred.abs().mean():.4f}   data |a|={a_stop_data.abs().mean():.4f}")


if __name__ == "__main__":
    main()

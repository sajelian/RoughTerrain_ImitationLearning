"""Measure how much each obs group contributes to the trained FQL policy's actions.

For each obs group (joystick / proprio / height_scan), we randomize that part
of the obs across the dataset's distribution and measure how much the
policy's action changes. Large change = policy uses that group. Tiny change =
policy ignores it.

Usage:
    python scripts/rsl_rl/probe_obs_sensitivity.py \\
        --checkpoint logs/fql/box_up_..../agent_N.pt \\
        --dataset data/fql_rollouts/box_up_flat_box_convex.pt
"""

import argparse
import os

import torch

from RT_IL.algorithms.fql import FQLAgent, FQLConfig, FQLReplayBuffer

# Layout assumed for ObservationsCfg_JOYSTICK_DISTILLATION.StudentPolicyCfg with
# concatenate_terms=True. Order matches the field declaration order in tracking_env_cfg.py.
# Each entry: (group_name, per_frame_dim, history_length).
OBS_LAYOUT = [
    ("joystick",        3,  5),
    ("base_ang_vel",    3,  8),
    ("projected_grav",  3,  5),
    ("joint_pos",       23, 8),
    ("joint_vel",       23, 8),
    ("actions",         23, 5),
    # remainder → height_scan (no history, no time dim)
]


def group_slices(total_obs_dim: int):
    out, cur = {}, 0
    for name, d, h in OBS_LAYOUT:
        sz = d * h
        out[name] = slice(cur, cur + sz)
        cur += sz
    if cur < total_obs_dim:
        out["height_scan"] = slice(cur, total_obs_dim)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--normalizer", default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--num_samples", type=int, default=4096)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=args.device)
    norm_path = args.normalizer or os.path.join(os.path.dirname(args.checkpoint), "normalizer.pt")
    norm = torch.load(norm_path, weights_only=False, map_location=args.device)

    obs_dim, action_dim = int(norm["obs_dim"]), int(norm["action_dim"])
    saved_cfg = ckpt.get("config", {})
    cfg = FQLConfig(
        obs_dim=obs_dim, action_dim=action_dim,
        flow_hidden_dims=tuple(saved_cfg.get("flow_hidden_dims", (1024, 1024, 512, 512))),
        onestep_hidden_dims=tuple(saved_cfg.get("onestep_hidden_dims", (1024, 1024, 512, 512))),
        q_hidden_dims=tuple(saved_cfg.get("q_hidden_dims", (1024, 1024, 512, 512))),
        activation=saved_cfg.get("activation", "gelu"),
        layer_norm_q=saved_cfg.get("layer_norm_q", True),
        device=args.device,
    )
    agent = FQLAgent(cfg)
    agent.load_state_dict(ckpt["agent"])
    agent.onestep.eval()

    buf = FQLReplayBuffer(args.dataset, device=args.device, normalize_obs=True)
    batch = buf.sample(args.num_samples)
    obs_base = batch.obs              # already normalized
    z = torch.zeros(args.num_samples, action_dim)

    with torch.no_grad():
        a_base = agent.onestep(obs_base, z)

    slices = group_slices(obs_dim)
    print(f"[probe] obs_dim={obs_dim}  groups:")
    for name, sl in slices.items():
        print(f"  {name:18s} dims [{sl.start}:{sl.stop}]  size={sl.stop-sl.start}")

    # Sensitivity: replace one group with a permuted version (within the batch) — preserves the
    # marginal distribution of that group but breaks its correlation with the rest. Then measure
    # how much the action changes.
    print(f"\n[sensitivity] action RMSE when each group is shuffled within the batch:")
    print(f"  {'group':18s}  rmse_dA   relative-to-baseline")
    a_baseline_rmse = a_base.std().item()  # action variability if input doesn't change
    print(f"  {'(baseline: a.std)':18s}  {a_baseline_rmse:.4f}")
    for name, sl in slices.items():
        perm = torch.randperm(args.num_samples)
        obs_perturbed = obs_base.clone()
        obs_perturbed[:, sl] = obs_base[perm, sl]
        with torch.no_grad():
            a_perturbed = agent.onestep(obs_perturbed, z)
        dA = (a_perturbed - a_base).pow(2).mean().sqrt().item()
        print(f"  {name:18s}  {dA:.4f}     {dA/a_baseline_rmse:.2f}x baseline")

    # Extra: zero out each group (set to all zeros in normalized space = "mean state").
    print(f"\n[ablation] action RMSE when each group is zeroed (set to mean):")
    print(f"  {'group':18s}  rmse_dA")
    for name, sl in slices.items():
        obs_perturbed = obs_base.clone()
        obs_perturbed[:, sl] = 0.0
        with torch.no_grad():
            a_perturbed = agent.onestep(obs_perturbed, z)
        dA = (a_perturbed - a_base).pow(2).mean().sqrt().item()
        print(f"  {name:18s}  {dA:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Multi-expert DAgger orchestrator.

Reads a YAML config defining a set of (expert, env, USD, teacher checkpoint)
pairings and runs the DAgger loop:

    for round in 1..MAX_ROUNDS:
        for expert in experts:
            collect_dagger_dataset.py  --> data/fql_rollouts/dagger_round_N_<expert>.pt
        train_fql.py --datasets <all .pt>  --> logs/.../round_N/agent_*.pt
        advance CURRENT_FQL to the new checkpoint

Resumes automatically: if a collection file for a (round, expert) already
exists, that expert is skipped. So killing and restarting the loop is safe.

Example:
    python scripts/dagger_orchestrator.py --config configs/dagger_multi_expert.yaml

Resume from round 3:
    python scripts/dagger_orchestrator.py --config configs/... --start_round 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML not installed. pip install pyyaml")


# ----------------------------------------------------------------------------
# Config dataclasses
# ----------------------------------------------------------------------------
@dataclass
class ExpertCfg:
    name: str                                # short identifier; used in output filenames
    task: str                                # gym task id
    wandb_path: str                          # W&B run path for the PPO teacher checkpoint
    usd_name: str | None = None              # obstacle USD (None for velocity env)
    registry_name: str | None = None         # W&B motion artifact (None for velocity env)
    teacher_cfg: str = "ppo"                 # "ppo" or "ppo_velocity"
    transitions: int = 1_000_000             # data quantity per round for this expert
    skill_mode: float | None = None          # override the student-obs skill bit (0=velocity regime,
                                             # 1=skill regime); None = use the env cfg default
    difficulty: float | None = None          # freeze the robustness curriculum at this difficulty
                                             # (1.0 = full spawn randomization the expert trained
                                             # against); None = env cfg default (no override)
    extra_args: list[str] = field(default_factory=list)  # passed through to collect script


@dataclass
class LoopCfg:
    max_rounds: int = 3
    initial_fql_checkpoint: str = ""
    data_dir: str = "data/fql_rollouts"
    log_dir: str = "logs/fql/dagger_multi"
    train_iterations: int = 300_000
    alpha: float = 3.0
    max_total_transitions: int = 10_000_000
    num_envs: int = 2048
    headless: bool = True
    wandb: bool = True
    wandb_project: str = "RT_IL_FQL"


# ----------------------------------------------------------------------------
# Loading + subprocess helpers
# ----------------------------------------------------------------------------
def load_config(path: str) -> tuple[list[ExpertCfg], LoopCfg]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "experts" not in cfg:
        sys.exit(f"Config {path} missing required 'experts' list.")
    experts = [ExpertCfg(**e) for e in cfg["experts"]]
    loop = LoopCfg(**cfg.get("loop", {}))
    return experts, loop


# ----------------------------------------------------------------------------
# RAM preflight
# ----------------------------------------------------------------------------
# FQLReplayBuffer footprint per transition (bytes), at obs_dim≈988 / action_dim≈29:
#   obs fp16 + next_obs fp16 + action fp16 + reward fp32 + done fp32
#   = 988*2 + 988*2 + 29*2 + 4 + 4  ≈ 4022 B  ≈ 4 GB per 1M transitions
def estimate_ram_gb(n_transitions: int, obs_dim: int, action_dim: int) -> float:
    bytes_per_row = (2 * obs_dim * 2) + (action_dim * 2) + 4 + 4
    return n_transitions * bytes_per_row / 1e9


def get_available_ram_gb() -> float:
    """Available RAM in GB. Returns -1 if it can't be determined."""
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except FileNotFoundError:
        pass
    return -1.0


def peek_dataset_sizes(data_dir: Path) -> tuple[int, int, int]:
    """Sum transitions across existing .pt files in data_dir. Returns (total_n, obs_dim, action_dim).
    Uses mmap to avoid loading the data."""
    import torch
    total_n = 0
    obs_dim = action_dim = 0
    for p in sorted(data_dir.glob("*.pt")):
        try:
            data = torch.load(str(p), weights_only=False, map_location="cpu", mmap=True)
            total_n += int(data["observations"].shape[0])
            obs_dim = int(data["observations"].shape[-1])
            action_dim = int(data["actions"].shape[-1])
        except Exception as e:
            print(f"  WARNING: could not peek {p}: {e}")
    return total_n, obs_dim, action_dim


def preflight_ram_check(experts: list[ExpertCfg], loop: LoopCfg, strict: bool) -> None:
    """Estimate end-of-loop RAM footprint and warn (or fail) if it won't fit."""
    data_dir = Path(loop.data_dir)
    if not data_dir.exists():
        existing_n, obs_dim, action_dim = 0, 988, 29
    else:
        existing_n, obs_dim, action_dim = peek_dataset_sizes(data_dir)
        if obs_dim == 0:
            obs_dim, action_dim = 988, 29  # sensible default if no existing files

    per_round = sum(e.transitions for e in experts)
    final_n_raw = existing_n + loop.max_rounds * per_round
    loaded_n = min(final_n_raw, loop.max_total_transitions)
    ram_gb = estimate_ram_gb(loaded_n, obs_dim, action_dim)
    available_gb = get_available_ram_gb()

    print(f"\n{'─'*72}")
    print(f"[preflight] RAM check")
    print(f"  obs_dim={obs_dim}  action_dim={action_dim}")
    print(f"  Existing data in {data_dir}: {existing_n:>15,} transitions ({len(list(data_dir.glob('*.pt')))} files)")
    print(f"  Per-round addition (sum across {len(experts)} experts): {per_round:>15,}")
    print(f"  Projected raw total after {loop.max_rounds} rounds:  {final_n_raw:>15,}")
    print(f"  max_total_transitions cap:                {loop.max_total_transitions:>15,}")
    print(f"  → effectively loaded into buffer:         {loaded_n:>15,}")
    print(f"  → estimated buffer RAM:                   {ram_gb:>15.1f} GB")
    if available_gb > 0:
        print(f"  Available RAM right now:                  {available_gb:>15.1f} GB")
        headroom_frac = ram_gb / available_gb
        if headroom_frac > 0.85:
            msg = (f"RAM estimate ({ram_gb:.1f} GB) is {100*headroom_frac:.0f}% of "
                   f"available ({available_gb:.1f} GB) — high OOM risk.")
            suggested_cap = int(available_gb * 0.65 / (estimate_ram_gb(1_000_000, obs_dim, action_dim)) * 1_000_000)
            print(f"  ⚠️  {msg}")
            print(f"  ⚠️  Suggested: lower loop.max_total_transitions to ≤ {suggested_cap:,}")
            if strict:
                sys.exit(f"\n[preflight] Aborting due to RAM risk (--strict). "
                         f"Edit your YAML's loop.max_total_transitions and rerun.")
            else:
                print(f"  ⚠️  Continuing (pass --strict to fail-fast on this check).")
        else:
            print(f"  ✓ Fits comfortably ({100*headroom_frac:.0f}% of available).")
    else:
        print(f"  (could not determine available RAM — psutil not installed and /proc/meminfo unreadable)")
    print(f"{'─'*72}\n")


def run_subprocess(cmd: list[str], description: str) -> None:
    """Run a subprocess inheriting stdout/stderr. Raise on nonzero exit."""
    print(f"\n>>> {description}")
    print(f"    {' '.join(cmd)}\n")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"{description} failed (exit code {rc})")


# ----------------------------------------------------------------------------
# Per-step orchestration
# ----------------------------------------------------------------------------
def collect_one_expert(
    expert: ExpertCfg, loop: LoopCfg, current_fql: str, round_n: int
) -> Path:
    """Run DAgger collection for one expert. Returns the output .pt path.
    Skips (no-op) if the file already exists — supports resuming."""
    output_name = f"dagger_round_{round_n}_{expert.name}"
    output_file = Path(loop.data_dir) / f"{output_name}.pt"

    if output_file.exists():
        print(f"\n>>> [SKIP] {output_file} already exists (resuming).")
        return output_file

    cmd = [
        "python", "scripts/rsl_rl/collect_dagger_dataset.py",
        "--task", expert.task,
        "--wandb_path", expert.wandb_path,
        "--fql_checkpoint", current_fql,
        "--teacher_cfg", expert.teacher_cfg,
        "--output_dir", loop.data_dir,
        "--output_name", output_name,
        "--num_envs", str(loop.num_envs),
        "--num_transitions", str(expert.transitions),
    ]
    if expert.usd_name:
        cmd.extend(["--usd_name", expert.usd_name])
    if expert.registry_name:
        cmd.extend(["--registry_name", expert.registry_name])
    if expert.skill_mode is not None:
        cmd.extend(["--skill_mode", str(expert.skill_mode)])
    if expert.difficulty is not None:
        cmd.extend(["--difficulty", str(expert.difficulty)])
    if loop.headless:
        cmd.append("--headless")
    cmd.extend(expert.extra_args)

    run_subprocess(cmd, f"DAgger collect [round {round_n}] expert='{expert.name}'")

    if not output_file.exists():
        raise RuntimeError(
            f"Expected output {output_file} not produced by collection script."
        )
    return output_file


def train_fql_step(loop: LoopCfg, round_n: int) -> Path:
    """Train FQL on the union of every .pt in data_dir. Returns final checkpoint path."""
    data_dir = Path(loop.data_dir)
    all_pt = sorted(data_dir.glob("*.pt"))
    if not all_pt:
        raise RuntimeError(f"No .pt files in {data_dir}; nothing to train on.")

    output_dir = Path(loop.log_dir) / f"round_{round_n}"
    output_dir.mkdir(parents=True, exist_ok=True)
    new_ckpt = output_dir / f"agent_{loop.train_iterations}.pt"

    if new_ckpt.exists():
        print(f"\n>>> [SKIP] {new_ckpt} already exists (resuming).")
        return new_ckpt

    cmd = [
        "python", "scripts/rsl_rl/train_fql.py",
        "--datasets", *[str(p) for p in all_pt],
        "--alpha", str(loop.alpha),
        "--num_iterations", str(loop.train_iterations),
        "--max_total_transitions", str(loop.max_total_transitions),
        "--output_dir", str(output_dir),
    ]
    if loop.wandb:
        cmd.extend(["--wandb",
                    "--wandb_project", loop.wandb_project,
                    "--wandb_name", f"dagger_round_{round_n}"])

    run_subprocess(cmd, f"FQL train [round {round_n}] (union of {len(all_pt)} files)")

    if not new_ckpt.exists():
        raise RuntimeError(f"Expected checkpoint {new_ckpt} not produced.")
    return new_ckpt


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to experts YAML config.")
    p.add_argument("--start_round", type=int, default=1,
                   help="Round number to start at (>=1). Use for resuming.")
    p.add_argument("--current_fql", type=str, default=None,
                   help="Override the FQL checkpoint to start from (defaults to the "
                        "config's initial_fql_checkpoint, or the previous round's checkpoint "
                        "if --start_round > 1).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be run but execute nothing.")
    p.add_argument("--strict", action="store_true",
                   help="Abort if the RAM preflight predicts OOM risk (>85% of available).")
    p.add_argument("--skip_preflight", action="store_true",
                   help="Skip the RAM check (not recommended).")
    args = p.parse_args()

    experts, loop = load_config(args.config)
    print(f"[orchestrator] Loaded {len(experts)} experts from {args.config}")
    for e in experts:
        print(f"  - {e.name:<14}  task={e.task}  usd={e.usd_name}  "
              f"transitions={e.transitions:,}")
    print(f"[orchestrator] Loop: max_rounds={loop.max_rounds}  "
          f"alpha={loop.alpha}  max_total_transitions={loop.max_total_transitions:,}")

    # Resolve starting FQL checkpoint
    if args.current_fql:
        current_fql = args.current_fql
    elif args.start_round > 1:
        # If resuming, expect the previous round's checkpoint to exist
        prev = Path(loop.log_dir) / f"round_{args.start_round - 1}" / f"agent_{loop.train_iterations}.pt"
        if not prev.is_file():
            sys.exit(f"Resuming from round {args.start_round} but previous-round checkpoint "
                     f"not found: {prev}\nPass --current_fql to override.")
        current_fql = str(prev)
        print(f"[orchestrator] Resuming from previous round's checkpoint: {current_fql}")
    else:
        current_fql = loop.initial_fql_checkpoint

    if not Path(current_fql).is_file():
        sys.exit(f"FQL checkpoint not found: {current_fql}")

    Path(loop.data_dir).mkdir(parents=True, exist_ok=True)
    Path(loop.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(loop.log_dir) / "_orchestrator_log.txt"

    # RAM preflight — done BEFORE any subprocess work so we fail-fast at startup.
    if not args.skip_preflight:
        preflight_ram_check(experts, loop, strict=args.strict)

    if args.dry_run:
        print("\n[DRY RUN] would execute:")
        for round_n in range(args.start_round, loop.max_rounds + 1):
            print(f"  round {round_n}:")
            for e in experts:
                fp = Path(loop.data_dir) / f"dagger_round_{round_n}_{e.name}.pt"
                tag = "skip (exists)" if fp.exists() else "collect"
                print(f"    [{tag}] {e.name} -> {fp}")
            print(f"    train_fql.py on union of *.pt -> logs/.../round_{round_n}/")
        return

    t_start = time.time()
    for round_n in range(args.start_round, loop.max_rounds + 1):
        round_t0 = time.time()
        print(f"\n{'='*72}")
        print(f" DAgger ROUND {round_n} / {loop.max_rounds}")
        print(f" Current FQL: {current_fql}")
        print(f" Experts to collect: {[e.name for e in experts]}")
        print(f"{'='*72}")

        for expert in experts:
            try:
                collect_one_expert(expert, loop, current_fql, round_n)
            except RuntimeError as e:
                print(f"\n[ERROR] {e}", file=sys.stderr)
                sys.exit(1)

        try:
            new_ckpt = train_fql_step(loop, round_n)
        except RuntimeError as e:
            print(f"\n[ERROR] {e}", file=sys.stderr)
            sys.exit(1)

        current_fql = str(new_ckpt)
        round_dt = (time.time() - round_t0) / 60

        # Append to round log
        with open(log_path, "a") as f:
            f.write(f"Round {round_n} ({time.strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"{round_dt:.1f} min):\n")
            f.write(f"  FQL: {current_fql}\n")
            for e in experts:
                fp = Path(loop.data_dir) / f"dagger_round_{round_n}_{e.name}.pt"
                f.write(f"  Data ({e.name}): {fp}\n")
            f.write("\n")

        print(f"\n>>> Round {round_n} complete in {round_dt:.1f} min.")
        print(f">>> New checkpoint: {current_fql}")
        print(f">>> Quick verify (in another shell):")
        print(f"      python scripts/rsl_rl/compare_ppo_fql.py \\")
        print(f"          --task Velocity-G1-Play-v0 \\")
        print(f"          --wandb_path <ppo_velocity_run> \\")
        print(f"          --fql_checkpoint {current_fql} \\")
        print(f"          --drive_with fql --num_steps 500")

    total_dt = (time.time() - t_start) / 60
    print(f"\n{'='*72}")
    print(f" Orchestrator finished. Total wall time: {total_dt:.1f} min.")
    print(f" Final FQL: {current_fql}")
    print(f" Round log: {log_path}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()

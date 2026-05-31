"""Inspect FQL rollout files and emit a ready-to-paste `--datasets` argument.

Multi-expert datasets accumulate quickly (each expert × each terrain = one .pt
file). Listing them all on the train_fql.py command line gets unwieldy. This
script lets you point at a directory (or any mix of globs / files), get a
per-file summary, and either copy-paste the joined `--datasets` list or write
it to a small file that you can `--datasets $(cat manifest.txt)` from.

Examples:
    # Summarize everything in a directory
    python scripts/rsl_rl/aggregate_datasets.py data/fql_rollouts/

    # Filter by glob and emit the ready-to-paste args
    python scripts/rsl_rl/aggregate_datasets.py 'data/fql_rollouts/box_up*.pt' --emit_args

    # Write a manifest text file
    python scripts/rsl_rl/aggregate_datasets.py data/fql_rollouts/ --write_manifest dataset.txt
    # then use it:
    python scripts/rsl_rl/train_fql.py --datasets $(cat dataset.txt) ...
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch


def resolve_paths(specs: list[str]) -> list[str]:
    """Expand directories / globs / explicit files into a deduped sorted file list."""
    seen: set[str] = set()
    files: list[str] = []
    for s in specs:
        if os.path.isdir(s):
            matches = sorted(glob.glob(os.path.join(s, "*.pt")))
        elif os.path.isfile(s):
            matches = [s]
        else:
            matches = sorted(glob.glob(s))
            if not matches:
                print(f"WARNING: no files matched '{s}'", file=sys.stderr)
        for m in matches:
            if m not in seen:
                seen.add(m)
                files.append(m)
    return files


def inspect(path: str) -> dict | None:
    try:
        data = torch.load(path, weights_only=False, map_location="cpu", mmap=True)
    except Exception as e:
        return {"path": path, "error": str(e)}
    md = data.get("metadata", {})
    n = int(data["observations"].shape[0])
    obs_dim = int(data["observations"].shape[-1])
    action_dim = int(data["actions"].shape[-1])
    # Probe stats from a subsample to keep this fast over many files.
    n_probe = min(n, 100_000)
    idx = torch.randint(0, n, (n_probe,))
    reward = float(data["rewards"][idx].float().mean())
    done_rate = float(data["dones"][idx].float().mean())
    return {
        "path": path,
        "n": n,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "reward": reward,
        "done_rate": done_rate,
        "task": str(md.get("task", "?")),
        "usd": str(md.get("usd_name", "?")),
        "registry": str(md.get("registry_name", "?")).split("/")[-1] if md.get("registry_name") else "-",
        "noise": float(md.get("action_noise_std", 0.0)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+",
                   help="Directories, .pt files, or glob patterns.")
    p.add_argument("--check_compat", action="store_true",
                   help="Error out if obs_dim/action_dim differ across files.")
    p.add_argument("--emit_args", action="store_true",
                   help="After the table, print a ready-to-paste `--datasets ...` string.")
    p.add_argument("--write_manifest", type=str, default=None,
                   help="Write the resolved file list (space-separated) to this path.")
    args = p.parse_args()

    files = resolve_paths(args.paths)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    rows = [inspect(f) for f in files]

    # Print table
    name_w = max(len(os.path.basename(f)) for f in files)
    name_w = min(max(name_w, 12), 60)
    hdr = (f"{'file':<{name_w}}  {'N':>11}  {'obs':>4}  {'act':>3}  "
           f"{'reward':>8}  {'done%':>6}  {'noise':>5}  task / usd")
    print(hdr)
    print("-" * len(hdr))
    total_n = 0
    obs_dim_ref = action_dim_ref = None
    incompat = False
    for r in rows:
        bn = os.path.basename(r["path"])[-name_w:]
        if "error" in r:
            print(f"{bn:<{name_w}}  ERROR: {r['error']}")
            continue
        if obs_dim_ref is None:
            obs_dim_ref, action_dim_ref = r["obs_dim"], r["action_dim"]
        compat = (r["obs_dim"] == obs_dim_ref and r["action_dim"] == action_dim_ref)
        if not compat:
            incompat = True
        compat_mark = "" if compat else " !"
        # Show task short name (last hyphen-segment) + USD when present.
        task_short = r["task"].split("-")[-1] if r["task"] != "?" else r["registry"]
        label = f"{task_short} / {r['usd']}" if r["usd"] != "?" else task_short
        print(f"{bn:<{name_w}}  {r['n']:>11,}  {r['obs_dim']:>4}  {r['action_dim']:>3}  "
              f"{r['reward']:>+8.3f}  {100*r['done_rate']:>5.1f}%  "
              f"{r['noise']:>5.2f}  {label}{compat_mark}")
        total_n += r["n"]
    print("-" * len(hdr))
    print(f"Total: {len(files)} file(s), {total_n:,} transitions  "
          f"(obs={obs_dim_ref}, action={action_dim_ref})")

    if incompat:
        msg = "Found files with mismatched obs/action dims — cannot mix in one buffer."
        if args.check_compat:
            print(f"\nERROR: {msg}", file=sys.stderr)
            sys.exit(2)
        else:
            print(f"\nWARNING: {msg}  (rerun with --check_compat to fail)")

    if args.emit_args:
        good = [r["path"] for r in rows if "error" not in r]
        print(f"\n--datasets " + " ".join(good))

    if args.write_manifest:
        good = [r["path"] for r in rows if "error" not in r]
        with open(args.write_manifest, "w") as f:
            f.write(" ".join(good))
        print(f"\nManifest written to {args.write_manifest}  "
              f"(use:  --datasets $(cat {args.write_manifest})  )")


if __name__ == "__main__":
    main()

"""Visualize footlock contact schedules from raw motion NPZ files.

Usage:
    python scripts/data_handling/plot_footlocks.py data/motions/raw/beam_sequence.npz
    python scripts/data_handling/plot_footlocks.py data/motions/raw/beam_sequence.npz --save footlocks.png
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- pickle compatibility shim (NumPy internal module rename) ----
import numpy.core as _np_core
sys.modules.setdefault("numpy._core", _np_core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)
# -----------------------------------------------------------------


def plot_footlocks(npz_path: str, save_path: str | None = None, gait_only: bool = False):
    d = np.load(npz_path, allow_pickle=True)

    hard_left = d["egomc_footlock_LeftFoot"].astype(bool)
    hard_right = d["egomc_footlock_RightFoot"].astype(bool)
    soft_left = d["egomc_footlock_soft_LeftFoot"]
    soft_right = d["egomc_footlock_soft_RightFoot"]

    T = len(hard_left)
    frames = np.arange(T)

    # SMPLX standard: 7=left_ankle, 8=right_ankle, 10=left_foot, 11=right_foot
    joints = d["smplx_egomc_joints"]  # (T, 127, 3)
    left_foot_z = joints[:, 10, 2]
    right_foot_z = joints[:, 11, 2]
    left_ankle_z = joints[:, 7, 2]
    right_ankle_z = joints[:, 8, 2]

    if gait_only:
        fig, ax = plt.subplots(1, 1, figsize=(18, 4))
        fig.suptitle(f"Contact Schedule — {Path(npz_path).stem}", fontsize=14)
        ax.fill_between(frames, 0.5, 1.0, where=hard_left, alpha=0.5, color="tab:blue", label="left contact")
        ax.fill_between(frames, 0.0, 0.5, where=hard_right, alpha=0.5, color="tab:red", label="right contact")
        ax.set_yticks([0.25, 0.75])
        ax.set_yticklabels(["Right", "Left"])
        ax.set_xlabel("Frame")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
    else:
        fig, axes = plt.subplots(4, 1, figsize=(18, 10), sharex=True)
        fig.suptitle(f"Footlock visualization — {Path(npz_path).stem}", fontsize=14)

        # --- Left foot ---
        ax = axes[0]
        ax.set_title("Left Foot — Hard Footlock vs Foot Height")
        ax.fill_between(frames, 0, 1, where=hard_left, alpha=0.3, color="tab:blue",
                        label="footlock ON", transform=ax.get_xaxis_transform())
        ax.plot(frames, left_foot_z, color="tab:blue", lw=0.8, label="foot joint z")
        ax.plot(frames, left_ankle_z, color="tab:cyan", lw=0.8, alpha=0.6, label="ankle joint z")
        ax.set_ylabel("z (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.set_title("Right Foot — Hard Footlock vs Foot Height")
        ax.fill_between(frames, 0, 1, where=hard_right, alpha=0.3, color="tab:red",
                        label="footlock ON", transform=ax.get_xaxis_transform())
        ax.plot(frames, right_foot_z, color="tab:red", lw=0.8, label="foot joint z")
        ax.plot(frames, right_ankle_z, color="tab:orange", lw=0.8, alpha=0.6, label="ankle joint z")
        ax.set_ylabel("z (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Soft footlocks ---
        ax = axes[2]
        ax.set_title("Soft Footlock Values")
        ax.plot(frames, soft_left, color="tab:blue", lw=0.8, label="left soft")
        ax.plot(frames, soft_right, color="tab:red", lw=0.8, label="right soft")
        ax.set_ylabel("soft value")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Combined gait pattern ---
        ax = axes[3]
        ax.set_title("Contact Schedule (gait pattern)")
        ax.fill_between(frames, 0.5, 1.0, where=hard_left, alpha=0.5, color="tab:blue", label="left contact")
        ax.fill_between(frames, 0.0, 0.5, where=hard_right, alpha=0.5, color="tab:red", label="right contact")
        ax.set_yticks([0.25, 0.75])
        ax.set_yticklabels(["Right", "Left"])
        ax.set_xlabel("Frame")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize footlock contact schedules")
    parser.add_argument("--npz_path", help="Path to raw motion NPZ file")
    parser.add_argument("--save", default=None, help="Save figure to path instead of showing")
    parser.add_argument("--gait_only", action="store_true", help="Show only the gait pattern plot")
    args = parser.parse_args()
    plot_footlocks(args.npz_path, args.save, gait_only=args.gait_only)

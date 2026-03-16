import numpy as np
import matplotlib.pyplot as plt
import sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/motions/prepared/beam_sequence_stripped_rt2g1_rdy.csv"

data = np.loadtxt(csv_path, delimiter=",")
joints = data[:, 7:36]  # (frames, 29 DOFs)

JOINT_NAMES = [
    "L hip pitch",  "L hip roll",   "L hip yaw",    # 0-2   left leg
    "L knee",       "L ankle pitch","L ankle roll",  # 3-5
    "R hip pitch",  "R hip roll",   "R hip yaw",    # 6-8   right leg
    "R knee",       "R ankle pitch","R ankle roll",  # 9-11
    "waist yaw",    "waist roll",   "waist pitch",  # 12-14 waist
    "L shldr pitch","L shldr roll", "L shldr yaw",  # 15-17 left arm
    "L elbow",      "L wrist roll", "L wrist pitch","L wrist yaw",  # 18-21
    "R shldr pitch","R shldr roll", "R shldr yaw",  # 22-24 right arm
    "R elbow",      "R wrist roll", "R wrist pitch","R wrist yaw",  # 25-28
]

# Body region spans: (start, end_exclusive, label)
REGIONS = [
    (0, 6,   "Left leg"),
    (6, 12,  "Right leg"),
    (12, 15, "Waist"),
    (15, 22, "Left arm"),
    (22, 29, "Right arm"),
]

# Per-DOF: subtract mean, scale to [-1, 1] by peak absolute deviation
mean = joints.mean(axis=0, keepdims=True)
diff = joints - mean
scale = np.abs(diff).max(axis=0, keepdims=True)
scale[scale < 1e-6] = 1.0  # avoid div-by-zero for constant DOFs
normed = diff / scale

fig, ax = plt.subplots(figsize=(16, 7))
activation = np.abs(normed)
gamma = 2.0  # >1 suppresses low values, emphasises high values
activation = activation ** gamma
im = ax.imshow(activation.T, aspect="auto", cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
ax.set_xlabel("Frame")
ax.set_title(f"{csv_path.split('/')[-1]}  (mean-subtracted, range-normalised)")

# Y-axis: joint names
ax.set_yticks(range(len(JOINT_NAMES)))
ax.set_yticklabels(JOINT_NAMES, fontsize=7)

# Region brackets on the right side
for start, end, label in REGIONS:
    mid = (start + end - 1) / 2.0
    ax.annotate(label, xy=(1.01, mid), xycoords=("axes fraction", "data"),
                fontsize=8, fontweight="bold", va="center",
                color="0.2")
    # horizontal line between regions
    if start > 0:
        ax.axhline(start - 0.5, color="k", linewidth=0.8, linestyle="--", alpha=0.5)

fig.colorbar(im, ax=ax, label="Activation (|deviation| from mean)", pad=0.12)
plt.tight_layout()
plt.savefig("joint_heatmap.png", dpi=150)
plt.show()
print(f"Saved joint_heatmap.png  —  shape {joints.shape}")

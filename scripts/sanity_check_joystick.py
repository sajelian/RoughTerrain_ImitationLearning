"""Offline sanity check for the motion-derived joystick command.

Loads a motion npz, computes (vx, vy, omega_yaw) by averaging the anchor body's
world-frame linear/angular velocity over a forward horizon and projecting into
the anchor's instantaneous body frame, then writes a plot.

Run without launching IsaacLab. Useful to confirm the derived joystick matches
the visible motion before wiring it into the env.

Example:
    python scripts/sanity_check_joystick.py \\
        --motion_file artifacts/beam_along:v1/motion.npz \\
        --anchor_body torso_link \\
        --horizon_s 0.5
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply inverse rotation by unit quaternion (wxyz) to vector v.

    Vectorized: q shape (..., 4), v shape (..., 3). Returns (..., 3).
    Matches isaaclab.utils.math.quat_rotate_inverse semantics.
    """
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * np.cross(xyz, v)
    return v - w * t + np.cross(xyz, t)


def yaw_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Extract yaw-only quaternion (wxyz) from full quat (wxyz). Vectorized."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    out = np.zeros_like(q)
    out[..., 0] = np.cos(0.5 * yaw)
    out[..., 3] = np.sin(0.5 * yaw)
    return out


def compute_joystick(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    fps: float,
    horizon_steps: int,
) -> np.ndarray:
    """Return (T, 3) joystick (vx_h, vy_h, omega_yaw_h) using forward position
    displacement over horizon_steps, in the instantaneous yaw-only heading frame.

    Position displacement cancels gait-cycle oscillation and represents user intent:
    the command describes where the motion will go in the next horizon_s seconds.
    """
    T = body_pos_w.shape[0]
    H = max(1, horizon_steps)
    control_dt = 1.0 / fps

    future_t = np.minimum(np.arange(T) + H, T - 1)
    actual_dt = np.maximum((future_t - np.arange(T)) * control_dt, 1e-6)  # (T,)

    # Average velocity from position displacement
    avg_vel_w = (body_pos_w[future_t] - body_pos_w) / actual_dt[:, None]  # (T, 3)

    # Yaw rate from heading change
    def _yaw(q):
        return np.arctan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    yaw_now = _yaw(body_quat_w)
    yaw_fut = _yaw(body_quat_w[future_t])
    d_yaw = (yaw_fut - yaw_now + np.pi) % (2 * np.pi) - np.pi
    omega_z = d_yaw / actual_dt  # (T,)

    heading_quat = yaw_quat_wxyz(body_quat_w)
    lin_h = quat_rotate_inverse_wxyz(heading_quat, avg_vel_w)
    return np.stack([lin_h[:, 0], lin_h[:, 1], omega_z], axis=-1)


def compute_instantaneous(
    body_quat_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
    body_ang_vel_w: np.ndarray,
) -> np.ndarray:
    """Instantaneous lin/ang vel in the yaw-only heading frame (for comparison)."""
    heading_quat = yaw_quat_wxyz(body_quat_w)
    lin_h = quat_rotate_inverse_wxyz(heading_quat, body_lin_vel_w)
    ang_h = quat_rotate_inverse_wxyz(heading_quat, body_ang_vel_w)
    return np.stack([lin_h[:, 0], lin_h[:, 1], ang_h[:, 2]], axis=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion_file", required=True, help="Path to motion.npz")
    p.add_argument("--anchor_body", default="torso_link",
                   help="Anchor body name; used if motion file has 'body_names' metadata")
    p.add_argument("--anchor_index", type=int, default=None,
                   help="Fallback if motion file lacks 'body_names'; raw body index in motion data")
    p.add_argument("--horizon_s", type=float, default=1.5,
                   help="Centered averaging window in seconds")
    p.add_argument("--out", default=None, help="Output plot path (default: alongside motion file)")
    p.add_argument("--no_plot", action="store_true", help="Skip plot, just print stats")
    args = p.parse_args()

    data = np.load(args.motion_file, allow_pickle=True)
    fps = float(np.asarray(data["fps"]).item())
    if "body_names" in data.files:
        body_names = list(data["body_names"])
        if args.anchor_body not in body_names:
            raise SystemExit(f"anchor_body '{args.anchor_body}' not in motion: {body_names}")
        anchor_idx = body_names.index(args.anchor_body)
    else:
        if args.anchor_index is None:
            raise SystemExit(
                f"motion file '{args.motion_file}' has no 'body_names' metadata; "
                "pass --anchor_index <int> to specify the raw body index."
            )
        anchor_idx = args.anchor_index

    body_pos_w = data["body_pos_w"][:, anchor_idx].astype(np.float32)
    body_quat_w = data["body_quat_w"][:, anchor_idx].astype(np.float32)
    body_lin_vel_w = data["body_lin_vel_w"][:, anchor_idx].astype(np.float32)
    body_ang_vel_w = data["body_ang_vel_w"][:, anchor_idx].astype(np.float32)
    T = body_quat_w.shape[0]
    K = max(1, int(args.horizon_s * fps))

    forward_speed = 0.8
    turn_slow = 0.785   # pi/4
    turn_fast = 1.571   # pi/2
    speed_ema_tau_s = 1.0

    joystick = compute_joystick(body_pos_w, body_quat_w, fps, K)
    instantaneous = compute_instantaneous(body_quat_w, body_lin_vel_w, body_ang_vel_w)

    # Causal EMA on planar speed — matches online env behaviour
    raw_speed = np.sqrt(joystick[:, 0] ** 2 + joystick[:, 1] ** 2)
    control_dt = 1.0 / fps
    alpha = float(np.exp(-control_dt / speed_ema_tau_s))
    speed_ema = np.zeros_like(raw_speed)
    speed_ema[0] = raw_speed[0]
    for i in range(1, T):
        speed_ema[i] = alpha * speed_ema[i - 1] + (1.0 - alpha) * raw_speed[i]

    def quantize(speed_ema_val, wz):
        vx_q = forward_speed if speed_ema_val >= 0.1 else 0.0
        slow_thresh = turn_slow * 0.5
        fast_thresh = (turn_slow + turn_fast) * 0.5
        if wz > fast_thresh:    wz_q = turn_fast
        elif wz > slow_thresh:  wz_q = turn_slow
        elif wz < -fast_thresh: wz_q = -turn_fast
        elif wz < -slow_thresh: wz_q = -turn_slow
        else:                   wz_q = 0.0
        return vx_q, wz_q

    duration = T / fps
    print(f"[motion] {args.motion_file}")
    print(f"  fps={fps}  frames={T}  duration={duration:.2f}s")
    print(f"  anchor_body={args.anchor_body}  horizon_s={args.horizon_s}  K={K} frames")
    print(f"  forward_speed={forward_speed}  turn_slow={turn_slow:.3f}  turn_fast={turn_fast:.3f}")
    print()
    print(f"[joystick stats over clip]")
    for i, lab in enumerate(["vx_b (m/s)", "vy_b (m/s)", "omega_yaw (rad/s)"]):
        col = joystick[:, i]
        print(f"  {lab:>20s}  min={col.min():+.3f}  max={col.max():+.3f}  mean={col.mean():+.3f}  std={col.std():.3f}")

    # Label distribution
    labels = [quantize(speed_ema[t], joystick[t, 2]) for t in range(T)]
    vx_labels = [l[0] for l in labels]
    wz_labels = [l[1] for l in labels]
    print()
    print(f"[label distribution over {T} frames]")
    for v in sorted(set(vx_labels)):
        pct = 100 * vx_labels.count(v) / T
        print(f"  vx={v:+.2f}  {pct:5.1f}%  ({'W' if v > 0 else 'stop'})")
    for w in sorted(set(wz_labels)):
        n = wz_labels.count(w)
        pct = 100 * n / T
        tag = {turn_fast: "fast-left", turn_slow: "slow-left", 0.0: "straight",
               -turn_slow: "slow-right", -turn_fast: "fast-right"}.get(w, "?")
        print(f"  wz={w:+.3f}  {pct:5.1f}%  ({tag})")

    print()
    print(f"[sample at 10 evenly-spaced frames]")
    print(f"  {'t(s)':>6s}  {'vx':>7s}  {'wz':>7s}  {'label_vx':>9s}  {'label_wz':>9s}   |  {'inst_vx':>8s}  {'inst_wz':>8s}")
    sample_ts = np.linspace(0, T - 1, 10, dtype=int)
    for t in sample_ts:
        j = joystick[t]
        inst = instantaneous[t]
        lv, lw = quantize(speed_ema[t], j[2])
        print(
            f"  {t/fps:6.2f}  {j[0]:+7.3f}  {j[2]:+7.3f}  {lv:+9.2f}  {lw:+9.3f}   |  "
            f"{inst[0]:+8.3f}  {inst[2]:+8.3f}"
        )

    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping plot. Use --no_plot to suppress this.")
        return

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.motion_file))[0]
        parent = os.path.basename(os.path.dirname(args.motion_file)) or "motion"
        out_path = f"joystick_sanity_{parent}_{base}.png"

    t_axis = np.arange(T) / fps
    label_vx = np.array([quantize(speed_ema[t], joystick[t, 2])[0] for t in range(T)])
    label_wz = np.array([quantize(speed_ema[t], joystick[t, 2])[1] for t in range(T)])

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # vx panel
    axes[0].plot(t_axis, instantaneous[:, 0], alpha=0.35, color="gray", label="instantaneous")
    axes[0].plot(t_axis, joystick[:, 0], color="C0", label=f"displacement avg ({args.horizon_s}s)")
    axes[0].step(t_axis, label_vx, color="C1", linewidth=2, where="mid", label="label (quantised)")
    axes[0].axhline(0.1, color="C1", linewidth=0.8, linestyle="--", alpha=0.6, label="W threshold (0.1)")
    axes[0].set_ylabel("vx_h (m/s)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # vy panel (continuous only — no label for vy)
    axes[1].plot(t_axis, instantaneous[:, 1], alpha=0.35, color="gray", label="instantaneous")
    axes[1].plot(t_axis, joystick[:, 1], color="C0", label="displacement avg")
    axes[1].set_ylabel("vy_h (m/s)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # wz panel
    axes[2].plot(t_axis, instantaneous[:, 2], alpha=0.35, color="gray", label="instantaneous")
    axes[2].plot(t_axis, joystick[:, 2], color="C0", label=f"displacement avg ({args.horizon_s}s)")
    axes[2].step(t_axis, label_wz, color="C1", linewidth=2, where="mid", label="label (quantised)")
    for lvl, name in [(turn_slow, f"±slow ({turn_slow:.2f})"), (turn_fast, f"±fast ({turn_fast:.2f})")]:
        axes[2].axhline( lvl, color="C1", linewidth=0.8, linestyle="--", alpha=0.5)
        axes[2].axhline(-lvl, color="C1", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[2].set_ylabel("omega_yaw (rad/s)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[-1].set_xlabel(f"time (s, fps={fps})")
    plt.suptitle(f"Joystick labels: {args.motion_file}  (anchor={args.anchor_body})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    print(f"\n[plot] saved to: {out_path}")


if __name__ == "__main__":
    main()

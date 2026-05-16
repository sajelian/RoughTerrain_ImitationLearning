"""This script replay a motion from a csv file and output it to a npz  and also calculates velocities.

.. code-block:: bash

    # Usage
    python scripts/csv_to_npz_terrain.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from pathlib import Path
import numpy as np

from isaaclab.app import AppLauncher
from usd_utils import load_usd_config  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion csv file.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument(
    "--raw_npz",
    type=str,
    default=None,
    help="Path to raw motion NPZ containing egomc_footlock_LeftFoot / egomc_footlock_RightFoot.",
)
parser.add_argument(
    "--raw_npz_frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "Frame range into the raw NPZ for footlock extraction (1-indexed, inclusive). "
        "Use this when --frame_range applies to the CSV but the raw NPZ needs a different slice."
    ),
)

parser.add_argument(
    "--usd_name", type=str, default=None, help="Name of the USD mesh."
)
parser.add_argument(
    "--no_foot_align", action="store_true",
    help="Skip foot-to-terrain vertical alignment (no root z shifting).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

usd_path, usd_cfg = load_usd_config(args_cli.usd_name)

"""Rest everything follows."""


def _resample_footlocks(raw_npz_path: str, input_fps: int, output_fps: int, output_frames: int, frame_range=None):
    """Resample binary footlock masks from raw NPZ (at input_fps) to output_fps.

    Uses nearest-neighbor interpolation so the result stays binary.
    Returns (foot_contact_left, foot_contact_right, foot_contact_mask) as numpy bool arrays
    of shape (output_frames,).  foot_contact_mask is False for frames where both raw
    footlocks are simultaneously off for an implausibly long run (>0.5s), indicating
    missing/invalid data.
    """
    raw = np.load(raw_npz_path, allow_pickle=True)
    left = raw["egomc_footlock_LeftFoot"].astype(bool)
    right = raw["egomc_footlock_RightFoot"].astype(bool)

    if frame_range is not None:
        start = frame_range[0] - 1
        end = frame_range[1]
        left = left[start:end]
        right = right[start:end]

    raw_frames = len(left)
    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    duration = (raw_frames - 1) * input_dt

    out_times = np.arange(0, duration, output_dt)[:output_frames]
    raw_indices = np.round(out_times / input_dt).astype(int)
    raw_indices = np.clip(raw_indices, 0, raw_frames - 1)

    left_out = left[raw_indices]
    right_out = right[raw_indices]

    both_off = ~left_out & ~right_out
    min_run = max(1, int(0.5 * output_fps))
    mask = np.ones(output_frames, dtype=bool)
    run_start = None
    for i in range(output_frames):
        if both_off[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= min_run:
                mask[run_start:i] = False
            run_start = None
    if run_start is not None and (output_frames - run_start) >= min_run:
        mask[run_start:output_frames] = False

    valid_count = int(mask.sum())
    print(
        f"[FOOTLOCK] Resampled {raw_frames} -> {output_frames} frames. "
        f"Valid: {valid_count}/{output_frames} ({100 * valid_count / output_frames:.1f}%)"
    )
    return left_out, right_out, mask

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from RT_IL.robots.g1 import G1_VIZ_CFG, G1_FOOT_LINKS, G1_HAND_LINKS, G1_DOF_JOINT_NAMES
from motion_preprocessor import MotionLoader  # isort: skip


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(), init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -3.0)))

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation (viz robot — no collision needed, detection is geometric)
    robot: ArticulationCfg = G1_VIZ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    obstacle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/obstacle",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            scale=(1.0, 1.0, 1.0),
            visual_material=sim_utils.materials.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.4, 0.4),
                emissive_color=(0.0, 0.0, 0.0),
                roughness=0.5,
                metallic=0.5,
                opacity=1.0
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        collision_group=0,
    )


def _extract_obstacle_mesh(obstacle_prim_path, device):
    """Extract mesh vertices/faces from a USD prim already loaded in Isaac Sim.

    Returns (warp_mesh, vertices, faces) — warp mesh for raycasting, raw arrays for proximity.
    """
    from pxr import UsdGeom

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    mesh_prim = sim_utils.get_first_matching_child_prim(
        obstacle_prim_path, lambda p: p.GetTypeName() == "Mesh", stage
    )
    if mesh_prim is None:
        raise RuntimeError(f"No Mesh prim found under {obstacle_prim_path}")

    mesh_prim = UsdGeom.Mesh(mesh_prim)
    points = np.asarray(mesh_prim.GetPointsAttr().Get()).astype(np.float64)
    # Apply world transform so vertices are in simulation world frame
    xform = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
    points = points @ xform[:3, :3].T + xform[:3, 3]
    indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())

    from isaaclab.utils.warp import convert_to_warp_mesh

    wp_mesh = convert_to_warp_mesh(points, indices, device=device)
    faces = indices.reshape(-1, 3)
    print(f"[ALIGN] Obstacle mesh: {len(points)} vertices, {len(faces)} faces")
    return wp_mesh, points, faces


def _visualize_alignment(log, shift, rsi_mask, fps, save_path="/tmp/motion_alignment.png"):
    """Save a visualization of the alignment results."""
    import matplotlib.pyplot as plt

    T = log["body_pos_w"].shape[0]
    times = np.arange(T) / fps
    root_pos = log["body_pos_w"][:, 0, :]  # (T, 3) — pelvis/root

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})

    # --- Top: trajectory top-down view, colored by RSI validity ---
    ax = axes[0]
    valid = rsi_mask
    ax.scatter(
        root_pos[valid, 0], root_pos[valid, 1],
        c="tab:green", s=4, label="valid RSI", zorder=2,
    )
    ax.scatter(
        root_pos[~valid, 0], root_pos[~valid, 1],
        c="tab:red", s=12, marker="x", label="hand contact (RSI disabled)", zorder=3,
    )
    ax.plot(root_pos[:, 0], root_pos[:, 1], c="grey", lw=0.3, zorder=1)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Root trajectory — RSI validity")
    ax.legend(loc="best", markerscale=2)
    ax.set_aspect("equal")

    # --- Bottom: vertical shift over time ---
    ax2 = axes[1]
    colors = np.where(shift > 0, "tab:blue", np.where(shift < 0, "tab:orange", "tab:grey"))
    ax2.bar(times, shift, width=1.0 / fps, color=colors, edgecolor="none")
    # Mark hand-clipping frames
    clip_times = times[~rsi_mask]
    if len(clip_times) > 0:
        ax2.scatter(
            clip_times, np.zeros_like(clip_times),
            c="tab:red", s=15, marker="x", zorder=3, label="hand contact",
        )
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Vertical shift (m)")
    ax2.set_title("Foot alignment shift (blue=down/floating, orange=up/clipping)")
    if len(clip_times) > 0:
        ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[ALIGN] Visualization saved to {save_path}")


def align_motion_to_terrain(log, wp_mesh, mesh_vertices, device):
    """Align motion to terrain mesh: fix foot height.

    Vertically shifts the entire pose so the lowest foot sits on the
    mesh surface. This fixes both floating (shift down) and foot-clipping
    (shift up) in one pass.

    Note: Hand clipping detection is handled separately via physics contact
    sensor — rsi_mask should already be set in log before calling this.
    """
    from isaaclab.utils.warp import raycast_mesh

    body_names = list(log["body_names"])
    body_pos_w = log["body_pos_w"]  # (T, B, 3)
    body_lin_vel_w = log["body_lin_vel_w"]  # (T, B, 3)
    T = body_pos_w.shape[0]
    fps = log["fps"][0] if isinstance(log["fps"], (list, np.ndarray)) else log["fps"]

    foot_names = G1_FOOT_LINKS
    foot_idx = [body_names.index(n) for n in foot_names]

    # Ankle-to-sole offset: collision cylinders at z=-0.025 with radius 0.01 → sole is 0.035m below ankle frame
    ANKLE_TO_SOLE = 0.05  # 0.035 <- this lead to penetrations

    print(f"[ALIGN] Raycasting for foot heights ({T * 2} rays)...")
    foot_pos = body_pos_w[:, foot_idx, :]  # (T, 2, 3)
    ray_starts = torch.tensor(foot_pos.reshape(-1, 3), dtype=torch.float32, device=device)
    ray_starts[:, 2] = 100.0  # cast from high above
    ray_dirs = torch.zeros_like(ray_starts)
    ray_dirs[:, 2] = -1.0

    ray_hits = raycast_mesh(ray_starts, ray_dirs, mesh=wp_mesh)[0]  # (T*2, 3)
    ground_z = ray_hits[:, 2].cpu().numpy().reshape(T, 2)
    ground_z[ground_z > 1e5] = np.nan  # inf = no hit

    # Sole height = ankle_z - ANKLE_TO_SOLE; gap = sole_z - ground_z
    sole_z = foot_pos[:, :, 2] - ANKLE_TO_SOLE
    foot_gap = sole_z - ground_z  # positive = above, negative = below (clipping)

    # Only correct frames where both feet have valid mesh hits
    both_have_mesh = ~np.isnan(ground_z).any(axis=1)  # (T,)
    # Shift = gap of the lowest foot → places lowest foot exactly on surface
    ALIGN_THRESH = 0.005  # 5mm dead-zone to avoid jitter on already-aligned frames
    min_gap = np.nanmin(foot_gap, axis=1)
    needs_correction = both_have_mesh & (np.abs(min_gap) > ALIGN_THRESH)
    shift = np.where(needs_correction, min_gap, 0.0)

    num_corrected = int(needs_correction.sum())
    num_shifted_down = int((shift > 0).sum())
    num_shifted_up = int((shift < 0).sum())
    print(
        f"[ALIGN] Foot alignment (raw): {num_corrected}/{T} frames corrected"
        f" ({num_shifted_down} down, {num_shifted_up} up)"
    )

    # Rate-limit shift so it doesn't change faster than the root's vertical velocity.
    # This prevents the robot from being yanked down when stepping off a box.
    root_vz = body_lin_vel_w[:, 0, 2]  # root vertical velocity (m/s)
    dt = 1.0 / fps
    raw_shift = shift.copy()
    for t in range(1, T):
        delta = shift[t] - shift[t - 1]
        # Max allowed change = how far the root naturally moves vertically in one frame
        # plus a small base rate to allow convergence when root is stationary
        BASE_RATE = 0.002  # 2mm/frame baseline
        max_delta = abs(root_vz[t]) * dt + BASE_RATE
        shift[t] = shift[t - 1] + np.clip(delta, -max_delta, max_delta)

    num_rate_limited = int((np.abs(shift - raw_shift) > 1e-4).sum())
    print(f"[ALIGN] Rate-limited {num_rate_limited}/{T} frames")

    if num_corrected > 0:
        body_pos_w[:, :, 2] -= shift[:, None]
        shift_vel = np.gradient(shift, dt)
        body_lin_vel_w[:, :, 2] -= shift_vel[:, None]

    log["body_pos_w"] = body_pos_w
    log["body_lin_vel_w"] = body_lin_vel_w
    log["_foot_shift"] = shift  # store for visualization later

    return log


def _run_interactive_offset(sim: SimulationContext, scene: InteractiveScene, robot, log: dict) -> np.ndarray:
    """Keyboard-driven root offset phase between Pass 1 and foot alignment.

    Replays the recorded motion with a live spatial offset so the user can
    visually assess motion/terrain alignment before committing.

    Keys (conflict-free with Isaac Sim viewport):
        ← / →      ±X          ↑ / ↓      ±Y          PgUp / PgDn   ±Z
        Numpad + / -   step ×2 / ÷2
        Enter      accept      Backspace   reset offset to zero
    """
    if getattr(args_cli, "headless", False):
        return np.zeros(3)

    import carb.input as carb_input
    import omni.appwindow

    state = {
        "offset": np.zeros(3),
        "step": 0.05,
        "dirty": True,
        "done": False,
        "reset": False,
    }

    input_iface = carb_input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    KEY = carb_input.KeyboardInput
    EV = carb_input.KeyboardEventType

    def _on_key(event, *_):
        t = event.type
        if t not in (EV.KEY_PRESS, EV.KEY_REPEAT):
            return True
        k = event.input
        s = state["step"]
        if   k == KEY.RIGHT:         state["offset"][0] += s
        elif k == KEY.LEFT:          state["offset"][0] -= s
        elif k == KEY.UP:            state["offset"][1] += s
        elif k == KEY.DOWN:          state["offset"][1] -= s
        elif k == KEY.PAGE_UP:       state["offset"][2] += s
        elif k == KEY.PAGE_DOWN:     state["offset"][2] -= s
        elif k == KEY.NUMPAD_ADD:    state["step"] = min(state["step"] * 2.0, 1.0)
        elif k == KEY.NUMPAD_SUB:    state["step"] = max(state["step"] / 2.0, 0.001)
        elif k == KEY.ENTER:         state["done"] = True
        elif k == KEY.BACKSPACE:     state["reset"] = True
        state["dirty"] = True
        return True

    sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_key)

    T_total = log["body_pos_w"].shape[0]
    replay_idx = 0
    dt = sim.get_physics_dt()

    print(
        "\n[OFFSET] Interactive offset — ←/→=±X  ↑/↓=±Y  PgUp/PgDn=±Z  Num+/-=step"
        "  Enter=accept  Backspace=reset"
    )
    print(f"[OFFSET] offset=(+0.000, +0.000, +0.000)m  step=0.050m", end="", flush=True)

    while simulation_app.is_running() and not state["done"]:
        if state["reset"]:
            state["offset"][:] = 0.0
            state["reset"] = False
            state["dirty"] = True

        if state["dirty"]:
            o = state["offset"]
            print(
                f"\r[OFFSET] offset=({o[0]:+.3f}, {o[1]:+.3f}, {o[2]:+.3f})m"
                f"  step={state['step']:.3f}m  [Enter=accept  Esc=reset]",
                end="",
                flush=True,
            )
            state["dirty"] = False

        disp_pos = log["body_pos_w"][replay_idx, 0] + state["offset"]
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = torch.tensor(disp_pos, dtype=torch.float32, device=sim.device)
        root_states[:, 3:7] = torch.tensor(log["body_quat_w"][replay_idx, 0], dtype=torch.float32, device=sim.device)
        root_states[:, 7:10] = torch.tensor(log["body_lin_vel_w"][replay_idx, 0], dtype=torch.float32, device=sim.device)
        root_states[:, 10:] = torch.tensor(log["body_ang_vel_w"][replay_idx, 0], dtype=torch.float32, device=sim.device)
        robot.write_root_state_to_sim(root_states)

        jp = torch.tensor(log["joint_pos"][replay_idx], dtype=torch.float32, device=sim.device).unsqueeze(0)
        jv = torch.tensor(log["joint_vel"][replay_idx], dtype=torch.float32, device=sim.device).unsqueeze(0)
        robot.write_joint_state_to_sim(jp, jv)

        sim.render()
        scene.update(dt)

        replay_idx = (replay_idx + 1) % T_total

    input_iface.unsubscribe_to_keyboard_events(keyboard, sub)
    print()

    accepted = state["offset"].copy()
    print(f"[OFFSET] Accepted: ({accepted[0]:+.3f}, {accepted[1]:+.3f}, {accepted[2]:+.3f})m")
    return accepted


def _save_output_csv(log: dict, robot_joint_indexes: list, env_origin: np.ndarray, output_name: str) -> Path:
    """Write a CSV from the processed log in the same column layout as the input CSV.

    Columns: root_pos(3)  root_rot_xyzw(4)  dof_pos(N)
    Root position is in motion-local frame (world-frame minus env_origin).
    """
    root_pos = log["body_pos_w"][:, 0, :] - env_origin  # (T, 3)
    root_quat_wxyz = log["body_quat_w"][:, 0, :]         # (T, 4) wxyz
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]     # → xyzw for CSV
    dof_pos = log["joint_pos"][:, robot_joint_indexes]    # (T, N_dof)

    data = np.concatenate([root_pos, root_quat_xyzw, dof_pos], axis=1)

    csv_path = Path(args_cli.input_file).resolve().parent / f"{output_name}.csv"
    np.savetxt(csv_path, data, delimiter=",", fmt="%.6f")
    print(f"[CSV] Saved {data.shape[0]} frames → {csv_path}")
    return csv_path


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    T = motion.output_frames
    dt = sim.get_physics_dt()

    # =========================================================================
    # Kinematic replay — record body data
    # =========================================================================
    print(f"[INFO] Recording {T} frames...")
    log = {
        "fps": [args_cli.output_fps],
        "body_names": np.array(robot.body_names, dtype=object),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    for frame_i in range(T):
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            _,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()  # kinematic replay only — no physics needed
        scene.update(dt)

        log["joint_pos"].append(joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if frame_i % 50 == 0:
            print(f"  [PASS 1] frame {frame_i}/{T}", flush=True)

    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[k] = np.stack(log[k], axis=0)
    print(f"[PASS 1] Done. Recorded {T} frames.")

    # =========================================================================
    # Interactive root offset — keyboard XY/Z adjustment before foot alignment
    # =========================================================================
    env_origin = scene.env_origins[0].cpu().numpy()
    offset = _run_interactive_offset(sim, scene, robot, log)
    if np.any(offset != 0.0):
        log["body_pos_w"] += offset[None, None, :]
        print(f"[OFFSET] Baked ({offset[0]:+.3f}, {offset[1]:+.3f}, {offset[2]:+.3f})m into {T} frames.")

    # =========================================================================
    # Foot vertical alignment (raycasting) — fix root z before contact resolve
    # =========================================================================
    obstacle_prim_path = "/World/envs/env_0/obstacle"
    wp_mesh, mesh_verts, _ = _extract_obstacle_mesh(obstacle_prim_path, sim.device)
    if args_cli.no_foot_align:
        print("[INFO] Skipping foot-to-terrain alignment (--no_foot_align).")
    else:
        log = align_motion_to_terrain(log, wp_mesh, mesh_verts, sim.device)

    # =========================================================================
    # Hand clipping detection via KDTree proximity to obstacle mesh
    # =========================================================================
    from scipy.spatial import KDTree

    HAND_CLIP_DIST = 0.03  # 3cm — if hand link is closer than this to mesh, mark as clipping
    hand_names = G1_HAND_LINKS
    body_names = list(log["body_names"])
    hand_idx = [body_names.index(n) for n in hand_names]

    mesh_tree = KDTree(mesh_verts)
    hand_pos_all = log["body_pos_w"][:, hand_idx, :]  # (T, 2, 3)

    contact_arr = np.zeros(T, dtype=bool)
    for fi in range(T):
        for hi in range(len(hand_idx)):
            dist, _ = mesh_tree.query(hand_pos_all[fi, hi])
            if dist < HAND_CLIP_DIST:
                contact_arr[fi] = True
                break

    num_clipping = int(contact_arr.sum())
    print(f"[RSI] Hand clipping frames (KDTree, <{HAND_CLIP_DIST*100:.0f}cm): {num_clipping}/{T}")

    from scipy.ndimage import gaussian_filter1d
    RSI_SIGMA = 10  # frames
    RSI_THRESHOLD = 0.1
    invalid_smooth = gaussian_filter1d(contact_arr.astype(float), sigma=RSI_SIGMA)
    rsi_mask = invalid_smooth < RSI_THRESHOLD
    num_invalid = int((~rsi_mask).sum())
    print(f"[RSI] After Gaussian dilation (σ={RSI_SIGMA}, thresh={RSI_THRESHOLD}): "
          f"{num_invalid}/{T} frames invalid")
    print(f"[RSI] Valid RSI frames: {rsi_mask.sum()}/{T} ({100 * rsi_mask.mean():.1f}%)")
    log["rsi_mask"] = rsi_mask

    # Use foot shift from alignment for visualization
    fps = args_cli.output_fps
    foot_shift = log.pop("_foot_shift", np.zeros(T))
    _visualize_alignment(log, foot_shift, rsi_mask, fps)

    # Resample footlocks from raw NPZ if provided
    if args_cli.raw_npz is not None:
        output_frames = log["joint_pos"].shape[0]
        fl_left, fl_right, fl_mask = _resample_footlocks(
            args_cli.raw_npz, args_cli.input_fps, args_cli.output_fps,
            output_frames, frame_range=args_cli.raw_npz_frame_range,
        )
        log["foot_contact_left"] = fl_left
        log["foot_contact_right"] = fl_right
        log["foot_contact_mask"] = fl_mask

    # Save and upload
    np.savez("/tmp/motion.npz", **log)

    import wandb

    COLLECTION = args_cli.output_name
    run = wandb.init(project="csv_to_npz", name=COLLECTION)
    print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
    REGISTRY = "motions"
    logged_artifact = run.log_artifact(artifact_or_path="/tmp/motion.npz", name=COLLECTION, type=REGISTRY)
    run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
    print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")
    _save_output_csv(log, robot_joint_indexes, env_origin, args_cli.output_name)

    # =========================================================================
    # Replay the resolved motion with green/red RSI visualization
    # =========================================================================
    print("[INFO]: Replaying resolved motion — green = valid RSI, red = invalid RSI.")
    print("[INFO]: Press Ctrl+C to exit.")

    rsi_mask = log["rsi_mask"]
    T_total = log["body_pos_w"].shape[0]
    replay_idx = 0

    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/RSIMarker",
        markers={
            "valid": sim_utils.SphereCfg(
                radius=0.12,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2)),
            ),
            "invalid": sim_utils.SphereCfg(
                radius=0.12,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
            ),
        },
    )
    rsi_marker = VisualizationMarkers(marker_cfg)

    while simulation_app.is_running():
        is_valid = bool(rsi_mask[replay_idx])

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = torch.tensor(log["body_pos_w"][replay_idx, 0], device=sim.device)
        root_states[:, 3:7] = torch.tensor(log["body_quat_w"][replay_idx, 0], device=sim.device)
        root_states[:, 7:10] = torch.tensor(log["body_lin_vel_w"][replay_idx, 0], device=sim.device)
        root_states[:, 10:] = torch.tensor(log["body_ang_vel_w"][replay_idx, 0], device=sim.device)
        robot.write_root_state_to_sim(root_states)

        jp = torch.tensor(log["joint_pos"][replay_idx], device=sim.device).unsqueeze(0)
        jv = torch.tensor(log["joint_vel"][replay_idx], device=sim.device).unsqueeze(0)
        robot.write_joint_state_to_sim(jp, jv)

        sim.render()
        scene.update(sim.get_physics_dt())

        marker_idx = torch.tensor([0 if is_valid else 1], device=sim.device)
        rsi_marker.visualize(
            translations=root_states[:, :3],
            marker_indices=marker_idx,
        )

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        replay_idx = (replay_idx + 1) % T_total


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    
    scene_cfg.obstacle = scene_cfg.obstacle.replace(
        spawn=scene_cfg.obstacle.spawn.replace(
            usd_path=usd_path,
            scale=tuple(usd_cfg.get("scale", (1.0, 1.0, 1.0))),
        ),
        init_state=scene_cfg.obstacle.init_state.replace(
            pos=tuple(usd_cfg.get("translation", (0.0, 0.0, 0.0))),
            rot=tuple(usd_cfg.get("rotation", (1.0, 0.0, 0.0, 0.0))),
        ),
    )

    print("[INFO]: Creating scene...", flush=True)
    scene = InteractiveScene(scene_cfg)
    print("[INFO]: Scene created. Resetting sim...", flush=True)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...", flush=True)
    # Run the simulator
    run_simulator(
        sim,
        scene,
        joint_names=G1_DOF_JOINT_NAMES,
    )


if __name__ == "__main__":
    try:
        # run the main function
        main()
    finally:
        # close sim app
        simulation_app.close()

"""This script replay a motion from a csv file and output it to a npz  and also calculates velocities.

.. code-block:: bash

    # Usage
    python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import numpy as np

from isaaclab.app import AppLauncher

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

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
        # frame_range is 1-indexed inclusive; raw data is 0-indexed
        start = frame_range[0] - 1
        end = frame_range[1]
        left = left[start:end]
        right = right[start:end]

    raw_frames = len(left)
    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    duration = (raw_frames - 1) * input_dt

    # Build output time grid and map to nearest raw frame (nearest-neighbor)
    out_times = np.arange(0, duration, output_dt)[:output_frames]
    raw_indices = np.round(out_times / input_dt).astype(int)
    raw_indices = np.clip(raw_indices, 0, raw_frames - 1)

    left_out = left[raw_indices]
    right_out = right[raw_indices]

    # Validity mask: mark frames where both feet are off for >0.5s as invalid
    both_off = ~left_out & ~right_out
    min_run = max(1, int(0.5 * output_fps))  # 0.5s worth of frames
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
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
from RT_IL.robots.g1 import G1_VIZ_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_VIZ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from a csv or npz file.

        CSV format (from GMR): columns are [root_pos(3), root_rot_xyzw(4), dof_pos(N)].
        NPZ format (from OmniRetarget): 'qpos' field with [root_pos(3), root_rot_wxyz(4), dof_pos(N)].
        """
        is_npz = self.motion_file.endswith(".npz")

        if is_npz:
            data = np.load(self.motion_file)
            motion_np = data["qpos"]
            if self.frame_range is not None:
                motion_np = motion_np[self.frame_range[0] - 1 : self.frame_range[1]]
            motion = torch.from_numpy(motion_np).to(torch.float32).to(self.device)
            self.motion_base_poss_input = motion[:, :3]
            self.motion_base_rots_input = motion[:, 3:7]  # already wxyz
            self.motion_dof_poss_input = motion[:, 7:]
        else:
            if self.frame_range is None:
                motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
            else:
                motion = torch.from_numpy(
                    np.loadtxt(
                        self.motion_file,
                        delimiter=",",
                        skiprows=self.frame_range[0] - 1,
                        max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                    )
                )
            motion = motion.to(torch.float32).to(self.device)
            self.motion_base_poss_input = motion[:, :3]
            self.motion_base_rots_input = motion[:, 3:7]
            self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert xyzw to wxyz
            self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = self.motion_base_poss_input.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


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

    # ------- data logger -------------------------------------------------------
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
    file_saved = False
    loop = False  # set to True to loop motion playback
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()  # type: ignore

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

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

            np.savez("/tmp/motion.npz", **log)

            import wandb

            COLLECTION = args_cli.output_name
            run = wandb.init(project="csv_to_npz", name=COLLECTION)
            print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
            REGISTRY = "motions"
            logged_artifact = run.log_artifact(artifact_or_path="/tmp/motion.npz", name=COLLECTION, type=REGISTRY)
            run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
            print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")

            # exit after saving
            if not loop:
                print("[INFO]: Motion playback complete, exiting...")
                # simulation_app.close()
                os._exit(0)  # hard exit to mirror Ctrl+C


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        sim,
        scene,
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    )


if __name__ == "__main__":
    try:
        # run the main function
        main()
    finally:
        # close sim app
        simulation_app.close()

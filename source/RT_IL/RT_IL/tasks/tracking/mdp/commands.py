from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        body_names: list[str],
        robot_body_indexes: torch.Tensor,
        device: str = "cpu",
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file, allow_pickle=True)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self.time_step_total = self.joint_pos.shape[0]

        if "rsi_mask" in data:
            self.rsi_mask = torch.tensor(data["rsi_mask"], dtype=torch.bool, device=device)
            valid_count = int(self.rsi_mask.sum())
            print(
                f"[RSI] Valid frames: {valid_count}/{self.time_step_total}"
                f" ({100 * valid_count / self.time_step_total:.1f}%)"
            )
        else:
            self.rsi_mask = None

        # Foot contact schedule (optional)
        if "foot_contact_left" in data and "foot_contact_right" in data:
            self.foot_contact_left = torch.tensor(data["foot_contact_left"], dtype=torch.bool, device=device)
            self.foot_contact_right = torch.tensor(data["foot_contact_right"], dtype=torch.bool, device=device)
            self.foot_contact_mask = torch.tensor(data["foot_contact_mask"], dtype=torch.bool, device=device)
            valid_count = int(self.foot_contact_mask.sum())
            print(
                f"[FOOTLOCK] Loaded contact schedule. "
                f"Valid: {valid_count}/{self.time_step_total} ({100 * valid_count / self.time_step_total:.1f}%)"
            )
        else:
            self.foot_contact_left = None
            self.foot_contact_right = None
            self.foot_contact_mask = None

        # Compute body indexes for motion data: use stored body_names if available,
        # otherwise fall back to robot body indexes (works when robot matches motion data).
        if "body_names" in data:
            all_body_names = list(data["body_names"])
            self._body_indexes = torch.tensor(
                [all_body_names.index(name) for name in body_names],
                dtype=torch.long,
                device=device,
            )
        else:
            max_idx = robot_body_indexes.max().item()
            num_motion_bodies = self._body_pos_w.shape[1]
            if max_idx >= num_motion_bodies:
                raise RuntimeError(
                    f"Motion file lacks 'body_names' metadata and robot body indices (max={max_idx}) "
                    f"exceed motion body count ({num_motion_bodies}). "
                    f"Regenerate the motion file with the updated csv_to_npz.py to include body_names."
                )
            self._body_indexes = robot_body_indexes

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(
            self.cfg.motion_file, self.cfg.body_names, self.body_indexes, device=self.device
        )

        # Auto-size the episode to the clip. Playback consumes one motion frame per env step,
        # so the clip lasts `time_step_total * step_dt` seconds (NOT frames/fps, which only
        # matches if the file is sampled at the control rate). If episode_length_s is shorter,
        # a from-the-start spawn times out before the final-10% goal window is ever reachable
        # — completion stays 0 and the curriculum silently stalls. The clip length is only
        # known after the motion loads, and motion_file is often injected at runtime after
        # cfg __post_init__, so this is the correct place to set it.
        if self.cfg.auto_episode_length:
            step_dt = env.cfg.decimation * env.cfg.sim.dt
            clip_length_s = self.motion.time_step_total * step_dt
            if abs(env.cfg.episode_length_s - clip_length_s) > 1e-6:
                print(
                    f"[MotionCommand] episode_length_s {env.cfg.episode_length_s:.2f}s -> "
                    f"{clip_length_s:.2f}s to match clip ({self.motion.time_step_total} frames)."
                )
            env.cfg.episode_length_s = clip_length_s

        if self.motion.rsi_mask is not None:
            self._valid_rsi_indices = torch.where(self.motion.rsi_mask)[0]
            assert len(self._valid_rsi_indices) > 0, "No valid RSI frames in motion data!"
        else:
            self._valid_rsi_indices = None
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        # --- robustness curriculum: motion-completion tracking ---
        # The goal window is the final `success_final_fraction` of the clip. For most
        # obstacle motions the reference is spatially meaningful there (e.g. "Box Up"
        # stands still on the box), so reaching it = completing the task regardless of
        # where the robot was spawned. We only score completion for envs that started
        # early enough to have actually traversed the clip (see `success_start_fraction`).
        total = self.motion.time_step_total
        self._goal_window_start = int((1.0 - self.cfg.success_final_fraction) * total)
        self._success_start_max = int(self.cfg.success_start_fraction * total)
        self._reached_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_start_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.completion_rate = 0.0  # scalar EMA over completed episodes, read by the curriculum

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["completion_rate"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def foot_contact_schedule(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Reference foot contact state at current timestep.

        Returns (left_contact, right_contact, valid_mask) each of shape (num_envs,) bool,
        or None if the motion data has no footlock information.
        """
        if self.motion.foot_contact_left is None:
            return None
        return (
            self.motion.foot_contact_left[self.time_steps],
            self.motion.foot_contact_right[self.time_steps],
            self.motion.foot_contact_mask[self.time_steps],
        )

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        self.metrics["completion_rate"][:] = self.completion_rate

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """True per-episode reset hook (timeout / termination), distinct from clip looping.

        Natural clip-wrap re-randomizes the RSI start mid-episode via ``_resample_command``
        but is *not* an episode boundary, so completion is scored here instead: record the
        just-ended episode's goal-reached flag, then arm the trackers for the fresh episode.
        """
        if env_ids is None:
            env_ids = slice(None)
        ids = (
            torch.arange(self.num_envs, device=self.device) if isinstance(env_ids, slice) else env_ids
        )
        self._record_completion(ids)
        extras = super().reset(env_ids)  # logs metrics + resamples a new episode start
        self._reached_goal[ids] = False
        self._episode_start_step[ids] = self.time_steps[ids]
        return extras

    def _record_completion(self, env_ids: torch.Tensor):
        """Fold the just-ended episodes' goal-reached flags into the completion EMA.

        Only episodes that *started* within ``success_start_fraction`` of the clip count,
        so a trivially-late RSI spawn beginning inside the goal window cannot inflate the
        metric — this measures "completed from a randomized start", which is the robustness
        signal the curriculum gates on.
        """
        if len(env_ids) == 0:
            return
        start = self._episode_start_step[env_ids]
        eligible = (start >= 0) & (start <= self._success_start_max)
        if eligible.any():
            reached = self._reached_goal[env_ids][eligible].float().mean().item()
            a = self.cfg.completion_ema
            self.completion_rate = (1.0 - a) * self.completion_rate + a * reached

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).clamp(0, self.motion.time_step_total - 1).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _snap_to_valid_rsi(self, env_ids: Sequence[int]):
        """Snap timesteps that land on invalid RSI frames to the nearest valid frame."""
        assert self.motion.rsi_mask is not None and self._valid_rsi_indices is not None
        sampled_ts = self.time_steps[env_ids]
        invalid = ~self.motion.rsi_mask[sampled_ts]
        if not invalid.any():
            return
        valid = self._valid_rsi_indices
        invalid_ts = sampled_ts[invalid]
        idx = torch.searchsorted(valid, invalid_ts)
        idx_right = idx.clamp(max=len(valid) - 1)
        idx_left = (idx - 1).clamp(min=0)
        dist_right = (valid[idx_right] - invalid_ts).abs()
        dist_left = (valid[idx_left] - invalid_ts).abs()
        nearest = torch.where(dist_left <= dist_right, valid[idx_left], valid[idx_right])
        self.time_steps[env_ids[invalid]] = nearest

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)
        if self._valid_rsi_indices is not None:
            self._snap_to_valid_rsi(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        if self.cfg.start_xy_noise_std > 0.0:
            early_mask = self.time_steps[env_ids] < int(0.1 * self.motion.time_step_total)
            if early_mask.any():
                xy_noise = (torch.rand(int(early_mask.sum()), 2, device=self.device) * 2 - 1) * self.cfg.start_xy_noise_std
                root_pos[env_ids[early_mask], :2] += xy_noise
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        # write to sim
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

        # --- goal-window tracking for the robustness curriculum ---
        # Latch "reached goal" once the anchor is within tolerance of the reference
        # while inside the final goal window. Absolute world-frame error (xyz) so the
        # signal stays tied to the obstacle/mesh, not to a re-anchored relative target.
        in_window = self.time_steps >= self._goal_window_start
        if in_window.any():
            anchor_err = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
            self._reached_goal |= in_window & (anchor_err < self.cfg.success_pos_threshold)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_w[:, i], self.body_quat_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    start_xy_noise_std: float = 0.0

    auto_episode_length: bool = True
    """If True, set env.episode_length_s to the clip's playback duration when the motion loads.

    Playback consumes one motion frame per env step, so the clip lasts
    ``time_step_total * decimation * sim.dt`` seconds. A shorter episode makes the final-10%
    goal window unreachable from an early spawn (completion stays 0, the curriculum stalls).
    Set False to keep a manually-tuned episode_length_s."""

    # --- robustness curriculum: motion-completion tracking ---
    success_final_fraction: float = 0.1
    """Final fraction of the clip treated as the goal window (spatially meaningful end pose)."""
    success_pos_threshold: float = 0.30
    """Absolute anchor xyz error (m) below which an env counts as having reached the goal."""
    success_start_fraction: float = 0.1
    """Only envs starting before this fraction of the clip contribute to ``completion_rate``.

    Aligned with the spawn xy-noise window in ``_resample_command`` so the metric tracks
    exactly the randomized-start envs we want robust — i.e. "can it complete from a
    randomized spawn", not "can it finish from a trivial near-the-end RSI point"."""
    completion_ema: float = 0.02
    """EMA coefficient for the running completion rate read by the curriculum."""

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

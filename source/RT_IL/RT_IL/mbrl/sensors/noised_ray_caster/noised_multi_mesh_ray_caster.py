from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.sensors.ray_caster import MultiMeshRayCaster
from isaaclab.sensors.ray_caster.ray_caster_data import RayCasterData
from isaaclab.sensors.ray_caster.ray_cast_utils import obtain_world_pose_from_view
from isaaclab.utils.math import quat_mul
from isaaclab.utils.warp import raycast_dynamic_meshes

if TYPE_CHECKING:
    from .noised_multi_mesh_ray_caster_cfg import NoisedMultiMeshRayCasterCfg


class NoisedMultiMeshRayCaster(MultiMeshRayCaster):
    """Multi-mesh ray caster with structured sensor noise for sim-to-real distillation.

    Shoots two sets of rays: clean (for teacher / critic) and noisy (for student policy).
    Exposes the same `.noisy_data` interface as NoisedRayCaster so the same
    `noised_height_scan` observation function works with both.
    """

    cfg: NoisedMultiMeshRayCasterCfg

    def __init__(self, cfg: NoisedMultiMeshRayCasterCfg):
        super().__init__(cfg)
        self._noisy_data = RayCasterData()

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()

        # Noise state (per-env, persistent across steps)
        self._noisy_data.pos_w = torch.zeros(self._view.count, 3, device=self.device)
        self._noisy_data.quat_w = torch.zeros(self._view.count, 4, device=self.device)
        self._noisy_data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        self.z_drift = torch.zeros(self._num_envs, device=self.device)
        self.xy_drift = torch.zeros(self._num_envs, 3, device=self.device)
        self.offset = torch.zeros(self._num_envs, 1, device=self.device)
        self.dropout = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)

        # Double ray arrays: first half = clean, second half = noisy
        self.half_num_rays = self.num_rays
        self.num_rays = 2 * self.num_rays
        self.ray_starts = self.ray_starts.repeat(1, 2, 1)
        self.ray_directions = self.ray_directions.repeat(1, 2, 1)

        # Re-allocate world-frame buffers to match doubled size
        self._ray_starts_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        self._ray_directions_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        # Combined hit buffer for clean + noisy rays
        self.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)

    def _update_buffers_impl(self, env_ids: torch.Tensor):
        # --- xy noise: perturb second-half ray starts in local sensor frame ---
        noisy_ray_starts = (
            self.cfg.xy_scale * torch.randn_like(self.ray_starts[env_ids, : self.half_num_rays, :])
            + self.xy_drift[env_ids].unsqueeze(1)
            + self.ray_starts[env_ids, : self.half_num_rays, :]
        )
        self.ray_starts[env_ids, self.half_num_rays :, :] = noisy_ray_starts

        # --- resample persistent noise state for all envs ---
        resample_drift = torch.rand_like(self.z_drift) < self.cfg.per_step_drift_prob
        self.z_drift[resample_drift] = self.cfg.drift_scale * torch.randn_like(self.z_drift[resample_drift])

        resample_xy_drift = torch.rand_like(self.xy_drift) < self.cfg.per_step_drift_prob
        self.xy_drift[resample_xy_drift] = self.cfg.xy_drift_scale * torch.randn_like(self.xy_drift[resample_xy_drift])

        resample_offset = torch.rand_like(self.offset) < self.cfg.per_step_offset_prob
        self.offset[resample_offset] = (
            self.cfg.offset_scale
            * torch.randn_like(self.offset[resample_offset])
            * (torch.rand_like(self.offset[resample_offset]) < self.cfg.offset_prob)
        )

        resample_dropout = torch.rand_like(self.z_drift) < self.cfg.per_step_dropout_prob
        self.dropout[resample_dropout] = torch.rand_like(self.z_drift[resample_dropout]) < self.cfg.dropout_prob

        # --- transform rays to world frame (also fills _data.pos_w / quat_w) ---
        self._update_ray_infos(env_ids)
        self._noisy_data.pos_w[env_ids] = self._data.pos_w[env_ids]
        self._noisy_data.quat_w[env_ids] = self._data.quat_w[env_ids]

        # --- update tracked mesh transforms (mirrors MultiMeshRayCaster logic) ---
        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue
            pos_w, ori_w = obtain_world_pose_from_view(view, None)
            pos_w = pos_w.squeeze(0) if pos_w.dim() == 3 else pos_w
            ori_w = ori_w.squeeze(0) if ori_w.dim() == 3 else ori_w
            if target_cfg.prim_expr in MultiMeshRayCaster.mesh_offsets:
                pos_offset, ori_offset = MultiMeshRayCaster.mesh_offsets[target_cfg.prim_expr]
                pos_w = pos_w - pos_offset
                ori_w = quat_mul(ori_offset.expand(ori_w.shape[0], -1), ori_w)
            count = view.count
            if count != 1:
                count = count // self._num_envs
                pos_w = pos_w.view(self._num_envs, count, 3)
                ori_w = ori_w.view(self._num_envs, count, 4)
            self._mesh_positions_w[:, mesh_idx : mesh_idx + count] = pos_w
            self._mesh_orientations_w[:, mesh_idx : mesh_idx + count] = ori_w
            mesh_idx += count

        # --- raycast all doubled rays in one call ---
        self.ray_hits_w[env_ids], _, _, _, _ = raycast_dynamic_meshes(
            self._ray_starts_w[env_ids],
            self._ray_directions_w[env_ids],
            mesh_ids_wp=self._mesh_ids_wp,
            max_dist=self.cfg.max_distance,
            mesh_positions_w=self._mesh_positions_w[env_ids],
            mesh_orientations_w=self._mesh_orientations_w[env_ids],
            return_mesh_id=False,
        )

    @property
    def data(self):
        self._update_outdated_buffers()
        self._data.ray_hits_w = self.ray_hits_w[:, : self.half_num_rays, :].clone()
        return self._data

    @property
    def noisy_data(self) -> RayCasterData:
        self._update_outdated_buffers()
        self._noisy_data.ray_hits_w = self.ray_hits_w[:, self.half_num_rays :, :].clone()
        z_data = self._noisy_data.ray_hits_w[..., 2]
        # point-wise z noise
        z_data += self.cfg.z_scale * torch.randn_like(z_data)
        # outliers
        outliers = torch.rand_like(z_data) < self.cfg.outlier_sample_prob
        z_data[outliers] += self.cfg.outlier_z_scale * torch.randn_like(z_data[outliers])
        # persistent drift and offset
        z_data += self.z_drift.unsqueeze(1)
        z_data += self.offset
        # dropout
        z_data[self.dropout] += self.cfg.dropout_z_scale * torch.randn_like(z_data[self.dropout])
        return self._noisy_data

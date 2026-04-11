from __future__ import annotations

import torch
from typing import Any, Dict
from isaaclab.sensors.ray_caster import RayCaster
from typing import TYPE_CHECKING
from isaaclab.sensors.ray_caster.ray_caster_data import RayCasterData
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
import omni.physics.tensors.impl.api as physx
import warp as wp
from isaacsim.core.prims import XFormPrim
if TYPE_CHECKING:
    from .noised_ray_caster_cfg import NoisedRayCasterCfg


class NoisedRayCaster(RayCaster):
    """Ray caster sensor with added noise for more realistic measurements."""
    
    cfg: NoisedRayCasterCfg
    
    def __init__(self, cfg: NoisedRayCasterCfg):
        super().__init__(cfg)
        self.cfg = cfg
        self._noisy_data = RayCasterData()
       
    
    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        self._noisy_data.pos_w = torch.zeros(self._view.count, 3, device=self.device)
        self._noisy_data.quat_w = torch.zeros(self._view.count, 4, device=self.device)
        self._noisy_data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
         # z drift applied to each body independently
        self.z_drift = torch.zeros(self._num_envs, device=self.device)
        # xy drift applied to each body independently
        self.xy_drift = torch.zeros(self._num_envs, 3, device=self.device)
        # z offset applied to all bodies
        self.offset = torch.zeros(self._num_envs, 1, device=self.device)
        # dropout applied to each body independently
        self.dropout = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        # when applying xy noise, we create a noisy copy of the original rays
        self.half_num_rays = self.num_rays
        self.num_rays = 2 * self.num_rays
        # -- repeat the tensors in the num_rays dimension
        self.ray_starts = self.ray_starts.repeat(1, 2, 1)
        self.ray_directions = self.ray_directions.repeat(1, 2, 1)
        self.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
   

    def _update_buffers_impl(self, env_ids: torch.Tensor):
        """Update the internal buffers and add noise to measurements."""
        # Call parent implementation
        noisy_ray_starts = (
            self.cfg.xy_scale * torch.randn_like(self.ray_starts[env_ids, : self.half_num_rays, :])
            + self.xy_drift[env_ids].unsqueeze(1)
            + self.ray_starts[env_ids, : self.half_num_rays, :]
        )
        self.ray_starts[env_ids, self.half_num_rays :, :] = noisy_ray_starts
        # resample drift
        resample_drift = torch.rand_like(self.z_drift) < self.cfg.per_step_drift_prob
        self.z_drift[resample_drift] = self.cfg.drift_scale * torch.randn_like(self.z_drift[resample_drift])
        # resample xy drift
        resample_xy_drift = torch.rand_like(self.xy_drift) < self.cfg.per_step_drift_prob
        self.xy_drift[resample_xy_drift] = self.cfg.xy_drift_scale * torch.randn_like(self.xy_drift[resample_xy_drift])
        # resample offset
        resample_offset = torch.rand_like(self.offset) < self.cfg.per_step_offset_prob
        self.offset[resample_offset] = (
            self.cfg.offset_scale
            * torch.randn_like(self.offset[resample_offset])
            * (torch.rand_like(self.offset[resample_offset]) < self.cfg.offset_prob)
        )
        # resample dropout
        resample_dropout = torch.rand_like(self.z_drift) < self.cfg.per_step_dropout_prob
        self.dropout[resample_dropout] = torch.rand_like(self.z_drift[resample_dropout]) < self.cfg.dropout_prob
        # obtain the poses of the sensors
        if isinstance(self._view, XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        # note: we clone here because we are read-only operations
        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        # apply drift
        pos_w += self.drift[env_ids]
        # store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w
        self._noisy_data.pos_w[env_ids] = pos_w
        self._noisy_data.quat_w[env_ids] = quat_w
        # ray cast based on the sensor poses
        if self.cfg.attach_yaw_only:
            # only yaw orientation is considered and directions are not rotated
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        else:
            # full orientation is considered
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        # ray cast and store the hits
        # TODO: Make this work for multiple meshes?
        self.ray_hits_w[env_ids] = raycast_mesh(
            ray_starts_w,
            ray_directions_w,
            max_dist=self.cfg.max_distance,
            mesh=self.meshes[self.cfg.mesh_prim_paths[0]],
        )[0]

    
    @property
    def data(self) -> RayCasterData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        self._data.ray_hits_w = self.ray_hits_w[:, :self.half_num_rays, :].clone()
        return self._data

    @property
    def noisy_data(self) -> RayCasterData:
        """Return the noisy ray caster data."""
        # update sensors if needed
        self._update_outdated_buffers()
        self._noisy_data.ray_hits_w = self.ray_hits_w[:, self.half_num_rays :, :].clone()
        z_data = self._noisy_data.ray_hits_w[...,2]
         # add noise
        # -- add z noise
        z_data += self.cfg.z_scale * torch.randn_like(z_data)
        # -- add outliers
        outliers = torch.rand_like(z_data) < self.cfg.outlier_sample_prob
        z_data[outliers] += self.cfg.outlier_z_scale * torch.randn_like(z_data[outliers])
        # -- add drift
        z_data += self.z_drift.unsqueeze(1)
        # -- add offset
        z_data += self.offset
        # -- add dropout
        z_data[self.dropout] += self.cfg.dropout_z_scale * torch.randn_like(z_data[self.dropout])
        # return the data with noise applied
        return self._noisy_data

    def _debug_vis_callback(self, event):
        # remove possible inf values
        original_points = self._data.ray_hits_w.reshape(-1, 3)
        original_points = original_points[~torch.any(torch.isinf(original_points), dim=1)]

        noisy_points = self._noisy_data.ray_hits_w.reshape(-1, 3)
        noisy_points = noisy_points[~torch.any(torch.isinf(noisy_points), dim=1)]

        all_points = torch.cat([original_points, noisy_points], dim=0)
        
        marker_indices = torch.cat([
            torch.zeros(original_points.shape[0], dtype=torch.int32),
            torch.ones(noisy_points.shape[0], dtype=torch.int32)
        ])
        # show ray hit positions
        self.ray_visualizer.visualize(all_points, marker_indices=marker_indices)

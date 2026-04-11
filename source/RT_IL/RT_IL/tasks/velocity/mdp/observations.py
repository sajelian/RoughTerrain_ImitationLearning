# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
#from mbrl.mbrl.sensors import NoisedRayCaster
from RT_IL.mbrl.sensors import NoisedRayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def body_contact(env: ManagerBasedEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Contact status of the body of the asset."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return is_contact


def noised_height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: NoisedRayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.noisy_data.pos_w[:, 2].unsqueeze(1) - sensor.noisy_data.ray_hits_w[..., 2] - offset

"""Observation functions for the velocity-tracking + joystick-distillation env."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def velocity_command_joystick(
    env: "ManagerBasedEnv",
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Return the continuous velocity command (vx, vy, wz) as the student's joystick obs.

    Intentionally NOT quantized — the student trains on whatever the velocity
    PPO was trained on (continuous), and at deploy the joystick value can be
    anything from an analog gamepad. The skill-mode toggle (separate obs) is
    what disambiguates "use the locomotion policy" vs "engage an expert skill",
    not the joystick value.

    Drops any heading dim if the command term emits one (heading_command=True).
    """
    cmd = env.command_manager.get_command(command_name)
    return cmd[:, :3]

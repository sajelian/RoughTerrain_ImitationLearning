"""Keyboard obs functions for the deploy env, with auto-coupling between
joystick and skill_mode so the user can hold both keys simultaneously.

Coupling rule: when the skill-mode key is held, the joystick obs is forced
to zero, regardless of WASD state. This matches the training distribution
(skill-expert data always has joystick=0) so deploy state stays in-distribution
even when the user is mid-button-swap.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _skill_mode_kb(env, key: str = "SPACE"):
    """Lazily-initialised shared Se2Keyboard listener for the skill-mode toggle.

    Both joystick and skill_mode obs terms call this so they read a single
    coherent key state and the listener is only registered once.
    """
    if not hasattr(env, "_skill_mode_kb_listener"):
        import numpy as np
        from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

        kb = Se2Keyboard(Se2KeyboardCfg(
            v_x_sensitivity=1.0,
            v_y_sensitivity=0.0,
            omega_z_sensitivity=0.0,
            sim_device=env.device,
        ))
        kb._INPUT_KEY_MAPPING.clear()
        kb._INPUT_KEY_MAPPING[key] = np.array([1.0, 0.0, 0.0])
        env._skill_mode_kb_listener = kb
        env._skill_mode_prev = -1
        print(f"[deploy] HOLD {key} to engage skill mode; release for velocity mode.")
    return env._skill_mode_kb_listener


def _is_skill_held(env, key: str = "SPACE") -> bool:
    raw = _skill_mode_kb(env, key).advance()
    return raw[0].item() > 0.5


def deploy_skill_mode(env: "ManagerBasedEnv", key: str = "SPACE") -> torch.Tensor:
    """1-bit skill mode obs. Reads from the shared keyboard listener."""
    val = 1.0 if _is_skill_held(env, key) else 0.0
    if int(val) != getattr(env, "_skill_mode_prev", -1):
        print(f"[deploy] skill_mode -> {int(val)}")
        env._skill_mode_prev = int(val)
    return torch.full((env.num_envs, 1), val, dtype=torch.float32, device=env.device)


def deploy_keyboard_joystick(
    env: "ManagerBasedEnv",
    forward_speed: float = 0.8,
    turn_slow: float = 0.785,
    turn_fast: float = 1.571,
    skill_key: str = "SPACE",
) -> torch.Tensor:
    """Keyboard WASD/QE/ZX → joystick, auto-zeroed when skill_key is held.

    Uses a separate Se2Keyboard listener for WASD; queries the shared skill-mode
    listener to decide whether to override with zeros. The skill-mode listener
    is initialised by `_skill_mode_kb` on first call from either obs term.
    """
    # Auto-coupling: if user is engaging skill mode, joystick is suppressed.
    if _is_skill_held(env, skill_key):
        return torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)

    if not hasattr(env, "_deploy_wasd_kb"):
        import numpy as np
        from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

        kb = Se2Keyboard(Se2KeyboardCfg(
            v_x_sensitivity=1.0,
            v_y_sensitivity=0.0,
            omega_z_sensitivity=1.0,
            sim_device=env.device,
        ))
        kb._INPUT_KEY_MAPPING.update({
            "W": np.array([1.0, 0.0, 0.0]),
            "UP": np.array([1.0, 0.0, 0.0]),
            "NUMPAD_8": np.array([1.0, 0.0, 0.0]),
            "A": np.array([0.0, 0.0, 1.0]),
            "LEFT": np.array([0.0, 0.0, 1.0]),
            "NUMPAD_4": np.array([0.0, 0.0, 1.0]),
            "D": np.array([0.0, 0.0, -1.0]),
            "RIGHT": np.array([0.0, 0.0, -1.0]),
            "NUMPAD_6": np.array([0.0, 0.0, -1.0]),
            "Z": np.array([0.0, 0.0, 2.0]),
            "NUMPAD_7": np.array([0.0, 0.0, 2.0]),
            "X": np.array([0.0, 0.0, -2.0]),
            "NUMPAD_9": np.array([0.0, 0.0, -2.0]),
        })
        for k in ("S", "DOWN", "NUMPAD_2"):
            kb._INPUT_KEY_MAPPING.pop(k, None)
        env._deploy_wasd_kb = kb
        print(f"[deploy] W=forward({forward_speed})  A/D=±{turn_slow:.2f}  Z/X=±{turn_fast:.2f}  "
              f"({skill_key} held → joystick auto-zeroed)")

    raw = env._deploy_wasd_kb.advance()
    vx = forward_speed if raw[0].item() > 0.5 else 0.0
    r = raw[2].item()
    if r > 1.5:
        wz = turn_fast
    elif r > 0.5:
        wz = turn_slow
    elif r < -1.5:
        wz = -turn_fast
    elif r < -0.5:
        wz = -turn_slow
    else:
        wz = 0.0
    cmd = torch.tensor([vx, 0.0, wz], dtype=torch.float32, device=env.device)
    return cmd.unsqueeze(0).expand(env.num_envs, -1)

"""Curriculum terms for the motion-tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg

    from RT_IL.tasks.tracking.mdp.commands import MotionCommand


def _lerp(bounds: tuple[float, float], t: float) -> float:
    return bounds[0] + (bounds[1] - bounds[0]) * t


class robustness_curriculum(ManagerTermBase):
    """Metric-driven curriculum that trades mimic fidelity for start-pose robustness.

    A single scalar difficulty ``d in [0, 1]`` linearly interpolates every robustness
    knob between an *easy* endpoint (the current pure-mimic config: tight terminations,
    little spawn randomization, goal reward off) and a *hard* endpoint (large spawn
    randomization, loose terminations, goal reward on, foot-tracking relaxed so an
    adjustment step is affordable). Each ``*`` arg below is an ``(easy, hard)`` tuple.

    ``d`` advances only when the policy *reliably completes the motion* at the current
    difficulty — measured by :attr:`MotionCommand.completion_rate`, the fraction of
    early-spawned envs that reach the goal window. If completion later collapses, ``d``
    backs off. This hysteresis is what prevents the catastrophic, irreversible ramps a
    fixed step-count schedule produces.

    All knobs are read live by their managers, so the term just mutates the stored cfgs in
    place each call. The easy endpoints reproduce the base config at ``d = 0``, so enabling
    the curriculum changes nothing until the agent earns it. Set ``freeze=True`` (eval/PLAY)
    to hold ``d`` at ``start_difficulty`` without advancing.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.difficulty: float = float(cfg.params.get("start_difficulty", 0.0))
        self._last_update_step: int = 0
        self._command: MotionCommand | None = None  # resolved lazily (managers built after curriculum)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        start_difficulty: float = 0.0,
        freeze: bool = False,
        warmup_steps: int = 500,
        cooldown_steps: int = 1500,
        advance_threshold: float = 0.8,
        backoff_threshold: float = 0.5,
        difficulty_step: float = 0.05,
        backoff_step: float = 0.05,
        anchor_term: str = "anchor_pos",
        feet_term: str = "ee_feet_pos",
        success_reward: str = "motion_terminal_goal",
        foot_contact_reward: str = "foot_contact_schedule",
        feet_pos_reward: str = "motion_global_feet_body_pos",
        start_xy_noise_std: tuple[float, float] = (0.10, 0.40),
        pose_yaw: tuple[float, float] = (0.10, 0.50),
        anchor_threshold_xy: tuple[float, float] = (0.35, 0.85),
        feet_threshold_xy: tuple[float, float] = (0.35, 0.60),
        success_weight: tuple[float, float] = (0.0, 2.0),
        foot_contact_weight: tuple[float, float] = (2.0, 1.0),
        feet_pos_weight: tuple[float, float] = (1.0, 0.5),
    ) -> dict[str, float]:
        if self._command is None:
            self._command = env.command_manager.get_term(command_name)
        completion = float(self._command.completion_rate)

        if not freeze:
            step = env.common_step_counter
            if step >= warmup_steps and (step - self._last_update_step) >= cooldown_steps:
                if completion >= advance_threshold and self.difficulty < 1.0:
                    self.difficulty = min(1.0, self.difficulty + difficulty_step)
                    self._last_update_step = step
                elif completion < backoff_threshold and self.difficulty > 0.0:
                    self.difficulty = max(0.0, self.difficulty - backoff_step)
                    self._last_update_step = step

        d = self.difficulty
        command = self._command
        log: dict[str, float] = {"difficulty": d, "completion_rate": completion}

        # --- spawn randomization (command cfg, read each reset) ---
        command.cfg.start_xy_noise_std = _lerp(start_xy_noise_std, d)
        log["start_xy_noise_std"] = command.cfg.start_xy_noise_std
        yaw = _lerp(pose_yaw, d)
        command.cfg.pose_range["yaw"] = (-yaw, yaw)
        log["pose_yaw"] = yaw

        # --- termination slack (let the adjustment step happen without dying) ---
        for term_name, bounds, key in (
            (anchor_term, anchor_threshold_xy, "anchor_threshold_xy"),
            (feet_term, feet_threshold_xy, "feet_threshold_xy"),
        ):
            if term_name in env.termination_manager.active_terms:
                val = _lerp(bounds, d)
                env.termination_manager.get_term_cfg(term_name).params["threshold_xy"] = val
                log[key] = val

        # --- reward reweighting (success up; foot-tracking down ~ const-sum) ---
        for term_name, bounds, key in (
            (success_reward, success_weight, "success_weight"),
            (foot_contact_reward, foot_contact_weight, "foot_contact_weight"),
            (feet_pos_reward, feet_pos_weight, "feet_pos_weight"),
        ):
            if term_name in env.reward_manager.active_terms:
                val = _lerp(bounds, d)
                env.reward_manager.get_term_cfg(term_name).weight = val
                log[key] = val

        return log

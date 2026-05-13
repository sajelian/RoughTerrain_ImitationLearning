from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RandomFootWrench(ManagerTermBase):
    """Apply short random wrenches to selected foot links at random intervals."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.time_until_next = torch.zeros(env.num_envs, device=self.asset.device)
        self.active_time_left = torch.zeros(env.num_envs, device=self.asset.device)
        self.is_active = torch.zeros(env.num_envs, dtype=torch.bool, device=self.asset.device)

        event_interval_range_s = cfg.params["event_interval_range_s"]
        self.time_until_next[:] = math_utils.sample_uniform(
            event_interval_range_s[0],
            event_interval_range_s[1],
            (env.num_envs,),
            self.asset.device,
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.asset.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.asset.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.asset.device)
        self._clear_wrench(env_ids)
        self.is_active[env_ids] = False
        self.active_time_left[env_ids] = 0.0
        event_interval_range_s = self.cfg.params["event_interval_range_s"]
        self.time_until_next[env_ids] = math_utils.sample_uniform(
            event_interval_range_s[0],
            event_interval_range_s[1],
            (len(env_ids),),
            self.asset.device,
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        force_range: dict[str, tuple[float, float]],
        torque_range: dict[str, tuple[float, float]],
        event_interval_range_s: tuple[float, float],
        duration_range_s: tuple[float, float],
    ):
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=self.asset.device)

        dt = env.step_dt
        active_env_ids = env_ids[self.is_active[env_ids]]
        if len(active_env_ids) > 0:
            self.active_time_left[active_env_ids] -= dt
            ended_env_ids = active_env_ids[self.active_time_left[active_env_ids] <= 0.0]
            if len(ended_env_ids) > 0:
                self._clear_wrench(ended_env_ids)
                self.is_active[ended_env_ids] = False
                self.active_time_left[ended_env_ids] = 0.0
                self.time_until_next[ended_env_ids] = math_utils.sample_uniform(
                    event_interval_range_s[0],
                    event_interval_range_s[1],
                    (len(ended_env_ids),),
                    self.asset.device,
                )

        inactive_env_ids = env_ids[~self.is_active[env_ids]]
        if len(inactive_env_ids) == 0:
            return

        self.time_until_next[inactive_env_ids] -= dt
        start_env_ids = inactive_env_ids[self.time_until_next[inactive_env_ids] <= 0.0]
        if len(start_env_ids) == 0:
            return

        self._apply_random_wrench(start_env_ids, asset_cfg, force_range, torque_range)
        self.is_active[start_env_ids] = True
        self.active_time_left[start_env_ids] = math_utils.sample_uniform(
            duration_range_s[0],
            duration_range_s[1],
            (len(start_env_ids),),
            self.asset.device,
        )

    def _apply_random_wrench(
        self,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        force_range: dict[str, tuple[float, float]],
        torque_range: dict[str, tuple[float, float]],
    ) -> None:
        num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else self.asset.num_bodies
        forces = self._sample_axis_ranges(force_range, (len(env_ids), num_bodies, 3))
        torques = self._sample_axis_ranges(torque_range, (len(env_ids), num_bodies, 3))
        self.asset.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=asset_cfg.body_ids,
            env_ids=env_ids,
        )

    def _clear_wrench(self, env_ids: torch.Tensor) -> None:
        num_bodies = len(self.asset_cfg.body_ids) if isinstance(self.asset_cfg.body_ids, list) else self.asset.num_bodies
        zeros = torch.zeros((len(env_ids), num_bodies, 3), device=self.asset.device)
        self.asset.permanent_wrench_composer.set_forces_and_torques(
            forces=zeros,
            torques=zeros,
            body_ids=self.asset_cfg.body_ids,
            env_ids=env_ids,
        )

    def _sample_axis_ranges(self, axis_range: dict[str, tuple[float, float]], size: tuple[int, int, int]) -> torch.Tensor:
        range_list = [axis_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device=self.asset.device)
        return math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], size, self.asset.device)


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)

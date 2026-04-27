# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os

import onnx
import torch

from isaaclab.envs import ManagerBasedRLEnv


def infer_export_observation_group(env: ManagerBasedRLEnv, preferred_group: str | None = None) -> str:
    """Pick the deployable observation group for ONNX export metadata."""
    if preferred_group is not None:
        return preferred_group

    active_terms = getattr(env.observation_manager, "active_terms", {})
    if "student" in active_terms:
        return "student"
    return "policy"


def select_export_normalizer(policy: object, obs_group: str) -> object | None:
    """Select the observation normalizer that matches the exported policy input."""
    if obs_group == "student" and hasattr(policy, "student_obs_normalizer"):
        return policy.student_obs_normalizer
    if hasattr(policy, "actor_obs_normalizer"):
        return policy.actor_obs_normalizer
    if hasattr(policy, "obs_normalizer"):
        return policy.obs_normalizer
    return None


def _resolve_export_actor(policy: object, prefer_student: bool) -> tuple[torch.nn.Module, int]:
    """Resolve the actor module and its observation dimension."""
    if prefer_student and hasattr(policy, "student"):
        actor = copy.deepcopy(policy.student)
        obs_dim = getattr(policy, "student_obs_dim", None)
        if obs_dim is None:
            obs_dim = getattr(policy, "obs_dim", actor[0].in_features)
        return actor, obs_dim

    if hasattr(policy, "mlp"):
        actor = copy.deepcopy(policy.mlp)
        obs_dim = getattr(policy, "obs_dim", actor[0].in_features)
        return actor, obs_dim

    if hasattr(policy, "actor"):
        actor = copy.deepcopy(policy.actor)
        obs_dim = getattr(policy, "obs_dim", actor[0].in_features)
        return actor, obs_dim

    if hasattr(policy, "student"):
        actor = copy.deepcopy(policy.student)
        obs_dim = getattr(policy, "student_obs_dim", None)
        if obs_dim is None:
            obs_dim = getattr(policy, "obs_dim", actor[0].in_features)
        return actor, obs_dim

    raise ValueError("Unsupported policy object for ONNX export: no mlp/actor/student module found.")


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
    prefer_student: bool = False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(
        env,
        actor_critic,
        normalizer,
        verbose,
        prefer_student=prefer_student,
    )
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(torch.nn.Module):
    def __init__(self, env: ManagerBasedRLEnv, actor_model, normalizer=None, verbose=False, prefer_student: bool = False):
        torch.nn.Module.__init__(self)
        self.verbose = verbose
        self.is_recurrent = getattr(actor_model, "is_recurrent", False)
        self.actor, self.obs_dim = _resolve_export_actor(actor_model, prefer_student=prefer_student)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.obs_dim)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr
    )


def attach_onnx_metadata(
    env: ManagerBasedRLEnv,
    run_path: str,
    path: str,
    filename="policy.onnx",
    obs_group: str | None = None,
) -> None:
    onnx_path = os.path.join(path, filename)

    obs_group = infer_export_observation_group(env, preferred_group=obs_group)
    observation_names = env.observation_manager.active_terms[obs_group]
    observation_history_lengths: list[int] = []
    group_cfg = getattr(env.observation_manager.cfg, obs_group)

    if group_cfg.history_length is not None:
        observation_history_lengths = [group_cfg.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = group_cfg.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_group": obs_group,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist()
        if hasattr(env.action_manager.get_term("joint_pos")._scale, "cpu")
        else env.action_manager.get_term("joint_pos")._scale,
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)

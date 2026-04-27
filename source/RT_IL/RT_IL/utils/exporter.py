# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv



def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(torch.nn.Module):
    def __init__(self, env: ManagerBasedRLEnv, actor_model, normalizer=None, verbose=False):
        torch.nn.Module.__init__(self)
        self.verbose = verbose
        self.is_recurrent = actor_model.is_recurrent
        self.obs_dim = actor_model.obs_dim
        self.actor = copy.deepcopy(actor_model.mlp)
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
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist() if hasattr(env.action_manager.get_term("joint_pos")._scale, 'cpu') else env.action_manager.get_term("joint_pos")._scale,
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

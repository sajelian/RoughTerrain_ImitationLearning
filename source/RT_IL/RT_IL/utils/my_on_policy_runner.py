import os

from rsl_rl.env import VecEnv
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from RT_IL.utils.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
    infer_export_observation_group,
    select_export_normalizer,
)


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            policy_nn = self.alg.get_policy()
            export_policy_as_onnx(policy_nn, normalizer=getattr(policy_nn, "obs_normalizer", None), path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            policy_nn = self.alg.get_policy()
            obs_group = infer_export_observation_group(self.env.unwrapped)
            normalizer = select_export_normalizer(policy_nn, obs_group)
            export_motion_policy_as_onnx(
                self.env.unwrapped,
                policy_nn,
                normalizer=normalizer,
                path=policy_path,
                filename=filename,
                prefer_student=(obs_group == "student"),
            )
            attach_onnx_metadata(
                self.env.unwrapped,
                wandb.run.name,
                path=policy_path,
                filename=filename,
                obs_group=obs_group,
            )
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None


class MotionDistillationRunner(DistillationRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            policy_nn = self.alg.get_policy()
            obs_group = infer_export_observation_group(self.env.unwrapped)
            normalizer = select_export_normalizer(policy_nn, obs_group)
            export_motion_policy_as_onnx(
                self.env.unwrapped,
                policy_nn,
                normalizer=normalizer,
                path=policy_path,
                filename=filename,
                prefer_student=(obs_group == "student"),
            )
            attach_onnx_metadata(
                self.env.unwrapped,
                wandb.run.name,
                path=policy_path,
                filename=filename,
                obs_group=obs_group,
            )
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None

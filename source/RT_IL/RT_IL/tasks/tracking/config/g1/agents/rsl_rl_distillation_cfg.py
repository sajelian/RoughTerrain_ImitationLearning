# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)


@configclass
class G1MimicMeshDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_mimic_mesh"
    clip_actions = 100.0
    obs_groups = {"policy": ["student"], "teacher": ["policy"], "student": ["student"]}
    student = RslRlMLPModelCfg(
        # hidden_dims=[1024, 512, 256, 128],
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="scalar"),
    )
    teacher = RslRlMLPModelCfg(
        # hidden_dims=[1024, 512, 256, 128],
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="scalar"),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=1,
        learning_rate=2.e-4,
        loss_type="huber",
        gradient_length=8,
        max_grad_norm=1.0,
    )

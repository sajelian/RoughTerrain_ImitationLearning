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
class UnitreeG1RoughDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 120
    max_iterations = 1500
    save_interval = 50
    experiment_name = "g1_rough"
    clip_actions = 100.0
    obs_groups = {"policy": ["student"], "teacher": ["policy"], "student": ["student"]}
    student = RslRlMLPModelCfg(
        hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="scalar"),
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="scalar"),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
        # max_grad_norm=1.0,
    )


@configclass
class UnitreeG1FlatDistillationRunnerCfg(UnitreeG1RoughDistillationRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.student.hidden_dims = [128, 128, 128]
        self.teacher.hidden_dims = [128, 128, 128]

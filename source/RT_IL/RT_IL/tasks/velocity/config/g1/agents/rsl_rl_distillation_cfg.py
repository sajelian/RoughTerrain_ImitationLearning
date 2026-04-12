# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
    # use this for horizon > 1? RslRlDistillationStudentTeacherRecurrentCfg
)


@configclass
class UnitreeG1RoughDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 120
    max_iterations = 500
    save_interval = 50
    experiment_name = "g1_rough"
    obs_groups = {"policy": ["student"], "teacher": ["policy"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )


@configclass
class UnitreeG1FlatDistillationRunnerCfg(UnitreeG1RoughDistillationRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.policy.student_hidden_dims = [128, 128, 128]
        self.policy.teacher_hidden_dims = [128, 128, 128]

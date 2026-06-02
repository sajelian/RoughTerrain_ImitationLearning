import gymnasium as gym

from . import agents


gym.register(
    id="Velocity-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1VelocityDistillationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1VelocityDistillationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-G1-Obstacle-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Flat terrain + obstacle prim + robot spawns at motion's first-frame anchor.
        # Used for collecting velocity-PPO data on skill terrains so the student
        # learns the (heightmap has box, skill_mode=0) → velocity actions case.
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1VelocityObstacleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityPPORunnerCfg",
    },
)

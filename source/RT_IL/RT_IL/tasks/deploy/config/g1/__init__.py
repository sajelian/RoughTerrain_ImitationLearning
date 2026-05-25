import gymnasium as gym

from . import agents


gym.register(
    id="Deploy-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1DeployEnvCfg",
        # play_fql.py's hydra_task_config decorator wants an rsl_rl_cfg_entry_point;
        # we never actually use the agent cfg here (no training in the deploy env).
        # Reuse the existing PPO runner cfg as a harmless stand-in.
        "rsl_rl_cfg_entry_point": "RT_IL.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg:G1MeshPPORunnerCfg",
    },
)

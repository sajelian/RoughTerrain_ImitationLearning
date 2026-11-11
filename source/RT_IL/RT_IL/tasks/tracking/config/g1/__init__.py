import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Mesh-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mesh_env_cfg:G1MeshEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MeshPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Mesh-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mesh_env_cfg:G1MimicMeshEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MeshPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Mesh-Deploy-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mesh_env_cfg:G1MeshEnvDeployCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MeshPPODeployRunnerCfg",
    },
)

gym.register(
    id="Tracking-Mesh-Deploy-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mesh_env_cfg:G1MimicMeshEnvDeployCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MeshPPODeployRunnerCfg",
    },
)

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    arg_group.add_argument(
        "--wandb_path", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def add_difficulty_arg(parser: argparse.ArgumentParser):
    """Add the ``--difficulty`` flag for freezing the robustness curriculum during eval/collection."""
    parser.add_argument(
        "--difficulty",
        type=float,
        default=None,
        help="Freeze the robustness curriculum at this difficulty in [0, 1] "
        "(1.0 = full spawn randomization + loosened terminations the expert trained against). "
        "No-op for envs without the curriculum.",
    )


def apply_difficulty_override(env_cfg, difficulty: float | None) -> float | None:
    """Freeze the robustness curriculum at a fixed difficulty, if the env has one.

    Reuses the exact same knobs (spawn randomization, termination slack, reward weights)
    the policy trained against, so collected rollouts match the expert's envelope. Returns
    the resolved difficulty (or None if no override / no curriculum), for logging.
    """
    if difficulty is None:
        return None
    robustness = getattr(getattr(env_cfg, "curriculum", None), "robustness", None)
    if robustness is None:
        print("[WARN] --difficulty given but this env has no robustness curriculum term; ignoring.")
        return None
    robustness.params["start_difficulty"] = difficulty
    robustness.params["freeze"] = True
    print(f"[INFO] Robustness curriculum frozen at difficulty={difficulty}")
    return difficulty


def add_skill_mode_arg(parser: argparse.ArgumentParser):
    """Add the ``--skill_mode`` flag for overriding the skill-mode bit during collection."""
    parser.add_argument(
        "--skill_mode",
        type=float,
        default=None,
        help="Override the constant skill-mode bit written into the student obs "
        "(0.0 = velocity/locomotion regime, 1.0 = terrain-driven skill regime). "
        "Default: leave the env cfg's value untouched (1.0 for the joystick-distillation "
        "env, 0.0 for the velocity env). Set 0.0 to collect a tracking expert "
        "(e.g. idle / stand-still) as velocity-regime data.",
    )


def apply_skill_mode_override(env_cfg, skill_mode: float | None) -> float | None:
    """Override the constant skill-mode bit in the student obs group, if present.

    The student obs carries a 1-bit skill_mode that disambiguates 'follow the velocity
    command' (0) from 'engage a terrain-driven skill' (1). Each data source is collected
    with a single fixed value. Most experts use the env-cfg default, but a stand-still /
    idle tracking expert belongs in the velocity regime (skill_mode=0, joystick=0), so its
    collection overrides the joystick-distillation env's default of 1.0. Returns the
    resolved value (or None if no override / no such term), for logging.
    """
    if skill_mode is None:
        return None
    student = getattr(getattr(env_cfg, "observations", None), "student", None)
    term = getattr(student, "skill_mode", None)
    if term is None or not hasattr(term, "params"):
        print("[WARN] --skill_mode given but this env's student obs has no skill_mode term; ignoring.")
        return None
    term.params["value"] = skill_mode
    print(f"[INFO] skill_mode bit overridden to {skill_mode}")
    return skill_mode


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg

"""Base config for the FQL student deploy environment.

Clean break from `TrackingEnvCfg` — no motion command, no motion-tracking
rewards/terminations, no teacher obs group. The deploy env exists *only* to
run a trained joystick+skill-mode student policy: it provides robot,
terrain, an optional obstacle USD, the matching height scanner, the
student obs (joystick + skill_mode + proprio + scan), and a minimal
termination set so the env auto-resets if the robot falls.

Concrete robot config lives in `config/g1/env_cfg.py:G1DeployEnvCfg`.
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import RT_IL.tasks.deploy.mdp as mdp


# Must match the tracking-env / velocity-env student layout for compatibility
# with a trained FQL checkpoint (obs_dim depends on these).
PROPIO_HISTORY_SHORT = 5
PROPIO_HISTORY_LONG = 8


##
# Scene
##
@configclass
class DeploySceneCfg(InteractiveSceneCfg):
    """Flat ground + optional obstacle USD prim, same scanner shape as tracking env."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(100.0, 100.0),
            num_rows=1,
            num_cols=1,
            border_width=0.0,
            sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg()},
        ),
        use_terrain_origins=False,
        env_spacing=2.5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    robot: ArticulationCfg = MISSING
    obstacle: AssetBaseCfg = MISSING  # filled by the G1 config; play_fql.py swaps usd_path at runtime

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )

    # Heightmap geometry matched to the tracking env so the student obs is the
    # same dim as what the FQL checkpoint was trained on.
    height_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        history_length=0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.2, 0.6]),
        debug_vis=True,
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/obstacle",
                is_shared=True,
                track_mesh_transforms=False,
            ),
        ],
    )


##
# Commands — none (no motion clip, no velocity sampler; user drives via keyboard)
##
@configclass
class CommandsCfg:
    pass


##
# Actions
##
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=True
    )


##
# Observations — only the student group (the deployed FQL one-step policy reads it).
##
@configclass
class ObservationsCfg:

    @configclass
    class StudentPolicyCfg(ObsGroup):
        """Byte-for-byte identical to the training student obs (tracking +
        velocity envs). joystick + skill_mode come from keyboard with
        auto-coupling so that SPACE-held forces joystick to zero.
        """

        joystick = ObsTerm(
            func=mdp.deploy_keyboard_joystick,
            params={
                "forward_speed": 0.8,
                "turn_slow": 0.785,
                "turn_fast": 1.571,
                "skill_key": "SPACE",
            },
            history_length=PROPIO_HISTORY_SHORT,
        )
        skill_mode = ObsTerm(
            func=mdp.deploy_skill_mode,
            params={"key": "SPACE"},
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=PROPIO_HISTORY_LONG,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=PROPIO_HISTORY_SHORT,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=PROPIO_HISTORY_LONG,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            scale=0.05,
            history_length=PROPIO_HISTORY_LONG,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=PROPIO_HISTORY_SHORT)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            # Match training: corruption + flatten.
            self.enable_corruption = True
            self.concatenate_terms = True

    # play_fql.py reads obs["student"], so the group key matters.
    student: StudentPolicyCfg = StudentPolicyCfg()


##
# Rewards — empty (deploy doesn't train, rewards aren't used).
##
@configclass
class RewardsCfg:
    pass


##
# Terminations — only generic safety triggers so play auto-resets on falls.
##
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(60.0)},  # lenient — only fully tipped
    )


##
# Events — empty by default. Optionally add push_robot in concrete configs.
##
@configclass
class EventCfg:
    pass


##
# Top-level env config
##
@configclass
class DeployEnvCfg(ManagerBasedRLEnvCfg):
    scene: DeploySceneCfg = DeploySceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 60.0  # long episodes — play is interactive
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

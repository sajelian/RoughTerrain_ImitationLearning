from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

from .noised_multi_mesh_ray_caster import NoisedMultiMeshRayCaster


@configclass
class NoisedMultiMeshRayCasterCfg(MultiMeshRayCasterCfg):
    """Configuration for the noised multi-mesh ray caster sensor.

    Mirrors NoisedRayCasterCfg noise parameters so the same distillation
    observation functions work regardless of sensor type.
    """

    class_type: type = NoisedMultiMeshRayCaster

    # Disable mesh-id tracking — incompatible with doubled ray arrays
    update_mesh_ids: bool = False

    # point-wise noise added every step
    xy_scale: float = 0.005
    z_scale: float = 0.01
    outlier_sample_prob: float = 0.05
    outlier_z_scale: float = 0.1

    # drift randomly resampled over time
    per_step_drift_prob: float = 0.003
    drift_scale: float = 0.1
    xy_drift_scale: float = 0.04

    # offset (large bias applied to all bodies) randomly resampled over time
    per_step_offset_prob: float = 0.003
    offset_scale: float = 1.0
    offset_prob: float = 0.05

    # dropout (whole-env data corruption) randomly resampled over time
    per_step_dropout_prob: float = 0.003
    dropout_prob: float = 0.1
    dropout_z_scale: float = 1.0

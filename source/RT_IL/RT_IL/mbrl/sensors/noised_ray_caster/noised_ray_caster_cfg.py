from isaaclab.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from .noised_ray_caster import NoisedRayCaster
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG


NOISED_RAY_CASTER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "original": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "noisy": sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)

@configclass
class NoisedRayCasterCfg(RayCasterCfg):
    class_type: type = NoisedRayCaster
    """Configuration for the noised ray caster sensor."""

    # noise added at every step
    xy_scale: float = 0.005
    z_scale: float = 0.01
    outlier_sample_prob: float = 0.05
    outlier_z_scale: float = 0.1

    # drift randomly resampled over time
    per_step_drift_prob: float = 0.003  # prob of resampling the drift
    drift_scale: float = 0.1
    xy_drift_scale: float = 0.04
    # offset (large drift applied to all bodies) randomly resampled over time
    per_step_offset_prob: float = 0.003  # prob of resampling the offset
    offset_scale: float = 1.0
    offset_prob: float = 0.05
    # dropout (missing / highly corrupted data) randomly resampled over time
    per_step_dropout_prob: float = 0.003  # prob of resampling
    dropout_prob: float = 0.1
    dropout_z_scale: float = 1.0  # std of the dropout

    visualizer_cfg: VisualizationMarkersCfg = NOISED_RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/NoisedRayCaster")

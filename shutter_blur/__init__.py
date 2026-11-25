"""Motion-field shutter smear (motion blur) effect toolkit."""
from .demo_scenes import (
    DIAL_LAYER_ID,
    STRIP_LAYER_ID,
    dial_scene,
    spoke_world_angle,
    strip_center_x,
)
from .integrators import (
    METHODS,
    gaussian_blur,
    ground_truth,
    render_shutter,
    segment_count_map,
)
from .kernel import DIRECTIONS, PROFILES, ShutterError, ShutterKernel, profile_window
from .metrics import (
    contour_energy,
    motion_roi,
    rmse,
    static_roi,
)
from .motion import (
    FlowMotionField,
    SceneMotionField,
    bilinear_sample,
    block_match_flow,
    consistency_occlusion_mask,
)
from .pipeline import (
    VideoSource,
    make_scene,
    render_all_methods,
    render_scene_frame,
    synthetic_video_source,
)
from .rolling import (
    RollingResult,
    render_rolling,
    rolling_ground_truth,
    row_sample_times,
)
from .scene import SCENES, Scene

__all__ = [
    "ShutterKernel", "ShutterError", "render_shutter", "ground_truth",
    "gaussian_blur", "segment_count_map", "METHODS", "Scene", "SCENES",
    "SceneMotionField", "FlowMotionField", "bilinear_sample",
    "block_match_flow", "consistency_occlusion_mask", "VideoSource",
    "make_scene", "render_scene_frame", "render_all_methods",
    "synthetic_video_source", "rmse", "contour_energy", "motion_roi",
    "static_roi", "PROFILES", "DIRECTIONS", "profile_window",
    "render_rolling", "rolling_ground_truth", "row_sample_times",
    "RollingResult", "dial_scene", "spoke_world_angle", "strip_center_x",
    "DIAL_LAYER_ID", "STRIP_LAYER_ID",
]

"""Motion-field shutter smear (motion blur) effect toolkit."""
from .integrators import (
    METHODS,
    gaussian_blur,
    ground_truth,
    render_shutter,
    segment_count_map,
)
from .kernel import ShutterError, ShutterKernel
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
from .scene import SCENES, Scene

__all__ = [
    "ShutterKernel", "ShutterError", "render_shutter", "ground_truth",
    "gaussian_blur", "segment_count_map", "METHODS", "Scene", "SCENES",
    "SceneMotionField", "FlowMotionField", "bilinear_sample",
    "block_match_flow", "consistency_occlusion_mask", "VideoSource",
    "make_scene", "render_scene_frame", "render_all_methods",
    "synthetic_video_source", "rmse", "contour_energy", "motion_roi",
    "static_roi",
]

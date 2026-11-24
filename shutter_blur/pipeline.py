"""High-level pipeline: build sources, run shutter integrators, collect results."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .integrators import (
    METHODS,
    SceneVisibilityProvider,
    VideoVisibilityProvider,
    ground_truth,
    render_shutter,
)
from .kernel import ShutterKernel
from .motion import FlowMotionField, SceneMotionField, block_match_flow
from .scene import SCENES, Scene


@dataclass
class FrameBundle:
    clean: np.ndarray            # sharp frame at t_center
    blurred: dict[str, np.ndarray]
    gt: np.ndarray
    paths: np.ndarray
    visibility: np.ndarray
    weight_map: np.ndarray
    t_center: float


def make_scene(name: str) -> Scene:
    if name not in SCENES:
        raise KeyError(f"unknown scene {name!r}; choose from {sorted(SCENES)}")
    return SCENES[name]()


def render_scene_frame(scene: Scene, kernel: ShutterKernel, frame_index: int,
                       method: str = "time_split", curved: bool = True,
                       gt_samples: int = 512) -> FrameBundle:
    t_center = frame_index * kernel.frame_interval
    clean, _ = scene.render(t_center)
    motion = SceneMotionField(scene, curved=curved)
    provider = SceneVisibilityProvider(scene)
    result = render_shutter(clean, motion, kernel, t_center, method, provider)
    gt = ground_truth(scene, kernel, t_center, gt_samples, motion)
    return FrameBundle(
        clean=clean, blurred={method: result.image}, gt=gt,
        paths=result.paths, visibility=result.visibility,
        weight_map=result.weight_map, t_center=t_center,
    )


def render_all_methods(scene: Scene, kernel: ShutterKernel, frame_index: int,
                       curved: bool = True,
                       gt_samples: int = 512) -> FrameBundle:
    t_center = frame_index * kernel.frame_interval
    clean, _ = scene.render(t_center)
    motion = SceneMotionField(scene, curved=curved)
    provider = SceneVisibilityProvider(scene)
    blurred = {}
    last = None
    for method in METHODS:
        last = render_shutter(clean, motion, kernel, t_center, method, provider)
        blurred[method] = last.image
    gt = ground_truth(scene, kernel, t_center, gt_samples, motion)
    return FrameBundle(
        clean=clean, blurred=blurred, gt=gt, paths=last.paths,
        visibility=last.visibility, weight_map=last.weight_map,
        t_center=t_center,
    )


# ------------------------------------------------------------- real video path
class VideoSource:
    """Frame-sampled video with block-matching flow and consistency masks."""

    def __init__(self, frames: list[np.ndarray], frame_interval: float,
                 block: int = 16, search: int = 16, stride: int = 8):
        self.frames = frames
        self.frame_interval = frame_interval
        self.flows_fwd = []
        self.flows_bwd = []
        for k in range(len(frames) - 1):
            fwd = block_match_flow(frames[k], frames[k + 1],
                                   block=block, search=search, stride=stride)
            bwd = block_match_flow(frames[k + 1], frames[k],
                                   block=block, search=search, stride=stride)
            self.flows_fwd.append(fwd)
            self.flows_bwd.append(bwd)
        # duplicate last interval so every frame has a flow pair
        self.flows_fwd.append(self.flows_fwd[-1].copy())
        self.flows_bwd.append(self.flows_bwd[-1].copy())

    @property
    def height(self):
        return self.frames[0].shape[0]

    @property
    def width(self):
        return self.frames[0].shape[1]

    def render_frame(self, kernel: ShutterKernel, frame_index: int,
                     method: str = "time_split") -> FrameBundle:
        t_center = frame_index * kernel.frame_interval
        clean = self.frames[frame_index]
        motion = FlowMotionField(
            self.flows_fwd, kernel.frame_interval, self.height, self.width)
        provider = VideoVisibilityProvider(
            self.frames, self.flows_fwd, self.flows_bwd,
            kernel.frame_interval, t_center)
        result = render_shutter(clean, motion, kernel, t_center, method,
                                provider)
        return FrameBundle(
            clean=clean, blurred={method: result.image}, gt=result.image,
            paths=result.paths, visibility=result.visibility,
            weight_map=result.weight_map, t_center=t_center,
        )


def synthetic_video_source(scene: Scene, frame_interval: float,
                           num_frames: int) -> VideoSource:
    """Treat an analytic scene like a real sampled video (for player parity)."""
    frames = [scene.render(k * frame_interval)[0] for k in range(num_frames)]
    return VideoSource(frames, frame_interval)

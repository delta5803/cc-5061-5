"""Motion fields: analytic scene fields, flow-based fields and block matching.

A ``MotionField`` answers two questions:

* ``flow_at(t)``      -> velocity (px/s) of the visible surface;
* ``backtrace(p, t0, t1)`` -> position at t1 of the surface seen at p at t0.

Analytic scene fields curve correctly through accelerations/rotations.
Flow-based fields integrate a (possibly estimated) velocity field with
straight segments, which is what a real video pipeline can do.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import Scene


def bilinear_sample(image: np.ndarray, points_xy: np.ndarray,
                    fill: float = 0.0) -> np.ndarray:
    """Sample an HxW or HxWxC image at floating pixel centres (x, y)."""
    h, w = image.shape[:2]
    x = points_xy[..., 0] - 0.5
    y = points_xy[..., 1] - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = x - x0, y - y0
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    if image.ndim == 2:
        Ia, Ib = image[y0c, x0c], image[y0c, x1c]
        Ic, Id = image[y1c, x0c], image[y1c, x1c]
        out = (Ia * (1 - wx) * (1 - wy) + Ib * wx * (1 - wy)
               + Ic * (1 - wx) * wy + Id * wx * wy)
    else:
        shape = (1,) * (image.ndim - 1)
        wx3 = wx[..., None]
        wy3 = wy[..., None]
        Ia = image[y0c, x0c]
        Ib = image[y0c, x1c]
        Ic = image[y1c, x0c]
        Id = image[y1c, x1c]
        out = (Ia * (1 - wx3) * (1 - wy3) + Ib * wx3 * (1 - wy3)
               + Ic * (1 - wx3) * wy3 + Id * wx3 * wy3)
    outside = (x0 < 0) | (x1 > w - 1) | (y0 < 0) | (y1 > h - 1)
    if image.ndim == 2:
        out[outside] = fill
    else:
        out[outside] = fill
    return out


class MotionField:
    """Interface used by every integrator."""

    def flow_at(self, t: float) -> np.ndarray:
        raise NotImplementedError

    def backtrace(self, points_xy: np.ndarray, t_from: float,
                  t_to: float) -> np.ndarray:
        raise NotImplementedError


class SceneMotionField(MotionField):
    """Exact analytic field for synthetic scenes."""

    def __init__(self, scene: Scene, curved: bool = True):
        self.scene = scene
        self.curved = curved

    def flow_at(self, t: float) -> np.ndarray:
        return self.scene.flow(t)

    def backtrace(self, points_xy, t_from, t_to):
        if self.curved:
            return self.scene.backtrace(points_xy, t_from, t_to)
        # Linear-flow baseline: a single straight displacement v * (dt).
        flow = self.flow_at(t_from)
        x = np.clip(np.round(points_xy[..., 0] - 0.5).astype(int),
                    0, self.scene.width - 1)
        y = np.clip(np.round(points_xy[..., 1] - 0.5).astype(int),
                    0, self.scene.height - 1)
        return points_xy + flow[y, x] * (t_to - t_from)


@dataclass
class FlowMotionField(MotionField):
    """Field built from discrete per-frame flows (real-video pipeline).

    ``flows[k]`` is the velocity (px/s) of frame k over [t_k, t_{k+1}].
    Backtraces are integrated with straight segments per frame interval.
    """

    flows: list[np.ndarray]
    frame_interval: float
    height: int
    width: int

    def _frame_index(self, t: float) -> int:
        return int(np.clip(np.floor(t / self.frame_interval),
                           0, len(self.flows) - 1))

    def flow_at(self, t: float) -> np.ndarray:
        return self.flows[self._frame_index(t)]

    def backtrace(self, points_xy, t_from, t_to):
        pts = points_xy.astype(np.float64).copy()
        if t_to >= t_from:
            t = t_from
            while t < t_to - 1e-12:
                step = min(self.frame_interval, t_to - t)
                pts += bilinear_sample(self.flow_at(t), pts) * step
                t += step
        else:
            t = t_from
            while t > t_to + 1e-12:
                step = min(self.frame_interval, t - t_to)
                pts -= bilinear_sample(self.flow_at(t - step), pts) * step
                t -= step
        return pts


def block_match_flow(frame_a: np.ndarray, frame_b: np.ndarray,
                     block: int = 12, search: int = 10,
                     stride: int = 6) -> np.ndarray:
    """Dense forward flow from frame_a to frame_b by exhaustive block matching.

    SAD on grayscale blocks; the flow is interpolated back to every pixel.
    Intentionally simple and dependency-free, sufficient for the player's
    real-video preview path.
    """
    if frame_a.ndim == 3:
        ga = frame_a.mean(axis=2)
        gb = frame_b.mean(axis=2)
    else:
        ga, gb = frame_a, frame_b
    h, w = ga.shape
    half = block // 2
    ys = np.arange(half, h - half, stride)
    xs = np.arange(half, w - half, stride)
    sparse = np.zeros((len(ys), len(xs), 2), dtype=np.float64)
    pad_a = np.pad(ga, half, mode="edge")
    pad_b = np.pad(gb, half + search, mode="edge")
    for i, cy in enumerate(ys):
        for j, cx in enumerate(xs):
            block_a = pad_a[cy: cy + block, cx: cx + block]
            best = np.inf
            best_dy = best_dx = 0
            for dy in range(-search, search + 1, 2):
                for dx in range(-search, search + 1, 2):
                    by = cy + dy + search
                    bx = cx + dx + search
                    block_b = pad_b[by: by + block, bx: bx + block]
                    sad = np.abs(block_a - block_b).sum()
                    if sad < best:
                        best, best_dy, best_dx = sad, dy, dx
            sparse[i, j] = [best_dx, best_dy]
    # nearest-neighbour upsample onto the full pixel grid
    flow = np.zeros((h, w, 2), dtype=np.float64)
    gy = np.clip(np.round((np.arange(h) - half) / stride).astype(int),
                 0, len(ys) - 1)
    gx = np.clip(np.round((np.arange(w) - half) / stride).astype(int),
                 0, len(xs) - 1)
    flow = sparse[np.ix_(gy, gx)]
    return flow


def consistency_occlusion_mask(flow_forward: np.ndarray,
                               flow_backward: np.ndarray,
                               threshold: float = 1.5) -> np.ndarray:
    """Boolean mask of occluded pixels via forward-backward flow consistency.

    A pixel whose forward-warped backward flow does not match its forward flow
    is marked occluded (True).  This is the visibility information available
    without depth in a real frame-sampled video.
    """
    h, w = flow_forward.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.stack([xx + 0.5 + flow_forward[..., 0],
                    yy + 0.5 + flow_forward[..., 1]], axis=-1)
    warped_back = bilinear_sample(flow_backward, pts, fill=0.0)
    residual = np.linalg.norm(flow_forward + warped_back, axis=-1)
    magnitude = np.linalg.norm(flow_forward, axis=-1)
    return residual > (threshold + 0.01 * magnitude)

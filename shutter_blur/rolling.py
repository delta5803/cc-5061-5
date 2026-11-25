"""Rolling-shutter and exposure-profile integrators.

A global shutter opens every photosite over the same exposure window.  A
rolling shutter opens each row over its own window: row ``y`` integrates over
``[t_c + r[y] - E/2, t_c + r[y] + E/2]`` where ``r[y]`` comes from
:meth:`ShutterKernel.row_offsets`.  Two effects therefore have distinct causes
and are controlled independently:

* **motion smear** - within one row, samples spread over the exposure window E;
* **progressive (rolling) distortion** - between rows, the exposure centres
  sweep across the frame over the readout duration R.

Setting R=0 reproduces a global shutter regardless of direction; setting E
small while R stays large keeps every row sharp while the geometry skews.
Every sample carries the temporal weight of the kernel's exposure profile,
which is how trapezoid / gaussian shutters differ from a box shutter.

Row times are constant across a row's width, so the hot path renders each
(row, sample) strip once with the ordinary scalar-time scene pipeline and
gathers full-width arrays; no per-pixel time maps are needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .integrators import (
    ShutterResult,
    VisibilityProvider,
    pixel_grid,
)
from .kernel import ShutterKernel
from .motion import MotionField, SceneMotionField, bilinear_sample


@dataclass
class RollingResult(ShutterResult):
    row_centers: np.ndarray = None   # (H,) exposure-centre offset per row (s)
    sample_times: np.ndarray = None  # (H, N) absolute sample times (s)
    sample_weights: np.ndarray = None  # (N,) normalized temporal weights


def row_sample_times(kernel: ShutterKernel, t_center: float,
                     height: int) -> tuple[np.ndarray, np.ndarray]:
    """Absolute sample time of every (row, sample) pair.

    Returns ``(times[H, N], row_offsets[H])`` where times[y, k] =
    t_center + row_offsets[y] + sample_offsets[k].
    """
    row_off = kernel.exposure_row_offsets(height)
    offsets = kernel.sample_offsets()
    times = t_center + row_off[:, None] + offsets[None, :]
    return times, row_off


def _row_owner_id(provider, points, t_center):
    """Centre-frame owner per pixel (H, W); scalar time so the cache is used."""
    return provider.center_owner_id(points, t_center)


def render_rolling(frame, motion: MotionField, kernel: ShutterKernel,
                   t_center: float,
                   provider: VisibilityProvider) -> RollingResult:
    """Per-row exposure integration with occlusion-aware visibility.

    The fixed-photosite observation model is identical to the
    ``visibility`` global-shutter integrator: at every sample time each pixel
    blends the centre-frame owner colour with the occlusion backdrop, weighted
    by the owner's sub-frame coverage and the exposure-profile time weight.
    The only extra degree of freedom is the per-row exposure centre.

    The scene path evaluates each (row, sample) strip once with
    :meth:`Scene.render_strip`, giving the full composite, the soft layer
    weights and the owner colour in a single narrow computation instead of
    several full-frame renders.  Non-scene providers fall back to their
    generic per-row calls.
    """
    h, w = frame.shape[:2]
    points = pixel_grid(h, w)
    offsets = kernel.sample_offsets()
    n = len(offsets)
    weights = kernel.profile_weights(offsets)
    times, row_off = row_sample_times(kernel, t_center, h)

    owner_id = _row_owner_id(provider, points, t_center)
    is_bg = (owner_id == 0)
    scene = getattr(provider, "scene", None)
    id_channels = None
    if scene is not None:
        ids = [0] + [layer.layer_id for layer in scene.layers]
        max_id = max(ids)
        id_channels = np.zeros(max_id + 1, dtype=np.int64)
        for ch, lid in enumerate(ids):
            id_channels[lid] = ch

    acc = np.zeros_like(frame, dtype=np.float64)
    vis = np.zeros((n, h, w), dtype=np.float64)
    paths = np.empty((n, h, w, 2), dtype=np.float64)

    for k, tau in enumerate(offsets):
        t_col = times[:, k]
        path_k = np.empty((h, w, 2), dtype=np.float64)
        colour_k = np.zeros((h, w, 3), dtype=np.float64)
        vis_k = np.ones((h, w), dtype=np.float64)
        # group consecutive rows sharing one instant (one group for global
        # shutter, one row per group for a fully rolling sensor)
        change = np.r_[0, np.flatnonzero(np.diff(t_col)) + 1]
        groups = [(s, change[i + 1] if i + 1 < len(change) else h)
                  for i, s in enumerate(change)]
        for y0, y1 in groups:
            t = float(t_col[y0])
            row_pts = points[y0:y1]
            row_owner = owner_id[y0:y1]
            row_bg = is_bg[y0:y1]
            t_map = np.broadcast_to(t, row_pts.shape[:-1])
            p = motion.backtrace(row_pts, t_center, t_map)
            path_k[y0:y1] = p
            if scene is not None:
                full, soft, owner_rgb = scene.render_strip(y0, y1, t)
                ch = id_channels[row_owner]
                alpha = np.take_along_axis(
                    soft, ch[..., None], axis=-1)[..., 0]
                a = alpha[..., None]
                # backdrop: full composite with the owner removed
                backdrop = np.clip(
                    (full - a * owner_rgb) / np.maximum(1.0 - a, 1e-6),
                    0.0, 1.0)
                split = a * owner_rgb + (1.0 - a) * backdrop
                colour_k[y0:y1] = np.where(row_bg[..., None], full, split)
                vis_k[y0:y1] = np.where(row_bg, 1.0, alpha)
            else:
                alpha = provider.fixed_visibility(row_pts, t, row_owner)
                owner_colour = provider.owner_color(row_pts, t, row_owner)
                backdrop = provider.background_color(
                    row_pts, t, row_owner)
                full = provider.instant_color(row_pts, t)
                a = alpha[..., None]
                split = a * owner_colour + (1.0 - a) * backdrop
                colour_k[y0:y1] = np.where(row_bg[..., None], full, split)
                vis_k[y0:y1] = np.where(row_bg, 1.0, alpha)
        paths[k] = path_k
        acc += weights[k] * colour_k
        vis[k] = vis_k

    image = np.clip(acc, 0.0, 1.0)
    wsum = np.ones((h, w), dtype=np.float64)
    return RollingResult(
        image=image, weight_map=wsum, paths=paths, visibility=vis,
        method="rolling", row_centers=row_off, sample_times=times,
        sample_weights=weights,
    )


def rolling_ground_truth(scene, kernel: ShutterKernel, t_center: float,
                         gt_samples: int = 96) -> np.ndarray:
    """High-sample physical reference using the exact per-row timing.

    Every row integrates the *full fixed-photosite composite* over its own
    exposure window ``[t_c + r[y] - E/2, t_c + r[y] + E/2]`` with the kernel's
    exposure-profile weights::

        pixel(y, x) = sum_k w_k * F(t_y + tau_k)[y, x]

    where ``F`` is the analytic anti-aliased scene composite.  This is the
    same fixed-photosite observation model used by :func:`render_rolling`
    (which additionally separates the centre-frame owner from its backdrop),
    so the integrator converges here as its sample count grows.  Each row
    time is constant across the row width, so the work is a sequence of
    narrow :meth:`Scene.render_strip` evaluations.
    """
    h, w = scene.height, scene.width
    half = kernel.exposure / 2.0
    offsets = np.linspace(-half, half, gt_samples)
    gw = kernel.profile_weights(offsets)
    row_off = kernel.exposure_row_offsets(h)
    acc = np.zeros((h, w, 3), dtype=np.float64)
    for y in range(h):
        row_acc = np.zeros((1, w, 3), dtype=np.float64)
        for k, tau in enumerate(offsets):
            t = float(t_center + row_off[y] + tau)
            rgb, _, _ = scene.render_strip(y, y + 1, t)
            row_acc += gw[k] * rgb
        acc[y] = row_acc[0]
    return np.clip(acc, 0.0, 1.0)


def rolling_ground_truth_fast(scene, kernel: ShutterKernel, t_center: float,
                              gt_samples: int = 96) -> np.ndarray:
    """Alias kept for callers that asked for the row-wise reference."""
    return rolling_ground_truth(scene, kernel, t_center, gt_samples)

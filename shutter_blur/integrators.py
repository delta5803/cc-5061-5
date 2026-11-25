"""Shutter integrators.

Four ways of turning a motion field and a :class:`ShutterKernel` into an
exposure-integrated frame, plus a deliberately direction-blind Gaussian
baseline used by the tests to show what motion blur is *not*.

The four methods differ only in how occlusion/visibility is handled:

naive       average colours along the whole motion path, ignoring visibility;
center_depth  keep only samples still inside the centre-frame layer mask
            (a depth decision taken once, at the frame centre);
visibility  per-sample visibility along the path, occluded samples take the
            instantaneous composite colour of the occluding surface;
time_split  split the exposure into visible/occluded time intervals and
            integrate each interval over its true visible duration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernel import ShutterKernel
from .motion import MotionField, SceneMotionField, bilinear_sample, consistency_occlusion_mask


# --------------------------------------------------------------------- helpers
def compute_paths(motion: MotionField, points_xy: np.ndarray,
                  t_center: float, offsets: np.ndarray) -> np.ndarray:
    """Path points P(t_center + tau) for every pixel and every shutter sample."""
    n = len(offsets)
    paths = np.empty((n,) + points_xy.shape, dtype=np.float64)
    for k, tau in enumerate(offsets):
        paths[k] = motion.backtrace(points_xy, t_center, t_center + tau)
    return paths


def pixel_grid(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([xx + 0.5, yy + 0.5], axis=-1).astype(np.float64)


# ----------------------------------------------------------------- providers
class VisibilityProvider:
    """Supplies sub-frame visibility and instantaneous composite colour."""

    def center_owner_id(self, points_xy, t_center) -> np.ndarray:
        """Dominant surface id at every pixel for the centre frame."""
        raise NotImplementedError

    def path_visibility(self, path_points, t, owner_id) -> np.ndarray:
        """Owner surface coverage (0..1) at sub-pixel path points at time t."""
        raise NotImplementedError

    def fixed_visibility(self, points_xy, t, owner_id) -> np.ndarray:
        """Owner coverage at the fixed photosite at time t."""
        raise NotImplementedError

    def visibility_stack(self, points_xy, t_center, offsets,
                         path_points=None) -> np.ndarray:
        raise NotImplementedError

    def instant_color(self, points_xy, t) -> np.ndarray:
        raise NotImplementedError

    def owner_color(self, path_points, t, owner_id) -> np.ndarray:
        raise NotImplementedError

    def background_color(self, points_xy, t, owner_id) -> np.ndarray:
        """Composite with the owner surface removed (the occlusion backdrop)."""
        raise NotImplementedError

    def path_color(self, points_xy, t) -> np.ndarray:
        """Colour of the centre-frame surface at a sub-pixel path point."""
        return self.instant_color(points_xy, t)


class SceneVisibilityProvider(VisibilityProvider):
    """Exact analytic provider for layered synthetic scenes."""

    def __init__(self, scene):
        self.scene = scene
        self._id_cache: dict[float, np.ndarray] = {}
        self._rgb_cache: dict[float, np.ndarray] = {}
        self._soft_cache: dict[float, np.ndarray] = {}
        self._limit = 32

    def _trim(self, cache):
        if len(cache) > self._limit:
            for key in list(cache)[: len(cache) - self._limit]:
                del cache[key]

    def _id_map(self, t):
        key = round(float(t), 9)
        if key not in self._id_cache:
            rgb, ids = self.scene.render(t)
            self._id_cache[key] = ids
            self._rgb_cache[key] = rgb
            self._trim(self._id_cache)
            self._trim(self._rgb_cache)
        return self._id_cache[key]

    def _rgb(self, t):
        self._id_map(t)
        return self._rgb_cache[round(float(t), 9)]

    def _soft(self, t):
        key = round(float(t), 9)
        if key not in self._soft_cache:
            self._soft_cache[key] = self.scene.soft_id_stack(t)
        return self._soft_cache[key]

    def center_owner_id(self, points_xy, t_center):
        return self._id_map(t_center)

    def path_visibility(self, path_points, t, owner_id):
        return self.scene.owner_alpha_at(t, path_points, owner_id)

    def fixed_visibility(self, points_xy, t, owner_id):
        return self.scene.owner_alpha_at(t, points_xy, owner_id)

    def visibility_stack(self, points_xy, t_center, offsets,
                         path_points=None):
        """Fixed-photosite owner coverage over the shutter (overlay/UI use)."""
        h, w = self.scene.height, self.scene.width
        ids_center = self._id_map(t_center)
        xi = np.clip(np.round(points_xy[..., 0] - 0.5).astype(int), 0, w - 1)
        yi = np.clip(np.round(points_xy[..., 1] - 0.5).astype(int), 0, h - 1)
        owner = ids_center[yi, xi]
        id_to_channel = {lid: k for k, lid in
                         enumerate([0] + [layer.layer_id
                                          for layer in self.scene.layers])}
        max_id = max(id_to_channel)
        lookup = np.zeros(max_id + 1, dtype=np.int64)
        for lid, ch in id_to_channel.items():
            lookup[lid] = ch
        owner_channel = lookup[owner]
        vis = np.zeros((len(offsets), h, w), dtype=np.float64)
        for k, tau in enumerate(offsets):
            soft = self.scene.soft_id_stack(t_center + tau, points_xy)
            vis[k] = soft[yi, xi, owner_channel]
        return vis

    def instant_color(self, points_xy, t):
        """Analytic composite at pixel centres (no extra spatial filtering).

        ``t`` may be a scalar or a per-pixel time map (rolling shutter); the
        latter evaluates each distinct row time once and gathers rows.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        if t_arr.ndim == 0:
            rgb = self._rgb(float(t_arr))
            h, w = rgb.shape[:2]
            xi = np.clip(np.round(points_xy[..., 0] - 0.5).astype(int), 0, w - 1)
            yi = np.clip(np.round(points_xy[..., 1] - 0.5).astype(int), 0, h - 1)
            return rgb[yi, xi]
        h, w = points_xy.shape[:2]
        out = np.zeros((h, w, 3), dtype=np.float64)
        for tv in np.unique(t_arr):
            rgb = self._rgb(float(tv))
            rows = np.unique(np.nonzero(t_arr == tv)[0])
            out[rows] = rgb[rows]
        return out

    def owner_color(self, points_xy, t, owner_layer_id):
        """Intrinsic colour of the owner layer at fixed photosite points.

        No compositing is applied: a silhouette cell keeps the layer's own
        colour while its coverage fraction is carried separately by the
        visibility weight.  Background owners read the static background.
        Accepts either a scalar time or a per-pixel time map (rolling shutter).
        """
        from .scene import _pattern_texture

        out = np.zeros(points_xy.shape[:-1] + (3,), dtype=np.float64)
        bg_mask = owner_layer_id == 0
        if bg_mask.any():
            x = points_xy[..., 0]
            y = points_xy[..., 1]
            out[bg_mask] = _pattern_texture(
                self.scene.bg_pattern, x, y, self.scene.bg_color)[bg_mask]
        t_arr = np.asarray(t, dtype=np.float64)
        hooks = getattr(self.scene, "_composite_hooks", {})
        for layer in self.scene.layers:
            m = owner_layer_id == layer.layer_id
            if not m.any():
                continue
            if layer.layer_id in hooks:
                # composite-hook layers (e.g. the dial) own their own colour
                black = np.zeros(points_xy.shape[:-1] + (3,),
                                 dtype=np.float64)
                if t_arr.ndim == 0:
                    local = self.scene._local_coords(float(t_arr), layer,
                                                      points_xy[m])
                    out[m], _ = hooks[layer.layer_id](local, black[m])
                else:
                    local_full = self.scene._local_coords(t_arr, layer,
                                                          points_xy)
                    rgb_full, _ = hooks[layer.layer_id](local_full, black)
                    out[m] = rgb_full[m]
                continue
            if t_arr.ndim == 0:
                local = self.scene._local_coords(float(t_arr), layer,
                                                  points_xy[m])
                out[m] = _pattern_texture(layer.pattern, local[..., 0],
                                          local[..., 1], layer.color)
            else:
                # per-pixel times: evaluate local coordinates on the full grid
                # (masked points with mismatched time shapes cannot share a
                # call) and keep only this owner's pixels
                local_full = self.scene._local_coords(t_arr, layer, points_xy)
                tex = _pattern_texture(layer.pattern, local_full[..., 0],
                                       local_full[..., 1], layer.color)
                out[m] = tex[m]
        return out

    def background_color(self, points_xy, t, owner_layer_id):
        """Backdrop at a fixed point: full composite with the owner removed.

        Derived algebraically from the rendered composite ``F``, the owner's
        intrinsic colour ``C`` and its effective composite weight ``w``::

            F = w*C + (1-w)*B   ->   B = (F - w*C) / (1-w)

        When the owner is fully occluded or absent (w -> 0) this reduces to
        the full composite, so every other layer - including layers above or
        below the owner - is preserved without special-casing the stack.
        """
        full = self.instant_color(points_xy, t)
        owner = self.owner_color(points_xy, t, owner_layer_id)
        weight = self.fixed_visibility(points_xy, t, owner_layer_id)[..., None]
        backdrop = (full - weight * owner) / np.maximum(1.0 - weight, 1e-6)
        return np.clip(backdrop, 0.0, 1.0)

    def path_color(self, points_xy, t):
        """Full composite colour sampled at sub-pixel path coordinates."""
        rgb, _ = self.scene.render(float(t))
        return bilinear_sample(rgb, points_xy, fill=0.0)


class VideoVisibilityProvider(VisibilityProvider):
    """Approximate provider for real, frame-sampled video.

    Visibility comes from a forward/backward consistency occlusion mask at the
    centre frame, propagated along motion paths; sub-frame colour comes from
    flow-based interpolation of the two bracketing frames.
    """

    def __init__(self, frames, flows_fwd, flows_bwd, frame_interval, t_center):
        self.frames = frames
        self.flows_fwd = flows_fwd
        self.flows_bwd = flows_bwd
        self.dt = frame_interval
        self.t_center = t_center
        k0 = int(np.floor(t_center / frame_interval))
        k0 = int(np.clip(k0, 0, len(flows_fwd) - 1))
        self.center_vis = (~consistency_occlusion_mask(
            flows_fwd[k0], flows_bwd[k0])).astype(np.float64)

    def _frame(self, k):
        k = int(np.clip(k, 0, len(self.frames) - 1))
        return self.frames[k].astype(np.float64)

    def center_owner_id(self, points_xy, t_center):
        h, w = self.center_vis.shape
        return np.ones((h, w), dtype=np.int64)  # single-surface video model

    def path_visibility(self, path_points, t, owner_id):
        return bilinear_sample(self.center_vis[..., None],
                               path_points)[..., 0]

    def fixed_visibility(self, points_xy, t, owner_id):
        return bilinear_sample(self.center_vis[..., None],
                               points_xy)[..., 0]

    def owner_color(self, path_points, t, owner_id):
        return self.instant_color(path_points, t)

    def background_color(self, points_xy, t, owner_id):
        # No layer decomposition for real video: the flow-interpolated frame
        # at the fixed photosite is the best available backdrop estimate.
        return self.instant_color(points_xy, t)

    def instant_color(self, points_xy, t):
        kf = t / self.dt
        k0 = int(np.floor(kf))
        alpha = float(np.clip(kf - k0, 0.0, 1.0))
        if k0 >= len(self.frames) - 1:
            return bilinear_sample(self._frame(len(self.frames) - 1), points_xy)
        f0 = self._frame(k0)
        f1 = self._frame(k0 + 1)
        pts0 = points_xy - self.flows_fwd[k0] * alpha * self.dt
        pts1 = points_xy + self.flows_bwd[k0] * (1 - alpha) * self.dt
        c0 = bilinear_sample(f0, pts0)
        c1 = bilinear_sample(f1, pts1)
        return (1 - alpha) * c0 + alpha * c1

    def visibility_stack(self, points_xy, t_center, offsets,
                         path_points=None):
        h, w = self.center_vis.shape
        vis = np.empty((len(offsets), h, w), dtype=np.float64)
        for k in range(len(offsets)):
            pts = path_points[k] if path_points is not None else points_xy
            vis[k] = bilinear_sample(self.center_vis[..., None],
                                     pts)[..., 0]
        return vis

    def path_color(self, points_xy, t):
        return self.instant_color(points_xy, t)


# ---------------------------------------------------------------- integrators
@dataclass
class ShutterResult:
    image: np.ndarray            # blurred RGB in [0, 1]
    weight_map: np.ndarray       # effective normalization weight per pixel
    paths: np.ndarray            # (N, H, W, 2) sampled path coordinates
    visibility: np.ndarray       # (N, H, W) sub-frame visibility
    method: str


def _gather(image, pts):
    return bilinear_sample(image, pts, fill=0.0)


def integrate_naive(frame, motion, kernel, t_center, points, paths,
                    provider=None):
    """Average colour along the whole motion path, ignoring visibility.

    Deliberately simplistic "draw one line and average colours" estimator.
    Under occlusion it drags the centre-frame surface through the occluder and
    the occluder through the background, producing ghosts; the benchmark test
    asserts this method loses to the visibility-aware methods.
    """
    weights = kernel.weights()
    offsets = kernel.sample_offsets()
    acc = np.zeros_like(frame, dtype=np.float64)
    wsum = np.zeros(frame.shape[:2], dtype=np.float64)
    for k, w in enumerate(weights):
        colour = (provider.path_color(paths[k], t_center + offsets[k])
                  if provider is not None else _gather(frame, paths[k]))
        acc += w * colour
        wsum += w
    return acc, wsum


def integrate_center_depth(frame, motion, kernel, t_center, points, paths,
                           center_id_map=None, provider=None):
    """Visibility decided once, at the frame centre, via the layer mask."""
    weights = kernel.weights()
    h, w = frame.shape[:2]
    if center_id_map is None:
        center_id_map = _center_owner_map(frame, points)
    xi = np.clip(np.round(points[..., 0] - 0.5).astype(int), 0, w - 1)
    yi = np.clip(np.round(points[..., 1] - 0.5).astype(int), 0, h - 1)
    owner = center_id_map[yi, xi]
    acc = np.zeros_like(frame, dtype=np.float64)
    wsum = np.zeros((h, w), dtype=np.float64)
    for k, wgt in enumerate(weights):
        p = paths[k]
        xj = np.clip(np.round(p[..., 0] - 0.5).astype(int), 0, w - 1)
        yj = np.clip(np.round(p[..., 1] - 0.5).astype(int), 0, h - 1)
        inside = (p[..., 0] >= 0) & (p[..., 0] < w) & \
                 (p[..., 1] >= 0) & (p[..., 1] < h)
        keep = inside & (owner == center_id_map[yj, xj])
        colour = (provider.path_color(p, t_center + _tau(kernel, k))
                  if provider is not None else _gather(frame, p))
        wk = (wgt * keep)[..., None]
        acc += wk * colour
        wsum += wgt * keep
    wsum_safe = np.maximum(wsum, 1e-12)
    return acc / wsum_safe[..., None], wsum


def _tau(kernel, k):
    return kernel.sample_offsets()[k]


def _center_owner_map(frame, points):
    """Approximate layer ownership at the frame centre from colour identity.

    For scene inputs the provider's exact id map is preferred; this fallback
    groups pixels by quantised colour so the video path still has a mask.
    """
    q = (frame * 32).astype(np.int64)
    return q[..., 0] * 1024 + q[..., 1] * 32 + q[..., 2]


def integrate_visibility(frame, motion, kernel, t_center, points, paths,
                         provider: VisibilityProvider):
    """Integrate visibility *along the pixel motion path* (method two).

    Every shutter time contributes, at the fixed photosite, the owner
    surface's colour weighted by its sub-frame coverage plus the occlusion
    backdrop for the remaining fraction.  Visibility is therefore integrated
    as a per-sample weight rather than averaged blindly along a line.  The
    motion field still drives where each surface came from (the path and, for
    real video, the flow interpolation); analytic scenes render each term
    exactly, so this converges to the physical ground truth.
    """
    weights = kernel.weights()
    offsets = kernel.sample_offsets()
    h, w = frame.shape[:2]
    owner_id = provider.center_owner_id(points, t_center)
    vis = np.zeros((len(offsets), h, w), dtype=np.float64)
    acc = np.zeros_like(frame, dtype=np.float64)
    wsum = np.zeros((h, w), dtype=np.float64)
    for k, wgt in enumerate(weights):
        t = t_center + offsets[k]
        alpha = provider.fixed_visibility(points, t, owner_id)
        owner_colour = provider.owner_color(points, t, owner_id)
        backdrop = provider.background_color(points, t, owner_id)
        full = provider.instant_color(points, t)
        a = alpha[..., None]
        is_bg = (owner_id == 0)[..., None]
        split = a * owner_colour + (1.0 - a) * backdrop
        acc += wgt * np.where(is_bg, full, split)
        wsum += wgt
        vis[k] = np.where(owner_id == 0, 1.0, alpha)
    return acc, wsum, vis


def integrate_time_split(frame, motion, kernel, t_center, points, paths,
                         provider: VisibilityProvider, grid_steps: int = 64):
    """Split the exposure into visibility segments (method one).

    A fine temporal grid classifies each cell at the fixed photosite as
    visible (owner covers the cell) or occluded.  Visible cells integrate the
    owner colour, occluded cells the backdrop, each weighted by the cell's
    true duration.  The fine grid resolves the exact entry/exit times of an
    occluding layer, so visible and occluded durations are integrated
    separately while the full exposure still normalises to one.
    """
    half = kernel.exposure / 2.0
    fine = np.linspace(-half, half, grid_steps)
    h, w = frame.shape[:2]
    owner_id = provider.center_owner_id(points, t_center)
    fine_vis = np.zeros((grid_steps, h, w), dtype=np.float64)
    acc = np.zeros((h, w, 3), dtype=np.float64)
    for g, tau in enumerate(fine):
        t = t_center + tau
        alpha = provider.fixed_visibility(points, t, owner_id)
        owner_colour = provider.owner_color(points, t, owner_id)
        backdrop = provider.background_color(points, t, owner_id)
        full = provider.instant_color(points, t)
        a = alpha[..., None]
        is_bg = (owner_id == 0)[..., None]
        split = a * owner_colour + (1.0 - a) * backdrop
        acc += np.where(is_bg, full, split)
        fine_vis[g] = np.where(owner_id == 0, 1.0, alpha)
    image = acc / grid_steps
    wsum = np.ones((h, w), dtype=np.float64)
    coarse_vis = np.stack([
        provider.fixed_visibility(points, t_center + tau, owner_id)
        for tau in kernel.sample_offsets()
    ], axis=0)
    return image, wsum, coarse_vis


def segment_count_map(provider: VisibilityProvider, points, t_center: float,
                      kernel: ShutterKernel, motion=None,
                      grid_steps: int = 64) -> np.ndarray:
    """Number of visibility segments (visible/occluded runs) per pixel."""
    half = kernel.exposure / 2.0
    fine = np.linspace(-half, half, grid_steps)
    owner_id = provider.center_owner_id(points, t_center)
    vis = np.stack([
        provider.fixed_visibility(points, t_center + fine[g], owner_id)
        for g in range(grid_steps)
    ], axis=0)
    state = vis > 0.5
    flips = np.zeros_like(state)
    flips[1:] = state[1:] != state[:-1]
    return 1 + flips.sum(axis=0)


# ------------------------------------------------------------ gaussian baseline
def gaussian_blur(frame: np.ndarray, sigma: float) -> np.ndarray:
    """Direction-blind, time-independent blur (explicit anti-baseline)."""
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    def conv1d(img, k, axis):
        pad_width = [(0, 0)] * img.ndim
        pad_width[axis] = (len(k) // 2, len(k) // 2)
        padded = np.pad(img, pad_width)
        out = np.zeros_like(img)
        for i, kv in enumerate(k):
            window = np.take(padded,
                             np.arange(i, i + img.shape[axis]), axis=axis)
            out += kv * window
        return out

    return conv1d(conv1d(frame, kernel, 0), kernel, 1)


# ------------------------------------------------------------------- entry API
METHODS = ("naive", "center_depth", "visibility", "time_split")


def render_shutter(frame, motion: MotionField, kernel: ShutterKernel,
                   t_center: float, method: str = "time_split",
                   provider: VisibilityProvider | None = None,
                   center_id_map: np.ndarray | None = None) -> ShutterResult:
    if method not in METHODS:
        raise ValueError(f"unknown integrator {method!r}, choose from {METHODS}")
    h, w = frame.shape[:2]
    points = pixel_grid(h, w)
    offsets = kernel.sample_offsets()
    paths = compute_paths(motion, points, t_center, offsets)

    if method == "naive":
        image, wsum = integrate_naive(frame, motion, kernel,
                                      t_center, points, paths, provider)
        vis = np.ones((len(offsets), h, w))
    elif method == "center_depth":
        id_map = None
        if provider is not None and hasattr(provider, "_id_map"):
            id_map = provider._id_map(t_center)
        image, wsum = integrate_center_depth(frame, motion, kernel,
                                             t_center, points, paths,
                                             id_map, provider)
        vis = np.ones((len(offsets), h, w))
    elif method == "visibility":
        if provider is None:
            raise ValueError("visibility integrator needs a VisibilityProvider")
        image, wsum, vis = integrate_visibility(
            frame, motion, kernel, t_center, points, paths, provider)
    else:
        if provider is None:
            raise ValueError("time_split integrator needs a VisibilityProvider")
        image, wsum, vis = integrate_time_split(
            frame, motion, kernel, t_center, points, paths, provider)
    image = np.clip(image, 0.0, 1.0)
    return ShutterResult(image=image, weight_map=wsum, paths=paths,
                         visibility=vis, method=method)


def ground_truth(scene, kernel: ShutterKernel, t_center: float,
                 gt_samples: int = 512, motion=None) -> np.ndarray:
    """High-sample truth on the same sub-pixel observation model.

    For every shutter time tau the analytic scene is rendered and bilinearly
    sampled at the backtraced path position P(t_center+tau).  Sampling the
    time-varying scene along the motion path is physically identical to
    integrating light at a fixed photosite, and it shares the exact spatial
    observation model of the path integrators, so their RMSE converges to zero
    as the number of shutter samples approaches ``gt_samples``.  Samples are
    weighted by the kernel's exposure profile (equal weights for a box
    shutter), so non-box shutters have their own matching reference.
    """
    h, w = scene.height, scene.width
    points = pixel_grid(h, w)
    offsets = np.linspace(-kernel.exposure / 2, kernel.exposure / 2, gt_samples)
    weights = kernel.profile_weights(offsets)
    if motion is None:
        motion = SceneMotionField(scene, curved=True)
    acc = np.zeros((h, w, 3), dtype=np.float64)
    for k, tau in enumerate(offsets):
        path_pts = motion.backtrace(points, t_center, t_center + tau)
        rgb = scene.render(t_center + tau)[0]
        acc += weights[k] * bilinear_sample(rgb, path_pts, fill=0.0)
    return np.clip(acc, 0.0, 1.0)

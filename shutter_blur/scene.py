"""Layered synthetic scenes with analytic, continuously defined motion.

A scene is a stack of layers ordered back-to-front.  Every layer has a shape,
a texture and a rigid/linear motion model.  All rendering is defined for *any*
continuous time t, which is what lets us render both the high-sample ground
truth and frame-sampled inputs from the same underlying world.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BG_LAYER_ID = 0


def _pattern_texture(pattern: str, x: np.ndarray, y: np.ndarray,
                     color: tuple[float, float, float]) -> np.ndarray:
    """RGB texture in [0, 1] evaluated at image coordinates x, y."""
    out = np.zeros(x.shape + (3,), dtype=np.float64)
    if pattern == "solid":
        out[..., :] = color
    elif pattern == "checker":
        cell = 18.0
        m = ((np.floor(x / cell) + np.floor(y / cell)).astype(np.int64) % 2) == 0
        out[~m] = color
        out[m] = [min(1.0, c + 0.55) for c in color]
    elif pattern == "stripes":
        m = (np.floor(x / 10.0).astype(np.int64) % 2) == 0
        out[~m] = color
        out[m] = [min(1.0, c + 0.5) for c in color]
    elif pattern == "gradient":
        g = np.clip(0.25 + 0.65 * (x / x.shape[1] if x.ndim == 2 else x / 256.0),
                    0.0, 1.0)
        for c in range(3):
            out[..., c] = g * color[c] + (1 - g) * 0.12
    else:
        raise ValueError(f"unknown texture pattern {pattern!r}")
    return out


@dataclass
class Layer:
    layer_id: int
    kind: str                      # "rect" | "disc"
    size: tuple[float, float]      # rect: (w, h); disc: (radius, ignored)
    color: tuple[float, float, float]
    pattern: str = "solid"
    # Motion: p(t) = p0 + v*t + 0.5*a*t^2, plus angular velocity omega (rad/s)
    # and an optional sinusoidal oscillation A*sin(2*pi*f*t + phase) on top,
    # which gives genuinely non-uniform velocity (curved exposure paths).
    p0: tuple[float, float] = (0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)
    accel: tuple[float, float] = (0.0, 0.0)
    omega: float = 0.0
    osc_amp: tuple[float, float] = (0.0, 0.0)
    osc_freq: float = 0.0
    osc_phase: float = 0.0

    def center(self, t) -> np.ndarray:
        t_arr = np.asarray(t, dtype=np.float64)
        osc = 0.0
        if self.osc_freq:
            osc = np.sin(2.0 * np.pi * self.osc_freq * t_arr
                         + self.osc_phase)
        x = (self.p0[0] + self.velocity[0] * t_arr
             + 0.5 * self.accel[0] * t_arr * t_arr
             + self.osc_amp[0] * osc)
        y = (self.p0[1] + self.velocity[1] * t_arr
             + 0.5 * self.accel[1] * t_arr * t_arr
             + self.osc_amp[1] * osc)
        if t_arr.ndim == 0:
            return np.array([float(x), float(y)], dtype=np.float64)
        return np.stack([x, y], axis=-1)

    def velocity_at(self, t: float) -> np.ndarray:
        osc_d = 0.0
        if self.osc_freq:
            osc_d = (2.0 * np.pi * self.osc_freq
                     * np.cos(2.0 * np.pi * self.osc_freq * t
                              + self.osc_phase))
        return np.array(
            [self.velocity[0] + self.accel[0] * t + self.osc_amp[0] * osc_d,
             self.velocity[1] + self.accel[1] * t + self.osc_amp[1] * osc_d],
            dtype=np.float64,
        )

    def contains(self, local_xy: np.ndarray) -> np.ndarray:
        x = local_xy[..., 0]
        y = local_xy[..., 1]
        if self.kind == "rect":
            w, h = self.size
            return (np.abs(x) <= w / 2.0) & (np.abs(y) <= h / 2.0)
        if self.kind == "vstrip":
            # A tall, very narrow vertical bar: width drives the smear width.
            w, h = self.size
            return (np.abs(x) <= w / 2.0) & (np.abs(y) <= h / 2.0)
        if self.kind == "disc":
            return x * x + y * y <= self.size[0] ** 2
        raise ValueError(self.kind)


@dataclass
class Scene:
    height: int
    width: int
    layers: list[Layer] = field(default_factory=list)
    bg_pattern: str = "checker"
    bg_color: tuple[float, float, float] = (0.18, 0.22, 0.30)

    def __post_init__(self) -> None:
        self._render_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self._alpha_cache: dict[float, np.ndarray] = {}
        self._soft_cache: dict[float, np.ndarray] = {}
        # rolling-shutter passes touch one distinct time per row per sample,
        # so the cache must stay small to bound peak memory
        self._cache_limit = 32
        # optional per-layer custom composite hook: layer_id -> fn(local, rgb)
        self._composite_hooks: dict = {}

    def _trim(self, cache: dict) -> None:
        if len(cache) > self._cache_limit:
            for key in list(cache)[: len(cache) - self._cache_limit]:
                del cache[key]

    def clear_cache(self) -> None:
        self._render_cache.clear()
        self._alpha_cache.clear()
        self._soft_cache.clear()

    # ------------------------------------------------------------------ grids
    def pixel_grid(self) -> tuple[np.ndarray, np.ndarray]:
        yy, xx = np.mgrid[0: self.height, 0: self.width]
        return xx.astype(np.float64) + 0.5, yy.astype(np.float64) + 0.5

    # --------------------------------------------------------------- rendering
    def _local_coords(self, t, layer: Layer,
                      points_xy: np.ndarray | None = None) -> np.ndarray:
        if points_xy is None:
            x, y = self.pixel_grid()
            pts = np.stack([x, y], axis=-1)
        else:
            pts = points_xy
        centre = np.asarray(layer.center(float(t) if np.ndim(t) == 0
                                         else t), dtype=np.float64)
        angle = layer.omega * np.asarray(t, dtype=np.float64)
        ca, sa = np.cos(angle), np.sin(angle)
        # centre is (2,) for a scalar time or (H, W, 2) for per-pixel times
        delta = pts - centre
        lx = ca * delta[..., 0] + sa * delta[..., 1]
        ly = -sa * delta[..., 0] + ca * delta[..., 1]
        return np.stack([lx, ly], axis=-1)

    def _alpha_at(self, layer: Layer, local_xy: np.ndarray) -> np.ndarray:
        """Analytic fraction of a unit photosite centred at local_xy covered."""
        x, y = local_xy[..., 0], local_xy[..., 1]
        if layer.kind in ("rect", "vstrip"):
            w, h = layer.size
            ax = _edge_coverage(x, w / 2.0)
            ay = _edge_coverage(y, h / 2.0)
            return np.clip(ax * ay, 0.0, 1.0)
        r = layer.size[0]
        return _disc_coverage(x, y, r)

    def layer_alpha(self, t: float,
                    points_xy: np.ndarray | None = None) -> np.ndarray:
        """Analytic area coverage alpha of every layer (L, ...) in [0, 1]."""
        if points_xy is None:
            key = round(float(t), 9)
            if key not in self._alpha_cache:
                self._alpha_cache[key] = self._compute_layer_alpha(t, None)
                self._trim(self._alpha_cache)
            return self._alpha_cache[key]
        return self._compute_layer_alpha(t, points_xy)

    def _compute_layer_alpha(self, t, points_xy):
        alphas = []
        for layer in self.layers:
            local = self._local_coords(t, layer, points_xy)
            alphas.append(self._alpha_at(layer, local))
        return np.stack(alphas, axis=0)

    def owner_alpha_at(self, t: float, points_xy: np.ndarray,
                       owner_layer_id: np.ndarray) -> np.ndarray:
        """Effective visible weight of the centre-frame owner at time t.

        For a foreground owner this is its layer weight in the full
        back-to-front composite at the fixed photosite, already discounted by
        any higher layer covering it; a background owner (id 0) returns 1 and
        receives the full composite through its backdrop term.
        """
        weights = self.soft_id_stack(t, points_xy)
        id_to_channel = {lid: k for k, lid in
                         enumerate([BG_LAYER_ID]
                                   + [layer.layer_id for layer in self.layers])}
        max_id = max(id_to_channel)
        lookup = np.zeros(max_id + 1, dtype=np.int64)
        for lid, ch in id_to_channel.items():
            lookup[lid] = ch
        channels = lookup[owner_layer_id]
        effective = np.take_along_axis(
            weights, channels[..., None], axis=-1)[..., 0]
        return np.where(owner_layer_id == BG_LAYER_ID, 1.0, effective)

    def render(self, t: float):
        """Anti-aliased instantaneous composite at time t.

        Returns (rgb in [0,1], hard id map).  Layer colours are alpha
        composited back-to-front with analytic area coverage.
        """
        key = round(float(t), 9)
        if key in self._render_cache:
            return self._render_cache[key]
        x, y = self.pixel_grid()
        pts = np.stack([x, y], axis=-1)
        rgb = _pattern_texture(self.bg_pattern, x, y, self.bg_color)
        ids = np.zeros((self.height, self.width), dtype=np.int64)
        for layer in self.layers:
            local = self._local_coords(t, layer, pts)
            if layer.layer_id in self._composite_hooks:
                rgb, alpha = self._composite_hooks[layer.layer_id](local, rgb)
            else:
                alpha = self._alpha_at(layer, local)
                tex = _pattern_texture(layer.pattern, local[..., 0],
                                       local[..., 1], layer.color)
                a = alpha[..., None]
                rgb = tex * a + rgb * (1.0 - a)
            ids[alpha > 0.5] = layer.layer_id
        result = (rgb, ids)
        self._render_cache[key] = result
        self._trim(self._render_cache)
        return result

    def id_map(self, t: float) -> np.ndarray:
        return self.render(t)[1]

    def render_strip(self, y0: int, y1: int, t: float):
        """Composite, soft weights and owner colour for rows [y0, y1) only.

        Scalar time and a narrow strip: this is the rolling-shutter hot path,
        where rendering whole frames for every row would dominate the cost.
        Returns ``(rgb, soft_weights, owner_rgb)`` with ``owner_rgb`` holding
        each layer's intrinsic (un-composited) colour stacked per channel.
        """
        yy, xx = np.mgrid[y0:y1, 0: self.width]
        pts = np.stack([xx.astype(np.float64) + 0.5,
                        yy.astype(np.float64) + 0.5], axis=-1)
        rgb = _pattern_texture(self.bg_pattern, pts[..., 0], pts[..., 1],
                               self.bg_color)
        n_layer = len(self.layers)
        weights = np.zeros(pts.shape[:-1] + (n_layer + 1,),
                           dtype=np.float64)
        owner_rgb = np.zeros(pts.shape[:-1] + (3,), dtype=np.float64)
        transmittance = np.ones(pts.shape[:-1], dtype=np.float64)
        for k, layer in enumerate(self.layers):
            local = self._local_coords(float(t), layer, pts)
            if layer.layer_id in self._composite_hooks:
                rgb, alpha = self._composite_hooks[layer.layer_id](local, rgb)
            else:
                alpha = self._alpha_at(layer, local)
                tex = _pattern_texture(layer.pattern, local[..., 0],
                                       local[..., 1], layer.color)
                a = alpha[..., None]
                rgb = tex * a + rgb * (1.0 - a)
            weights[..., k + 1] = alpha * transmittance
            transmittance = transmittance * (1.0 - alpha)
            owner_rgb = np.where((alpha > 0.5)[..., None], rgb, owner_rgb)
        weights[..., BG_LAYER_ID] = transmittance
        bg_pixels = weights[..., BG_LAYER_ID] > 0.5
        owner_rgb[bg_pixels] = rgb[bg_pixels]
        return rgb, weights, owner_rgb

    def soft_id_stack(self, t: float,
                      points_xy: np.ndarray | None = None) -> np.ndarray:
        """Soft id weights (..., L+1) under standard Porter-Duff ``over``.

        Layers composite back-to-front with
        ``out = alpha_front * colour_front + (1 - alpha_front) * out``.
        Each layer's stored weight is its own alpha multiplied by the
        transmittance accumulated from the layers *above* it in the final
        ordering, so an opaque foreground correctly occludes an opaque
        background layer.  Channel 0 is the background.
        """
        if points_xy is None:
            key = round(float(t), 9)
            if key not in self._soft_cache:
                self._soft_cache[key] = self._compute_soft(t, None)
                self._trim(self._soft_cache)
            return self._soft_cache[key]
        return self._compute_soft(t, points_xy)

    def _compute_soft(self, t, points_xy):
        alphas = self.layer_alpha(t, points_xy)
        base_shape = ((self.height, self.width) if points_xy is None
                      else points_xy.shape[:-1])
        n_layer = len(self.layers)
        weights = np.zeros(base_shape + (n_layer + 1,), dtype=np.float64)
        # transmittance from above: start fully open, multiply as we go front
        transmittance = np.ones(base_shape, dtype=np.float64)
        for k in range(n_layer - 1, -1, -1):
            alpha = alphas[k]
            weights[..., k + 1] = alpha * transmittance
            transmittance = transmittance * (1.0 - alpha)
        weights[..., BG_LAYER_ID] = transmittance
        return weights

    # ------------------------------------------------------------- motion math
    def layer_velocity(self, layer_id: int, t: float) -> np.ndarray:
        for layer in self.layers:
            if layer.layer_id == layer_id:
                return layer.velocity_at(t)
        return np.zeros(2)

    def flow(self, t: float, ids: np.ndarray | None = None) -> np.ndarray:
        """Analytic velocity field (px/s) of the visible surface at time t."""
        if ids is None:
            ids = self.id_map(t)
        flow = np.zeros((self.height, self.width, 2), dtype=np.float64)
        x, y = self.pixel_grid()
        for layer in self.layers:
            m = ids == layer.layer_id
            if not m.any():
                continue
            v = layer.velocity_at(t)
            if layer.omega != 0.0:
                cx, cy = layer.center(t)
                vmap = np.broadcast_to(v, (m.sum(), 2)).copy()
                vmap[:, 0] += -layer.omega * (y[m] - cy)
                vmap[:, 1] += layer.omega * (x[m] - cx)
                flow[m] = vmap
            else:
                flow[m] = v
        return flow

    def backtrace(self, points_xy: np.ndarray, t_from: float,
                  t_to: float, ids: np.ndarray | None = None) -> np.ndarray:
        """Position at t_to of the surface seen at ``points_xy`` at t_from.

        Full analytic motion (translation + acceleration + rotation), so
        turning paths are curved rather than straight flow extrapolations.
        """
        if ids is None:
            ids = self.id_map(t_from)
        pts = points_xy.astype(np.float64)
        out = pts.copy()
        x_idx = np.clip(np.round(pts[..., 0] - 0.5).astype(np.int64),
                        0, self.width - 1)
        y_idx = np.clip(np.round(pts[..., 1] - 0.5).astype(np.int64),
                        0, self.height - 1)
        visible_ids = ids[y_idx, x_idx]
        t_to_arr = np.broadcast_to(np.asarray(t_to, dtype=np.float64),
                                   pts.shape[:-1])
        for layer in self.layers:
            m = visible_ids == layer.layer_id
            if not m.any():
                continue
            p = pts[m]
            c_from = layer.center(t_from)
            rel = p - c_from
            dt = (t_to_arr - t_from)[m]
            if layer.omega != 0.0:
                dtheta = layer.omega * dt
                ca, sa = np.cos(dtheta), np.sin(dtheta)
                rx = ca * rel[:, 0] - sa * rel[:, 1]
                ry = sa * rel[:, 0] + ca * rel[:, 1]
                rel = np.stack([rx, ry], axis=-1)
            # evaluate the centre at every (possibly per-pixel) target time;
            # layer.center already contains translation, acceleration and the
            # optional oscillation, so no extra shift term is added
            target_times = t_to_arr[m]
            cto = np.empty((m.sum(), 2), dtype=np.float64)
            for tt in np.unique(target_times):
                sel = target_times == tt
                cto[sel] = layer.center(float(tt))
            out[m] = cto + rel
        return out


def _edge_coverage(coord: np.ndarray, half_extent: float) -> np.ndarray:
    """Fraction of a unit cell [c-0.5, c+0.5] inside |u| <= half_extent."""
    lo = np.clip(coord - 0.5, -half_extent, half_extent)
    hi = np.clip(coord + 0.5, -half_extent, half_extent)
    return np.clip(hi - lo, 0.0, 1.0)


def _disc_coverage(x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    """Approximate unit-cell coverage by a disc (2x2 deterministic sub-grid).

    Exact polygon clipping of a circle against a square is expensive; a 2x2
    sub-cell grid is vectorised, cheap and accurate to a few percent, which is
    ample once the result is integrated over many shutter time samples.
    """
    s = 2
    offs = (np.arange(s) + 0.5) / s - 0.5
    ox, oy = np.meshgrid(offs, offs)
    sx = x[..., None] + ox.reshape(-1)
    sy = y[..., None] + oy.reshape(-1)
    inside = (sx * sx + sy * sy) <= r * r
    return inside.mean(axis=-1)


# --------------------------------------------------------------------- scenes
def constant_scene(h: int = 256, w: int = 256) -> Scene:
    """A disc moving at constant velocity across a static checker background."""
    fg = Layer(
        layer_id=1, kind="disc", size=(34.0, 0.0),
        color=(0.92, 0.35, 0.18), pattern="solid",
        p0=(-20.0, h * 0.5), velocity=(140.0, 0.0),
    )
    return Scene(h, w, [fg])


def turning_scene(h: int = 256, w: int = 256) -> Scene:
    """A rectangle on a curved path: constant x velocity plus y acceleration,
    with slow rotation, so the true exposure path is curved."""
    fg = Layer(
        layer_id=2, kind="rect", size=(52.0, 30.0),
        color=(0.30, 0.72, 0.85), pattern="solid",
        p0=(40.0, 70.0), velocity=(110.0, 10.0),
        accel=(0.0, 95.0), omega=1.2,
    )
    return Scene(h, w, [fg], bg_pattern="gradient")


def occlusion_scene(h: int = 256, w: int = 320) -> Scene:
    """Fast foreground disc crossing a slow background bar over the shutter."""
    background_bar = Layer(
        layer_id=1, kind="rect", size=(w * 0.9, 46.0),
        color=(0.25, 0.55, 0.32), pattern="solid",
        p0=(w * 0.5, h * 0.55), velocity=(18.0, 0.0),
    )
    foreground = Layer(
        layer_id=2, kind="disc", size=(26.0, 0.0),
        color=(0.90, 0.25, 0.30), pattern="solid",
        p0=(-30.0, h * 0.55), velocity=(260.0, 0.0),
    )
    return Scene(h, w, [background_bar, foreground], bg_pattern="checker")


SCENES = {
    "constant": constant_scene,
    "turning": turning_scene,
    "occlusion": occlusion_scene,
}


def _dial_scene_factory(h: int = 256, w: int = 320):
    # lazy import: demo_scenes imports symbols from this module
    from .demo_scenes import dial_scene
    return dial_scene(h, w)


SCENES["dial"] = _dial_scene_factory

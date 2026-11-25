"""Demo scenes for the rolling-shutter / exposure-window material.

The ``dial`` scene is purpose-built so a single sample frame lets the viewer
tell motion smear apart from progressive (rolling) distortion:

* a rotating **dial** carrying thin radial spokes and a rim of tick marks
  with numbered angular labels - the ticks are a physical time ruler;
* a separate **foreground narrow strip** moving on a non-uniform (sinusoidal
  plus constant-drift) path in front of the dial, so occlusion changes inside
  one exposure window;
* a static frame around the dial, which must stay perfectly sharp under every
  shutter setting and therefore exposes any direction-blind fake blur.

The scene stays inside the layered :class:`Scene` model: one rotating layer
and one oscillating layer, both with analytic, continuously defined motion, so
the same world renders the high-sample reference and every shutter variant.
"""
from __future__ import annotations

import numpy as np

from .scene import Layer, Scene, _edge_coverage, _pattern_texture

BG_LAYER_ID = 0
DIAL_LAYER_ID = 3
STRIP_LAYER_ID = 4

DIAL_RADIUS_FRAC = 0.40
HUB_RADIUS_FRAC = 0.055
SPOKE_COUNT = 12
SPOKE_HALF_WIDTH = 1.1          # px: genuinely thin lines
TICK_COUNT = 60
TICK_HALF_WIDTH = 0.7
TICK_RADIAL = (0.86, 0.97)      # inner/outer radius fraction of the tick band
FACE_OPACITY = 0.28             # translucent disc so the line art stays visible
LINE_COLOR = (0.93, 0.95, 1.0)
TICK_COLOR = (1.0, 0.82, 0.45)


def dial_geometry(h: int, w: int) -> dict:
    """Fixed geometric parameters of the dial for a given canvas size."""
    cx, cy = w / 2.0, h / 2.0
    r = DIAL_RADIUS_FRAC * min(h, w)
    return {
        "center": (cx, cy),
        "radius": r,
        "hub_radius": HUB_RADIUS_FRAC * min(h, w),
        "spoke_count": SPOKE_COUNT,
        "spoke_half_width": SPOKE_HALF_WIDTH,
        "tick_count": TICK_COUNT,
        "tick_inner": r * TICK_RADIAL[0],
        "tick_outer": r * TICK_RADIAL[1],
        "tick_half_width": TICK_HALF_WIDTH,
    }


def _polar(points_xy, cx, cy):
    dx = points_xy[..., 0] - cx
    dy = points_xy[..., 1] - cy
    radius = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    return radius, theta


def _spoke_coverage(geo, local_xy):
    """Area coverage of the rotating thin radial spokes + hub."""
    r, theta = _polar(local_xy, 0.0, 0.0)
    inside = r <= geo["radius"]
    hub = r <= geo["hub_radius"]
    # angular distance to the nearest spoke centre, wrapped to [-pi/k, pi/k]
    k = geo["spoke_count"]
    step = 2.0 * np.pi / k
    delta = np.mod(theta + step / 2.0, step) - step / 2.0
    # arc length at this radius vs the fixed pixel half-width
    arc = np.abs(delta) * np.maximum(r, 1.0)
    spoke = inside & (arc <= geo["spoke_half_width"])
    # unit-cell coverage approximation from the distance to a 1-D edge
    cov = np.clip(geo["spoke_half_width"] + 0.5 - arc, 0.0, 1.0)
    cov = np.where(inside, cov, 0.0)
    cov = np.maximum(cov, hub.astype(np.float64))
    return np.clip(cov, 0.0, 1.0)


def _tick_coverage(geo, local_xy):
    """Coverage of the rim tick marks (the physical time ruler)."""
    r, theta = _polar(local_xy, 0.0, 0.0)
    in_band = (r >= geo["tick_inner"]) & (r <= geo["tick_outer"])
    k = geo["tick_count"]
    step = 2.0 * np.pi / k
    delta = np.mod(theta + step / 2.0, step) - step / 2.0
    arc = np.abs(delta) * np.maximum(r, 1.0)
    cov = np.clip(geo["tick_half_width"] + 0.5 - arc, 0.0, 1.0)
    return np.where(in_band, cov, 0.0)


def dial_dial_alpha(geo, local_xy):
    """Per-component dial RGBA coverage (face, spokes, tick ruler).

    Returns an array with a trailing axis of three coverage channels that the
    scene composites with separate colours: the face is translucent so the
    opaque radial thin lines and the tick ruler stay visible through it.
    """
    r = np.hypot(local_xy[..., 0], local_xy[..., 1])
    face = _edge_coverage(r, geo["radius"]) * FACE_OPACITY
    ticks = _tick_coverage(geo, local_xy)
    spokes = _spoke_coverage(geo, local_xy)
    return np.clip(np.stack([face, spokes, ticks], axis=-1), 0.0, 1.0)


def dial_composite(local_xy, geo, bg_rgb):
    """Composite the dial's three coloured components over the backdrop."""
    comps = dial_dial_alpha(geo, local_xy)
    face_a, spoke_a, tick_a = comps[..., 0], comps[..., 1], comps[..., 2]
    face_col = np.array([0.62, 0.66, 0.74])
    out = bg_rgb.copy()
    out = face_col * face_a[..., None] + out * (1.0 - face_a[..., None])
    out = np.array(LINE_COLOR) * spoke_a[..., None] + out * (
        1.0 - spoke_a[..., None])
    out = np.array(TICK_COLOR) * tick_a[..., None] + out * (
        1.0 - tick_a[..., None])
    # scalar coverage for the layer's id/visibility bookkeeping
    alpha = np.clip(comps.max(axis=-1), 0.0, 1.0)
    return np.clip(out, 0.0, 1.0), alpha


class DialLayer(Layer):
    """Rotating dial rendered as thin analytic spokes and a tick ruler."""

    kind = "dial"

    def __init__(self, layer_id: int, h: int, w: int, omega: float):
        super().__init__(
            layer_id=layer_id, kind="dial",
            size=(DIAL_RADIUS_FRAC * min(h, w), 0.0),
            color=(0.82, 0.85, 0.90), pattern="solid",
            p0=(w / 2.0, h / 2.0), omega=omega,
        )
        self.geo = dial_geometry(h, w)


def _install_dial_render(scene: Scene, dial: DialLayer) -> None:
    """Teach a Scene how to alpha-composite the dial's analytic line art."""
    original_alpha = scene._alpha_at

    def alpha_at(layer, local_xy):
        if layer is dial:
            # scalar coverage for id maps and the soft visibility stack
            return dial_dial_alpha(dial.geo, local_xy).max(axis=-1)
        return original_alpha(layer, local_xy)

    def composite(local_xy, bg_rgb):
        return dial_composite(local_xy, dial.geo, bg_rgb)

    scene._alpha_at = alpha_at  # type: ignore[method-assign]
    scene._composite_hooks[dial.layer_id] = composite


def dial_scene(h: int = 256, w: int = 320,
               omega: float = 1.6,
               strip_drift: float = 95.0,
               strip_osc_amp: float = 26.0,
               strip_osc_freq: float = 0.9) -> Scene:
    """Rotating thin-line dial with an oscillating narrow foreground strip.

    Parameters are plain numbers and are echoed into the saved manifest so a
    rebuilt scene is bit-for-bit the same world.

    Parameters
    ----------
    omega:
        Dial angular velocity in rad/s (constant rotation).
    strip_drift:
        Mean horizontal velocity (px/s) of the foreground strip.
    strip_osc_amp, strip_osc_freq:
        Sinusoidal horizontal oscillation (px, Hz) added to the drift, making
        the strip's velocity non-uniform over a single exposure window.
    """
    scene = Scene(h, w, [], bg_pattern="checker",
                  bg_color=(0.10, 0.12, 0.17))
    dial = DialLayer(DIAL_LAYER_ID, h, w, omega)
    strip_h = h * 0.62
    strip = Layer(
        layer_id=STRIP_LAYER_ID, kind="vstrip",
        size=(5.0, strip_h),
        color=(0.95, 0.32, 0.20), pattern="stripes",
        p0=(w * 0.18, h * 0.5),
        velocity=(strip_drift, 0.0),
        osc_amp=(strip_osc_amp, 0.0),
        osc_freq=strip_osc_freq,
        osc_phase=0.4,
    )
    scene.layers = [dial, strip]
    _install_dial_render(scene, dial)
    scene.dial_geometry = dial.geo  # type: ignore[attr-defined]
    scene.dial_params = {  # type: ignore[attr-defined]
        "height": h, "width": w, "omega": omega,
        "strip_drift": strip_drift, "strip_osc_amp": strip_osc_amp,
        "strip_osc_freq": strip_osc_freq,
    }
    return scene


# ----------------------------------------------------- analytic truth helpers
def spoke_world_angle(spoke_index: int, t: float, omega: float) -> float:
    """World angle of spoke ``spoke_index`` at time t (radians)."""
    step = 2.0 * np.pi / SPOKE_COUNT
    return spoke_index * step + omega * t


def strip_center_x(scene: Scene, t: float) -> float:
    """Analytic x position of the oscillating foreground strip centre."""
    strip = next(layer for layer in scene.layers
                 if layer.layer_id == STRIP_LAYER_ID)
    return float(strip.center(t)[0])


def dial_omega(scene: Scene) -> float:
    return scene.dial_params["omega"]  # type: ignore[attr-defined]

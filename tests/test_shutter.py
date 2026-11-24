"""Unit tests for the shutter motion-blur toolkit.

Run with:  python3 -m pytest tests/ -q
(or)       python3 tests/test_shutter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shutter_blur import (
    ShutterError,
    ShutterKernel,
    gaussian_blur,
    make_scene,
    rmse,
)
from shutter_blur.integrators import (
    METHODS,
    SceneVisibilityProvider,
    integrate_time_split,
    pixel_grid,
    compute_paths,
    render_shutter,
    segment_count_map,
)
from shutter_blur.io import draw_paths_overlay, draw_shutter_band, save_frame
from shutter_blur.motion import SceneMotionField
from shutter_blur.pipeline import VideoSource, synthetic_video_source

DT = 1.0 / 24.0
T = 4 * DT


def _provider_and_motion(scene):
    return SceneMotionField(scene), SceneVisibilityProvider(scene)


def _physical_gt(scene, kernel, t, samples=400):
    offs = np.linspace(-kernel.exposure / 2, kernel.exposure / 2, samples)
    gt = np.zeros((scene.height, scene.width, 3))
    for tau in offs:
        gt += scene.render(t + tau)[0]
    return np.clip(gt / samples, 0.0, 1.0)


# ------------------------------------------------------------- kernel rules
def test_shutter_angle_exposure_relationship():
    kernel = ShutterKernel.from_angle(180.0, DT, 16)
    assert np.isclose(kernel.exposure, DT / 2)
    assert np.isclose(kernel.shutter_angle_deg, 180.0)
    # displacement is velocity*E, never conflated with E itself
    scene = make_scene("constant")
    v = scene.layers[0].velocity[0]
    kernel = ShutterKernel.from_angle(360.0, DT, 16)
    smear_px = v * kernel.exposure
    assert np.isclose(smear_px, v * DT)
    assert smear_px != kernel.exposure  # length (px) vs duration (s)


def test_overlong_exposure_rejected():
    try:
        ShutterKernel(exposure=DT * 1.1, frame_interval=DT)
    except ShutterError:
        return
    raise AssertionError("exposure > frame interval must raise ShutterError")


def test_weights_normalized_and_independent_of_count():
    for n in (4, 16, 64, 128):
        w = ShutterKernel(DT / 2, DT, n).weights()
        assert np.isclose(w.sum(), 1.0)
        assert np.allclose(w, 1.0 / n)


# ------------------------------------------------- static brightness invariance
def test_static_region_does_not_brighten_with_more_samples():
    scene = make_scene("constant")
    clean, _ = scene.render(T)
    # interior pixels that never change over the shutter
    static = np.ones((scene.height, scene.width), bool)
    kernel = ShutterKernel.from_angle(180.0, DT, 8)
    ref = scene.render(T - kernel.exposure / 2)[0]
    for tau in np.linspace(-kernel.exposure / 2, kernel.exposure / 2, 16):
        static &= np.abs(scene.render(T + tau)[0] - ref).mean(2) < 1e-9
    static[:8] = False
    static[-8:] = False
    static[:, :8] = False
    static[:, -8:] = False
    brightness = {}
    for n in (8, 32, 128):
        k = ShutterKernel.from_angle(180.0, DT, n)
        motion, provider = _provider_and_motion(scene)
        r = render_shutter(clean, motion, k, T, "time_split", provider)
        brightness[n] = r.image[static].mean()
    assert np.allclose(list(brightness.values()), clean[static].mean(),
                       atol=1e-8)


# --------------------------------------------------------- convergence to truth
def test_visibility_methods_converge_to_ground_truth():
    scene = make_scene("constant")
    motion, provider = _provider_and_motion(scene)
    clean, _ = scene.render(T)
    points = pixel_grid(scene.height, scene.width)
    gt = _physical_gt(scene, ShutterKernel.from_angle(180, DT, 64), T)
    errors = {}
    for grid in (16, 64, 256):
        k = ShutterKernel.from_angle(180.0, DT, grid)
        paths = compute_paths(motion, points, T, k.sample_offsets())
        img, _, _ = integrate_time_split(clean, motion, k, T, points, paths,
                                         provider, grid_steps=grid)
        errors[grid] = rmse(img, gt)
    assert errors[256] < errors[16]
    assert errors[256] < 5e-4


def test_all_three_scenes_achieve_low_error():
    for name in ("constant", "turning", "occlusion"):
        scene = make_scene(name)
        angle = 360.0 if name == "occlusion" else 180.0
        k = ShutterKernel.from_angle(angle, DT, 48)
        motion, provider = _provider_and_motion(scene)
        clean, _ = scene.render(T)
        r = render_shutter(clean, motion, k, T, "time_split", provider)
        gt = _physical_gt(scene, k, T)
        assert rmse(r.image, gt) < 0.01, name


# ------------------------------------------------- occlusion exposes naive flaw
def test_naive_line_average_leaks_more_than_visibility_methods():
    scene = make_scene("occlusion")
    k = ShutterKernel.from_angle(360.0, DT, 48)
    motion, provider = _provider_and_motion(scene)
    clean, _ = scene.render(T)
    gt = _physical_gt(scene, k, T)
    # foreground-owner pixels: naive drags foreground colour beyond its reach
    owner = scene.id_map(T)
    fg = owner == 2
    results = {m: render_shutter(clean, motion, k, T, m, provider).image
               for m in METHODS}
    naive_err = rmse(results["naive"], gt, fg)
    split_err = rmse(results["time_split"], gt, fg)
    vis_err = rmse(results["visibility"], gt, fg)
    assert split_err < naive_err
    assert vis_err < naive_err


def test_occlusion_produces_multiple_temporal_segments():
    scene = make_scene("occlusion")
    k = ShutterKernel.from_angle(360.0, DT, 48)
    motion, provider = _provider_and_motion(scene)
    points = pixel_grid(scene.height, scene.width)
    seg = segment_count_map(provider, points, T, k, motion)
    assert int((seg > 1).sum()) > 50


# ------------------------------------------------------- gaussian is not motion
def test_gaussian_baseline_is_direction_blind_and_worse():
    scene = make_scene("constant")
    k = ShutterKernel.from_angle(180.0, DT, 48)
    motion, provider = _provider_and_motion(scene)
    clean, _ = scene.render(T)
    r = render_shutter(clean, motion, k, T, "time_split", provider)
    gt = _physical_gt(scene, k, T)
    gauss = gaussian_blur(clean, 6.0)
    assert rmse(gauss, gt) > 3.0 * rmse(r.image, gt)
    # a gaussian also blurs static vertical texture that motion blur preserves
    corner = np.zeros(clean.shape[:2], bool)
    corner[:24, :24] = True
    assert np.abs(gauss[corner] - clean[corner]).mean() > \
        np.abs(r.image[corner] - clean[corner]).mean()


# ------------------------------------------------------------- curved paths
def test_turning_path_is_curved_not_straight():
    scene = make_scene("turning")
    motion = SceneMotionField(scene, curved=True)
    ids = scene.id_map(T)
    ys, xs = np.nonzero(ids == 2)
    y, x = ys[len(ys) // 2], xs[len(xs) // 2]
    points = np.array([[[x + 0.5, y + 0.5]]])
    linear = SceneMotionField(scene, curved=False)
    k = ShutterKernel.from_angle(180.0, DT, 32)
    curved_end = motion.backtrace(points, T, T - k.exposure / 2)
    straight_end = linear.backtrace(points, T, T - k.exposure / 2)
    assert np.linalg.norm(curved_end - straight_end) > 0.5


def test_exposure_length_changes_blur_width():
    scene = make_scene("constant")
    widths = {}
    for angle in (45.0, 180.0, 360.0):
        k = ShutterKernel.from_angle(angle, DT, 48)
        motion, provider = _provider_and_motion(scene)
        clean, _ = scene.render(T)
        r = render_shutter(clean, motion, k, T, "time_split", provider)
        # total foreground smear extent grows linearly with exposure
        widths[angle] = (np.abs(r.image - clean).mean(2) > 0.02).sum()
    assert widths[360.0] > widths[180.0] > widths[45.0]


# ------------------------------------------------- real sampled-video path
def test_video_source_pipeline_runs_and_blurs_motion():
    # Treat the analytic world as a frame-sampled video: only rendered frames
    # and estimated block-matching flow are available, no ground-truth layers.
    scene = make_scene("constant")
    src = synthetic_video_source(scene, DT, 6)
    assert len(src.flows_fwd) == 6
    k = ShutterKernel.from_angle(270.0, DT, 16)
    bundle = src.render_frame(k, 3, "time_split")
    out = bundle.blurred["time_split"]
    clean = src.frames[3]
    assert out.shape == clean.shape
    assert np.isfinite(out).all()
    # static background keeps roughly constant brightness; block-matching on
    # checker edges introduces small spurious flow, so tolerance is looser
    # than the exact 1e-8 invariance of the analytic visibility integrator
    corner = np.zeros(clean.shape[:2], bool)
    corner[:20, :20] = True
    assert abs(out[corner].mean() - clean[corner].mean()) < 0.06
    changed = np.abs(out - clean).mean(2) > 0.01
    assert changed.sum() > 100
    # naive and visibility paths both execute on the video provider
    for method in METHODS:
        b = src.render_frame(k, 3, method)
        assert np.isfinite(b.blurred[method]).all()


# ------------------------------------------------- overlay save round trip
def test_overlay_frame_saves_without_whiteout(tmp_path=None):
    # draw_paths_overlay/draw_shutter_band return uint8 [0,255]; save_frame
    # must not re-scale them as if they were [0,1] floats (regression: the
    # benchmark overlay PNGs were clipped to pure white).
    import tempfile
    from pathlib import Path
    from PIL import Image
    scene = make_scene("occlusion")
    motion, provider = _provider_and_motion(scene)
    k = ShutterKernel.from_angle(360.0, DT, 24)
    clean, _ = scene.render(T)
    result = render_shutter(clean, motion, k, T, "time_split", provider)
    overlay = draw_paths_overlay(result.image, result.paths, every=20)
    overlay = draw_shutter_band(
        overlay, T, k.exposure, DT, [T - DT, T, T + DT])
    with tempfile.TemporaryDirectory() as d:
        path = save_frame(overlay, Path(d) / "overlay.png")
        back = np.asarray(Image.open(path))
    assert back.min() < 250  # a whiteout has min == 255
    assert back.shape == overlay.shape


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failures}/{len(fns)} tests passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)

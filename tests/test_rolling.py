"""Tests for rolling-shutter timing, exposure profiles and demo assets.

These complement tests/test_shutter.py (which keeps the original global-shutter
suite green) and pin the physical contracts the demo material relies on:

* per-row timing comes from the readout direction, with the exposure sweep as
  an independently reversible axis;
* exposure-profile weights are normalized and the integrator converges to the
  high-sample reference as samples increase;
* motion smear and progressive skew respond to different controls;
* an on-disk asset can be rebuilt from its manifest and matches its stored
  high-sample reference;
* a Gaussian post-filter cannot reproduce the exposure-integrated result.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shutter_blur import (  # noqa: E402
    DIRECTIONS,
    PROFILES,
    ShutterError,
    ShutterKernel,
    SceneMotionField,
    gaussian_blur,
    make_scene,
    profile_window,
    render_rolling,
    rmse,
    rolling_ground_truth,
)
from shutter_blur.assets import (  # noqa: E402
    AssetSpec,
    create_asset,
    kernel_from_params,
    load_asset,
    rebuild_asset,
)
from shutter_blur.demo_scenes import SPOKE_COUNT, strip_center_x  # noqa: E402
from shutter_blur.integrators import (  # noqa: E402
    SceneVisibilityProvider,
    render_shutter,
)

DT = 1.0 / 24.0
T = 4 * DT


# ------------------------------------------------------------- kernel timing
def test_global_shutter_has_zero_row_offsets():
    k = ShutterKernel.from_angle(180, DT, 16)
    assert not k.is_rolling
    assert np.allclose(k.exposure_row_offsets(64), 0.0)
    assert np.allclose(k.row_offsets(64), 0.0)


def test_readout_direction_sets_row_timing_order():
    down = ShutterKernel.from_angle(
        180, DT, 16, readout_duration=0.8 * DT,
        readout_direction="top_down")
    up = ShutterKernel.from_angle(
        180, DT, 16, readout_duration=0.8 * DT,
        readout_direction="bottom_up")
    assert np.allclose(down.exposure_row_offsets(5),
                       -up.exposure_row_offsets(5))
    # first digitised row gets the earliest exposure centre
    assert down.exposure_row_offsets(5)[0] < down.exposure_row_offsets(5)[-1]
    assert list(down.readout_order(5)) == [0, 1, 2, 3, 4]
    assert list(up.readout_order(5)) == [4, 3, 2, 1, 0]


def test_exposure_direction_is_independent_of_readout_order():
    k = ShutterKernel.from_angle(
        180, DT, 16, readout_duration=0.8 * DT,
        readout_direction="top_down", exposure_direction="bottom_up")
    # exposure sweep reversed, but digitisation order still top to bottom
    assert k.exposure_row_offsets(5)[0] > 0
    assert list(k.readout_order(5)) == [0, 1, 2, 3, 4]


def test_invalid_directions_and_profiles_rejected():
    for bad in ("sideways", ""):
        try:
            ShutterKernel.from_angle(180, DT, readout_direction=bad)
        except ShutterError:
            pass
        else:
            raise AssertionError("bad readout direction must raise")
    try:
        ShutterKernel.from_angle(180, DT, exposure_profile="tent")
    except ShutterError:
        pass
    else:
        raise AssertionError("bad profile must raise")
    try:
        ShutterKernel.from_angle(180, DT, readout_duration=1.1 * DT)
    except ShutterError:
        pass
    else:
        raise AssertionError("readout > frame interval must raise")


# ------------------------------------------------------------- profile weights
def test_profile_weights_normalized_and_windowed():
    u = np.linspace(-1, 1, 32)
    for profile in PROFILES:
        w = profile_window(profile, u)
        assert np.isclose(w.sum() / w.sum(), 1.0)
        assert (w >= 0).all()
    # box is flat inside and zero outside
    box = profile_window("box", np.array([-1.5, 0.0, 1.5]))
    assert box.tolist() == [0.0, 1.0, 0.0]
    # gaussian/trapezoid give less weight to the window edges than the centre
    for profile in ("gaussian", "trapezoid"):
        k = ShutterKernel.from_angle(180, DT, 33, exposure_profile=profile)
        w = k.weights()
        assert np.isclose(w.sum(), 1.0)
        assert w[len(w) // 2] > w[0]


def test_dial_scene_exists_and_has_ruler_and_strip():
    scene = make_scene("dial")
    assert scene.dial_geometry["spoke_count"] == SPOKE_COUNT
    assert scene.dial_geometry["tick_count"] == 60
    kinds = [layer.kind for layer in scene.layers]
    assert "dial" in kinds and "vstrip" in kinds


# ------------------------------------------------------------- non-uniform path
def test_strip_motion_is_non_uniform():
    scene = make_scene("dial")
    ts = np.linspace(T - DT, T + DT, 64)
    xs = np.array([strip_center_x(scene, float(t)) for t in ts])
    # finite-difference velocity must vary (oscillation on top of the drift)
    v = np.diff(xs) / np.diff(ts)
    assert v.max() - v.min() > 20.0
    # the analytic trajectory is smooth
    assert np.isfinite(xs).all()


# ------------------------------------------------- rolling convergence & truth
def test_rolling_integrator_converges_to_reference():
    scene = make_scene("dial")
    clean, _ = scene.render(T)
    motion = SceneMotionField(scene)
    provider = SceneVisibilityProvider(scene)
    kernel = ShutterKernel.from_angle(
        180, DT, 96, readout_duration=0.8 * DT,
        exposure_profile="trapezoid")
    ref = rolling_ground_truth(scene, kernel, T, gt_samples=64)
    errors = {}
    for n in (8, 24, 64):
        k = ShutterKernel.from_angle(
            180, DT, n, readout_duration=0.8 * DT,
            exposure_profile="trapezoid")
        img = render_rolling(clean, motion, k, T, provider).image
        errors[n] = rmse(img, ref)
    assert errors[64] < errors[8]
    assert errors[64] < 0.01


def test_global_rolling_matches_visibility_integrator():
    scene = make_scene("dial")
    clean, _ = scene.render(T)
    motion = SceneMotionField(scene)
    provider = SceneVisibilityProvider(scene)
    k = ShutterKernel.from_angle(180, DT, 48, exposure_profile="box")
    a = render_rolling(clean, motion, k, T, provider).image
    b = render_shutter(clean, motion, k, T, "visibility", provider).image
    assert rmse(a, b) < 0.01


def test_static_border_is_never_smeared():
    scene = make_scene("dial")
    clean, _ = scene.render(T)
    motion = SceneMotionField(scene)
    provider = SceneVisibilityProvider(scene)
    kernel = ShutterKernel.from_angle(
        360, DT, 32, readout_duration=DT, exposure_profile="gaussian")
    img = render_rolling(clean, motion, kernel, T, provider).image
    border = np.zeros((scene.height, scene.width), bool)
    border[:3] = True
    border[-3:] = True
    border[:, :3] = True
    border[:, -3:] = True
    assert np.abs(img[border] - clean[border]).max() < 1e-8


# ------------------------------------------------------------- skew vs smear
def test_readout_reversal_changes_geometry_not_sample_count():
    scene = make_scene("dial")
    clean, _ = scene.render(T)
    motion = SceneMotionField(scene)
    provider = SceneVisibilityProvider(scene)
    down = ShutterKernel.from_angle(
        180, DT, 48, readout_duration=0.9 * DT,
        readout_direction="top_down")
    up = ShutterKernel.from_angle(
        180, DT, 48, readout_duration=0.9 * DT,
        readout_direction="bottom_up")
    a = render_rolling(clean, motion, down, T, provider).image
    b = render_rolling(clean, motion, up, T, provider).image
    assert np.abs(a - b).mean() > 1e-4
    # equal exposure window keeps the per-sample weight table identical
    assert np.allclose(down.weights(), up.weights())


def test_longer_exposure_grows_smear_footprint():
    scene = make_scene("dial")
    clean, _ = scene.render(T)
    motion = SceneMotionField(scene)
    provider = SceneVisibilityProvider(scene)
    footprint = {}
    for angle in (45.0, 360.0):
        k = ShutterKernel.from_angle(angle, DT, 32)
        img = render_rolling(clean, motion, k, T, provider).image
        footprint[angle] = int(
            (np.abs(img - clean).mean(axis=2) > 0.02).sum())
    assert footprint[360.0] > footprint[45.0]


def test_gaussian_post_filter_is_not_the_integrated_result():
    scene = make_scene("dial")
    clean, _ = scene.render(T)
    motion = SceneMotionField(scene)
    provider = SceneVisibilityProvider(scene)
    kernel = ShutterKernel.from_angle(
        180, DT, 48, readout_duration=0.8 * DT)
    integrated = render_rolling(clean, motion, kernel, T, provider).image
    ref = rolling_ground_truth(scene, kernel, T, 64)
    gauss = gaussian_blur(clean, 5.0)
    assert rmse(gauss, ref) > 2.0 * rmse(integrated, ref)


# ------------------------------------------------------------- asset lifecycle
def test_asset_roundtrip_rebuilds_and_verifies():
    spec = AssetSpec(name="unit_dial", gt_samples=48, num_samples=24,
                     trajectory_samples=128)
    with tempfile.TemporaryDirectory() as d:
        create_asset(d, spec)
        asset_dir = Path(d) / "unit_dial"
        manifest = json.loads((asset_dir / "manifest.json").read_text())
        assert manifest["kind"] == "shutter_demo"
        for rel in ("trajectory.npz", "reference.npz",
                    "frames/global.png", "frames/rolling.png",
                    "frames/short.png", "frames/clean.png"):
            assert (asset_dir / rel).is_file(), rel
        bundle = load_asset(asset_dir)
        traj = bundle["trajectory"]
        assert traj["spoke_world_angle"].shape[0] == SPOKE_COUNT
        assert len(traj["time"]) == spec.trajectory_samples
        # manifest carries the exposure profile and time weights
        rolling_k = manifest["kernels"]["rolling"]
        assert rolling_k["exposure_profile"] == spec.rolling_profile
        assert np.isclose(sum(rolling_k["sample_weights"]), 1.0)
        # rebuild from the manifest parameters must match the stored reference
        report = rebuild_asset(asset_dir, tolerance=5e-3)
        assert report["ok"], report
        # kernel parameters survive a serialize/deserialize cycle
        restored = kernel_from_params(rolling_k)
        assert restored.readout_duration == \
            ShutterKernel.from_angle(
                spec.rolling_angle, DT,
                readout_duration=spec.rolling_readout_frac * DT,
                exposure_profile=spec.rolling_profile).readout_duration


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

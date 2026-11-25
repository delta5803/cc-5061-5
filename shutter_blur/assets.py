"""Authoring and persistence of shutter demo material.

A demo *asset* is a self-describing bundle on disk, never just an image:

* ``manifest.json`` - every scene and kernel parameter needed to rebuild the
  exact same world, plus file hashes and the exposure-profile time weights;
* ``trajectory.npz`` - the analytic source trajectories (dial spoke angles and
  the oscillating strip centre) sampled on a fine time grid, i.e. the known
  motion path the frames were generated from;
* ``reference.npz`` - the high-sample exposure-integrated references, one per
  shutter variant;
* ``frames/*.png`` - the actual rendered frames (global, rolling, clean),
  produced by real temporal exposure integration, not a post blur filter.

A bundle can be reloaded and rebuilt: :func:`rebuild_asset` re-runs the
integrators from the manifest parameters and the result must match the stored
reference within an explicit tolerance.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .demo_scenes import (
    SPOKE_COUNT,
    spoke_world_angle,
    strip_center_x,
)
from .integrators import SceneVisibilityProvider
from .io import save_frame
from .kernel import ShutterKernel
from .motion import SceneMotionField
from .rolling import render_rolling, rolling_ground_truth

DT = 1.0 / 24.0
ASSET_VERSION = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kernel_params(kernel: ShutterKernel) -> dict:
    return {
        "exposure": kernel.exposure,
        "frame_interval": kernel.frame_interval,
        "num_samples": kernel.num_samples,
        "readout_duration": kernel.readout_duration,
        "readout_direction": kernel.readout_direction,
        "exposure_direction": kernel.exposure_direction,
        "exposure_profile": kernel.exposure_profile,
        "shutter_angle_deg": kernel.shutter_angle_deg,
        "is_rolling": kernel.is_rolling,
        "sample_offsets": kernel.sample_offsets().tolist(),
        "sample_weights": kernel.weights().tolist(),
    }


def kernel_from_params(p: dict) -> ShutterKernel:
    return ShutterKernel(
        exposure=float(p["exposure"]),
        frame_interval=float(p["frame_interval"]),
        num_samples=int(p["num_samples"]),
        readout_duration=float(p.get("readout_duration", 0.0)),
        readout_direction=p.get("readout_direction", "top_down"),
        exposure_direction=p.get("exposure_direction", "top_down"),
        exposure_profile=p.get("exposure_profile", "box"),
    )


@dataclass
class AssetSpec:
    """Editable parameters of one demo asset (all plain JSON-able values)."""

    name: str
    height: int = 256
    width: int = 320
    frame_index: int = 4
    gt_samples: int = 96
    trajectory_samples: int = 240
    omega: float = 1.6
    strip_drift: float = 95.0
    strip_osc_amp: float = 26.0
    strip_osc_freq: float = 0.9
    # three shutter variants exercise the independent time axes
    global_angle: float = 180.0
    rolling_angle: float = 180.0
    rolling_readout_frac: float = 0.9
    rolling_profile: str = "trapezoid"
    short_angle: float = 45.0
    num_samples: int = 32

    def to_dict(self) -> dict:
        return asdict(self)


def build_trajectory(scene, spec: AssetSpec, t_center: float,
                     kernel: ShutterKernel) -> dict:
    """Analytic source trajectories over a window wider than the exposure.

    Includes every spoke's world angle and the strip centre position/velocity,
    so the stored motion path is independent of any rendered frame.
    """
    half = max(kernel.exposure, kernel.readout_duration) / 2.0
    ts = np.linspace(t_center - half * 1.5, t_center + half * 1.5,
                     spec.trajectory_samples)
    omega = scene.dial_params["omega"]
    spoke_angles = np.stack([
        spoke_world_angle(j, ts, omega) for j in range(SPOKE_COUNT)
    ], axis=0)
    strip_x = np.array([strip_center_x(scene, float(t)) for t in ts])
    strip = next(layer for layer in scene.layers if layer.kind == "vstrip")
    strip_vx = np.array([float(strip.velocity_at(float(t))[0]) for t in ts])
    return {
        "time": ts,
        "spoke_world_angle": spoke_angles,
        "strip_center_x": strip_x,
        "strip_velocity_x": strip_vx,
        "t_center": np.array(t_center),
        "exposure": np.array(kernel.exposure),
        "readout_duration": np.array(kernel.readout_duration),
    }


def create_asset(out_dir: str | Path, spec: AssetSpec) -> dict:
    """Render and persist one complete demo asset bundle."""
    from .demo_scenes import dial_scene

    out = Path(out_dir) / spec.name
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    scene = dial_scene(
        spec.height, spec.width, omega=spec.omega,
        strip_drift=spec.strip_drift,
        strip_osc_amp=spec.strip_osc_amp,
        strip_osc_freq=spec.strip_osc_freq,
    )
    t_center = spec.frame_index * DT
    motion = SceneMotionField(scene)
    provider = SceneVisibilityProvider(scene)
    clean, _ = scene.render(t_center)

    kernels = {
        "global": ShutterKernel.from_angle(
            spec.global_angle, DT, spec.num_samples,
            exposure_profile="box"),
        "rolling": ShutterKernel.from_angle(
            spec.rolling_angle, DT, spec.num_samples,
            readout_duration=spec.rolling_readout_frac * DT,
            readout_direction="top_down",
            exposure_direction="top_down",
            exposure_profile=spec.rolling_profile),
        "short": ShutterKernel.from_angle(
            spec.short_angle, DT, spec.num_samples,
            readout_duration=spec.rolling_readout_frac * DT,
            exposure_profile="box"),
    }

    rendered = {}
    references = {}
    frame_files = {}
    for label, kernel in kernels.items():
        result = render_rolling(clean, motion, kernel, t_center, provider)
        rendered[label] = result
        # rolling_ground_truth with R=0 is exactly the fixed-photosite global
        # reference, so both variants share one observation model and one
        # convergence target with render_rolling
        ref = rolling_ground_truth(scene, kernel, t_center,
                                   spec.gt_samples)
        references[label] = ref
        png = frames_dir / f"{label}.png"
        save_frame(result.image, png)
        frame_files[label] = str(png.relative_to(out))
    clean_png = frames_dir / "clean.png"
    save_frame(clean, clean_png)
    frame_files["clean"] = str(clean_png.relative_to(out))

    # source trajectory and high-sample references in one NPZ each
    traj = build_trajectory(scene, spec, t_center, kernels["rolling"])
    traj_path = out / "trajectory.npz"
    np.savez_compressed(traj_path, **traj)
    ref_path = out / "reference.npz"
    np.savez_compressed(ref_path, **references)

    manifest = {
        "asset_version": ASSET_VERSION,
        "kind": "shutter_demo",
        "created_from": "shutter_blur.assets.create_asset",
        "spec": spec.to_dict(),
        "scene": scene.dial_params,
        "frame_interval": DT,
        "t_center": t_center,
        "kernels": {k: kernel_params(v) for k, v in kernels.items()},
        "files": {
            "trajectory": "trajectory.npz",
            "reference": "reference.npz",
            "frames": frame_files,
        },
        "file_hashes": {
            "trajectory.npz": _sha256(traj_path),
            "reference.npz": _sha256(ref_path),
        },
        "profile_window_check": _profile_check(kernels),
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest["file_hashes"]["manifest.json"] = _sha256(manifest_path)
    return manifest


def _profile_check(kernels: dict) -> dict:
    """Per-variant exposure profile with its normalized time weights."""
    return {
        label: {
            "profile": k.exposure_profile,
            "offsets": k.sample_offsets().tolist(),
            "weights": k.weights().tolist(),
            "weights_sum": float(k.weights().sum()),
            "row_offsets": k.exposure_row_offsets(8).tolist(),
        }
        for label, k in kernels.items()
    }


def load_asset(asset_dir: str | Path) -> dict:
    asset_dir = Path(asset_dir)
    manifest = json.loads((asset_dir / "manifest.json").read_text())
    traj = dict(np.load(asset_dir / manifest["files"]["trajectory"]))
    ref = dict(np.load(asset_dir / manifest["files"]["reference"]))
    return {"manifest": manifest, "trajectory": traj, "reference": ref,
            "dir": asset_dir}


def rebuild_asset(asset_dir: str | Path, tolerance: float = 5e-3) -> dict:
    """Re-render every variant from the manifest and compare to the reference.

    Returns a report with the RMSE per variant.  A non-zero tolerance covers
    the finite sample count of the stored integrator output versus the
    high-sample reference; the check fails loudly if the reconstruction
    agreement is worse than the agreed bound.
    """
    from .demo_scenes import dial_scene
    from .metrics import rmse

    bundle = load_asset(asset_dir)
    manifest = bundle["manifest"]
    spec = AssetSpec(**manifest["spec"])
    sp = manifest["scene"]
    scene = dial_scene(
        sp["height"], sp["width"], omega=sp["omega"],
        strip_drift=sp["strip_drift"],
        strip_osc_amp=sp["strip_osc_amp"],
        strip_osc_freq=sp["strip_osc_freq"],
    )
    t_center = manifest["t_center"]
    clean, _ = scene.render(t_center)
    motion = SceneMotionField(scene)
    provider = SceneVisibilityProvider(scene)

    report = {"tolerance": tolerance, "variants": {}, "ok": True}
    for label, kp in manifest["kernels"].items():
        kernel = kernel_from_params(kp)
        result = render_rolling(clean, motion, kernel, t_center, provider)
        err = rmse(result.image, bundle["reference"][label])
        report["variants"][label] = {
            "rmse_vs_reference": err,
            "within_tolerance": err <= tolerance,
        }
        if err > tolerance:
            report["ok"] = False
    return report

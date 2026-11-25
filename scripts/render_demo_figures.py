"""Render visual verification figures for the dial shutter demo.

Outputs (under outputs/shutter_demo/figures):

* ``dial_variants.png`` - clean, global, rolling and short-exposure frames
  side by side, so motion smear and progressive skew are distinguishable;
* ``exposure_profiles.png`` - the box / trapezoid / gaussian exposure windows
  with their normalized sample time weights;
* ``source_trajectory.png`` - the saved analytic source trajectories (spoke
  angles and the non-uniform strip motion) on a time ruler.

All frames come from real temporal exposure integration; no post blur filter
is applied anywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shutter_blur import (  # noqa: E402
    ShutterKernel,
    SceneMotionField,
    make_scene,
    profile_window,
    render_rolling,
)
from shutter_blur.assets import load_asset  # noqa: E402
from shutter_blur.integrators import SceneVisibilityProvider  # noqa: E402
from shutter_blur.kernel import PROFILES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 24.0
FRAME_INDEX = 4


def variant_frames():
    scene = make_scene("dial")
    t = FRAME_INDEX * DT
    clean, _ = scene.render(t)
    motion = SceneMotionField(scene)
    provider = SceneVisibilityProvider(scene)
    kernels = {
        "clean (instant)": None,
        "global E=180°, R=0": ShutterKernel.from_angle(
            180, DT, 48, exposure_profile="box"),
        "rolling E=180°, R=0.9dt": ShutterKernel.from_angle(
            180, DT, 48, readout_duration=0.9 * DT,
            exposure_profile="trapezoid"),
        "short E=45°, R=0.9dt": ShutterKernel.from_angle(
            45, DT, 48, readout_duration=0.9 * DT,
            exposure_profile="box"),
    }
    frames = [("clean (instant)", clean)]
    for label, kernel in kernels.items():
        if kernel is None:
            continue
        frames.append((label,
                       render_rolling(clean, motion, kernel, t,
                                      provider).image))
    return frames


def write_variants(fig_dir: Path) -> None:
    frames = variant_frames()
    fig, axes = plt.subplots(1, len(frames), figsize=(16, 4.4))
    for ax, (label, img) in zip(axes, frames):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        "Dial demo: motion smear (exposure window) vs rolling skew "
        "(per-row readout)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "dial_variants.png", dpi=120)
    plt.close(fig)


def write_profiles(fig_dir: Path) -> None:
    u = np.linspace(-1.2, 1.2, 400)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
    for profile in PROFILES:
        ax.plot(u, profile_window(profile, u), label=profile, linewidth=2)
    samples = np.linspace(-1, 1, 17)
    ax.stem(samples, profile_window("trapezoid", samples),
            linefmt="C1-", markerfmt="C1o", basefmt=" ",
            label="trapezoid sample weights")
    ax.set_xlabel("normalized time u over the exposure window")
    ax.set_ylabel("relative sensitivity")
    ax.set_title("Exposure profiles and temporal sample weights")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "exposure_profiles.png", dpi=120)
    plt.close(fig)


def write_trajectory(fig_dir: Path, asset_dir: Path) -> None:
    bundle = load_asset(asset_dir)
    traj = bundle["trajectory"]
    ts = traj["time"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for j in range(traj["spoke_world_angle"].shape[0]):
        axes[0].plot(ts, traj["spoke_world_angle"][j],
                     color="C0", alpha=0.55, linewidth=1)
    axes[0].set_ylabel("spoke world angle (rad)")
    axes[0].set_title("Analytic source trajectories (saved in trajectory.npz)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(ts, traj["strip_center_x"], color="C3",
                 label="strip centre x")
    ax2 = axes[1].twinx()
    ax2.plot(ts, traj["strip_velocity_x"], color="C2", alpha=0.7,
             linestyle="--", label="strip velocity x")
    axes[1].set_ylabel("strip centre x (px)")
    ax2.set_ylabel("strip velocity x (px/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(alpha=0.3)
    t_c = float(traj["t_center"])
    E = float(traj["exposure"])
    R = float(traj["readout_duration"])
    for ax in axes:
        ax.axvspan(t_c - E / 2, t_c + E / 2, color="C1", alpha=0.15,
                   label="exposure window E")
        ax.axvline(t_c, color="k", linewidth=1)
    handles, labels = axes[1].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(handles + handles2, labels + labels2, loc="upper left",
                   fontsize=8)
    fig.suptitle(f"Time ruler: E={E*1000:.1f} ms, R={R*1000:.1f} ms")
    fig.tight_layout()
    fig.savefig(fig_dir / "source_trajectory.png", dpi=120)
    plt.close(fig)


def main() -> None:
    asset_dir = ROOT / "outputs" / "shutter_demo" / "assets" / "dial_demo"
    if not (asset_dir / "manifest.json").is_file():
        raise SystemExit(
            "asset missing; run: python -m scripts.shutter_cli author")
    fig_dir = ROOT / "outputs" / "shutter_demo" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    write_variants(fig_dir)
    write_profiles(fig_dir)
    write_trajectory(fig_dir, asset_dir)
    print("wrote figures to", fig_dir)


if __name__ == "__main__":
    main()

"""Benchmark: four integrators vs a high-sample physical ground truth.

Generates, for the constant-velocity, turning and layered-occlusion scenes:

* a metrics table (RMSE, contour/edge energy, static brightness ratio);
* a method-vs-GT comparison figure and an exposure sweep;
* real output frames (clean, GT, every method, Gaussian baseline);
* an MP4 video and an animated GIF of the time-split result.

Run:  python3 scripts/run_benchmark.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shutter_blur import (
    METHODS,
    ShutterKernel,
    contour_energy,
    gaussian_blur,
    make_scene,
    motion_roi,
    rmse,
    static_roi,
)
from shutter_blur.integrators import (
    SceneVisibilityProvider,
    ground_truth,
    render_shutter,
    segment_count_map,
)
from shutter_blur.io import (
    draw_paths_overlay,
    draw_shutter_band,
    save_frame,
    write_video,
)
from shutter_blur.motion import SceneMotionField
from shutter_blur.integrators import pixel_grid

OUT = Path(__file__).resolve().parents[1] / "outputs"
FRAME_DIR = OUT / "frames"
VIDEO_DIR = OUT / "videos"
CHART_DIR = OUT / "charts"
DT = 1.0 / 24.0
FRAME_INDEX = 4
ANGLE = {"constant": 180.0, "turning": 180.0, "occlusion": 360.0}


def physical_gt(scene, kernel, t, samples=512):
    offs = np.linspace(-kernel.exposure / 2, kernel.exposure / 2, samples)
    gt = np.zeros((scene.height, scene.width, 3))
    for tau in offs:
        gt += scene.render(t + tau)[0]
    return np.clip(gt / samples, 0.0, 1.0)


def run():
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    video_frames = []

    for name in ("constant", "turning", "occlusion"):
        scene = make_scene(name)
        t = FRAME_INDEX * DT
        kernel = ShutterKernel.from_angle(ANGLE[name], DT, num_samples=32)
        motion = SceneMotionField(scene)
        provider = SceneVisibilityProvider(scene)
        clean, _ = scene.render(t)
        gt = physical_gt(scene, kernel, t)
        roi = motion_roi(gt, clean)
        corner_static = static_roi(clean, scene.render(t + DT)[0])

        scene_metrics = {}
        images = {"clean": clean, "ground_truth": gt}
        result = None
        for method in METHODS:
            t0 = time.time()
            result = render_shutter(clean, motion, kernel, t, method, provider)
            elapsed = time.time() - t0
            img = result.image
            images[method] = img
            scene_metrics[method] = {
                "rmse": rmse(img, gt, roi),
                "contour_energy": contour_energy(img, roi),
                "static_brightness_ratio": float(
                    img[corner_static].mean() /
                    max(clean[corner_static].mean(), 1e-9)),
                "seconds_per_frame": elapsed,
            }

        gauss_sigma = max(2.0, 0.5 * kernel.exposure * 140.0)
        gauss = gaussian_blur(clean, gauss_sigma)
        images["gaussian_baseline"] = gauss
        scene_metrics["gaussian_baseline"] = {
            "rmse": rmse(gauss, gt, roi),
            "contour_energy": contour_energy(gauss, roi),
            "static_brightness_ratio": float(
                gauss[corner_static].mean() /
                max(clean[corner_static].mean(), 1e-9)),
        }
        scene_metrics["ground_truth"] = {
            "contour_energy": contour_energy(gt, roi)}

        # occlusion segment count (exposes split behaviour)
        points = pixel_grid(scene.height, scene.width)
        seg = segment_count_map(provider, points, t, kernel, motion)
        scene_metrics["pixels_with_multiple_segments"] = int((seg > 1).sum())

        summary[name] = scene_metrics

        # ------------------------------------------------------------ frames
        for label, img in images.items():
            save_frame(img, FRAME_DIR / f"{name}_{label}.png")
        overlay = draw_paths_overlay(result.image, result.paths, every=20)
        overlay = draw_shutter_band(
            overlay, t, kernel.exposure, DT,
            [(FRAME_INDEX - 1) * DT, FRAME_INDEX * DT,
             (FRAME_INDEX + 1) * DT])
        save_frame(overlay, FRAME_DIR / f"{name}_paths_overlay.png")
        video_frames.append((name, overlay, images))

        # ------------------------------------------------------------- figure
        labels = ["clean", *METHODS, "gaussian_baseline", "ground_truth"]
        fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
        for ax, label in zip(axes.ravel(), labels):
            ax.imshow(images[label])
            tag = label
            if label in scene_metrics and "rmse" in scene_metrics[label]:
                tag += f"\nRMSE {scene_metrics[label]['rmse']:.4f}"
            ax.set_title(tag, fontsize=9)
            ax.axis("off")
        axes.ravel()[-1].axis("off")
        fig.suptitle(
            f"{name}: shutter {kernel.shutter_angle_deg:.0f} deg, "
            f"E={kernel.exposure * 1000:.1f} ms, dt={DT * 1000:.1f} ms",
            fontsize=11)
        fig.tight_layout()
        fig.savefig(CHART_DIR / f"{name}_comparison.png", dpi=110)
        plt.close(fig)

    # --------------------------------------------------------- exposure sweep
    sweep_fig, sweep_ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, name in zip(sweep_ax, ("constant", "turning", "occlusion")):
        scene = make_scene(name)
        t = FRAME_INDEX * DT
        angles = np.linspace(15, 360, 12)
        curves = {m: [] for m in (*METHODS, "gaussian_baseline")}
        edges = []
        for angle in angles:
            kernel = ShutterKernel.from_angle(float(angle), DT, 32)
            motion = SceneMotionField(scene)
            provider = SceneVisibilityProvider(scene)
            clean, _ = scene.render(t)
            gt = physical_gt(scene, kernel, t, samples=256)
            roi = motion_roi(gt, clean)
            edges.append(contour_energy(gt, roi))
            for method in METHODS:
                r = render_shutter(clean, motion, kernel, t, method, provider)
                curves[method].append(rmse(r.image, gt, roi))
            sigma = max(2.0, 0.5 * kernel.exposure * 140.0)
            curves["gaussian_baseline"].append(
                rmse(gaussian_blur(clean, sigma), gt, roi))
        for method, vals in curves.items():
            ax.plot(angles, vals, label=method,
                    linestyle="--" if method == "gaussian_baseline" else "-")
        ax.set_title(name)
        ax.set_xlabel("shutter angle (deg)")
        ax.set_ylabel("RMSE vs physical GT")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    sweep_fig.tight_layout()
    sweep_fig.savefig(CHART_DIR / "exposure_sweep.png", dpi=110)
    plt.close(sweep_fig)

    # ------------------------------------------------------------- video
    # a short clip of the occlusion scene through the time-split integrator
    scene = make_scene("occlusion")
    clip = []
    num_frames = 12
    kernel = ShutterKernel.from_angle(270.0, DT, num_samples=24)
    for fi in range(num_frames):
        t = fi * DT
        motion = SceneMotionField(scene)
        provider = SceneVisibilityProvider(scene)
        clean, _ = scene.render(t)
        r = render_shutter(clean, motion, kernel, t, "time_split", provider)
        clip.append(r.image)
    fps = int(round(1 / DT))
    write_video(clip, VIDEO_DIR / "occlusion_timesplit.mp4", fps, "mp4")
    write_video(clip, VIDEO_DIR / "occlusion_timesplit.gif", fps, "gif")
    # clean reference clip so the difference is visible side by side
    clean_clip = [scene.render(fi * DT)[0] for fi in range(num_frames)]
    write_video(clean_clip, VIDEO_DIR / "occlusion_clean.mp4", fps, "mp4")

    with open(OUT / "benchmark_metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print("\nWrote frames to", FRAME_DIR)
    print("Wrote videos to", VIDEO_DIR)
    print("Wrote charts to", CHART_DIR)


if __name__ == "__main__":
    run()

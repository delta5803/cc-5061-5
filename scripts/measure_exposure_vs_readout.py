"""Empirical choice: vary the exposure window vs reorder readout timing.

Two physically different controls are often conflated as "more blur":

* **exposure sweep** - keep one fixed scene and fixed readout timing, change
  only the exposure window E (shutter angle).  This changes how long each row
  integrates, i.e. the *motion smear* length, while every row shares the same
  instant.
* **readout reorder** - keep E fixed and reorder the per-row readout timing
  (readout duration R and direction).  Each row stays sharp for the same E,
  but the exposure centres sweep across the frame, producing *progressive
  rolling distortion* (skew) rather than longer smear.

This script renders both sweeps on the same dial scene and measures, per
variant, the smear footprint (pixels changed versus the clean frame) and the
row-to-row skew (horizontal displacement of the dial centre across rows).
The measured numbers - not an assertion - decide which control explains which
effect, and the table is written to JSON/CSV next to the demo assets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shutter_blur import (  # noqa: E402
    ShutterKernel,
    SceneMotionField,
    make_scene,
    rmse,
)
from shutter_blur.integrators import SceneVisibilityProvider  # noqa: E402
from shutter_blur.rolling import render_rolling  # noqa: E402

DT = 1.0 / 24.0
FRAME_INDEX = 4
NUM_SAMPLES = 32


def _row_angular_shifts_vs_reference(image, reference_image, scene):
    """Per-row angular lag of the tick ruler against a global-shutter frame.

    The 60-tick rim is a periodic angular modulation: taking the Fourier
    component at the tick harmonic of each (polar-resampled) row band and
    dividing its phase by the reference's phase gives a sub-pixel-equivalent
    angular shift per row.  A global shutter has the same phase on every row;
    a rolling shutter's per-row exposure centres sweep, so the phase rotates
    monotonically down the rows.  Converted to horizontal pixels at the dial
    centre line, this is the skew the old lag-0 cross-correlation could not see
    against the symmetric radial spoke pattern.
    Returns ``(shifts_px, rows)``.
    """
    geo = scene.dial_geometry
    cx, cy = geo["center"]
    r_mid = 0.5 * (geo["tick_inner"] + geo["tick_outer"])
    harmonic = geo["tick_count"]
    h, w = image.shape[:2]
    rows = np.where(np.abs(np.arange(h) - cy) <= geo["radius"] * 0.8)[0]

    def phase_row(img, y):
        x0, x1 = max(0, int(cx - r_mid) - 2), min(w, int(cx + r_mid) + 3)
        xs = np.arange(x0, x1, dtype=np.float64) + 0.5
        band = img[y, x0:x1].mean(axis=1)
        band = band - band.mean()
        if (band ** 2).sum() < 1e-9:
            return None
        angles = np.arctan2(y + 0.5 - cy, xs - cx)
        return np.sum(band * np.exp(-1j * harmonic * angles))

    ref_phase = {}
    for y in rows:
        c = phase_row(reference_image, int(y))
        if c is not None:
            ref_phase[int(y)] = np.angle(c)

    shifts, used = [], []
    for y in rows:
        if int(y) not in ref_phase:
            continue
        c = phase_row(image, int(y))
        if c is None:
            continue
        dtheta = np.angle(np.exp(1j * (np.angle(c) - ref_phase[int(y)])))
        # arc length at the ruler radius, projected onto the horizontal axis
        shifts.append(float(dtheta * r_mid))
        used.append(int(y))
    return np.array(shifts), np.array(used)


def _dial_row_centres(image, scene, t_center, reference_image=None):
    """Rolling-skew probe: standard deviation and signed slope of row shifts.

    Row shifts come from the tick-ruler Fourier phase against a global-shutter
    reference, so they measure only the progressive readout distortion and
    stay sub-pixel accurate.  Without a reference the probe cannot isolate the
    rolling component and returns zeros.
    """
    if reference_image is None:
        return 0.0, 0.0
    shifts, rows = _row_angular_shifts_vs_reference(
        image, reference_image, scene)
    if len(shifts) < 8:
        return 0.0, 0.0
    slope = float(np.polyfit(rows, shifts, 1)[0])
    return float(np.std(shifts)), slope


def _smear_footprint(image, clean, threshold=0.02):
    return int((np.abs(image - clean).mean(axis=2) > threshold).sum())


def run(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = make_scene("dial")
    t_center = FRAME_INDEX * DT
    clean, _ = scene.render(t_center)
    motion = SceneMotionField(scene)
    provider = SceneVisibilityProvider(scene)
    global_kernel = ShutterKernel.from_angle(
        180.0, DT, NUM_SAMPLES, readout_duration=0.0)
    global_ref = render_rolling(
        clean, motion, global_kernel, t_center, provider).image

    report = {
        "frame_interval": DT,
        "t_center": t_center,
        "scene": "dial",
        "exposure_window_sweep": [],
        "readout_reorder_sweep": [],
    }

    # A) fixed scene + fixed (global) readout, vary only the exposure window
    for angle in (30.0, 90.0, 180.0, 270.0, 360.0):
        kernel = ShutterKernel.from_angle(
            angle, DT, NUM_SAMPLES, readout_duration=0.0)
        img = render_rolling(clean, motion, kernel, t_center, provider).image
        skew, _ = _dial_row_centres(img, scene, t_center, global_ref)
        report["exposure_window_sweep"].append({
            "shutter_angle_deg": angle,
            "exposure_s": kernel.exposure,
            "readout_duration_s": 0.0,
            "smear_footprint_px": _smear_footprint(img, clean),
            "row_skew_std_px": skew,
            "rmse_vs_clean": rmse(img, clean),
        })

    # B) fixed exposure E (180 deg), reorder readout timing only
    for frac in (0.0, 0.25, 0.5, 0.75, 0.95):
        kernel = ShutterKernel.from_angle(
            180.0, DT, NUM_SAMPLES, readout_duration=frac * DT)
        img = render_rolling(clean, motion, kernel, t_center, provider).image
        skew, slope = _dial_row_centres(img, scene, t_center, global_ref)
        report["readout_reorder_sweep"].append({
            "shutter_angle_deg": 180.0,
            "exposure_s": kernel.exposure,
            "readout_duration_s": kernel.readout_duration,
            "readout_frac_of_dt": frac,
            "smear_footprint_px": _smear_footprint(img, clean),
            "row_skew_std_px": skew,
            "row_skew_slope_px_per_row": slope,
            "rmse_vs_clean": rmse(img, clean),
        })

    # C) readout-direction reversal at fixed E and R with the exposure sweep
    # held fixed: the skew slope sign must reverse while the smear footprint
    # stays essentially unchanged
    direction = {}
    for d in ("top_down", "bottom_up"):
        kernel = ShutterKernel.from_angle(
            180.0, DT, NUM_SAMPLES, readout_duration=0.9 * DT,
            readout_direction=d, exposure_direction="top_down")
        img = render_rolling(clean, motion, kernel, t_center, provider).image
        skew, slope = _dial_row_centres(img, scene, t_center, global_ref)
        direction[d] = {
            "row_skew_std_px": skew,
            "row_skew_slope_px_per_row": slope,
            "smear_footprint_px": _smear_footprint(img, clean),
        }
    report["direction_reversal"] = direction

    # decision derived from the measured table (data, not a hard-coded claim)
    exp = report["exposure_window_sweep"]
    ro = report["readout_reorder_sweep"]
    smear_grows = exp[-1]["smear_footprint_px"] > exp[0]["smear_footprint_px"]
    skew_grows = ro[-1]["row_skew_std_px"] > ro[0]["row_skew_std_px"] + 0.05
    smear_flat_under_readout = (
        abs(ro[-1]["smear_footprint_px"] - ro[0]["smear_footprint_px"])
        < 0.25 * max(ro[0]["smear_footprint_px"], 1))
    slope_reverses = (
        direction["top_down"]["row_skew_slope_px_per_row"]
        * direction["bottom_up"]["row_skew_slope_px_per_row"] < 0.0)
    report["measured_decision"] = {
        "exposure_window_controls_smear": bool(smear_grows),
        "readout_reorder_controls_skew": bool(skew_grows),
        "readout_leaves_smear_footprint_nearly_flat":
            bool(smear_flat_under_readout),
        "direction_reversal_flips_skew_sign": bool(slope_reverses),
        "selection": (
            "vary exposure window for motion smear; reorder readout timing "
            "for progressive skew"
            if smear_grows and skew_grows else
            "inconclusive: inspect the measured table"),
    }

    json_path = out_dir / "exposure_vs_readout.json"
    json_path.write_text(json.dumps(report, indent=2))
    csv_path = out_dir / "exposure_vs_readout.csv"
    with open(csv_path, "w") as fh:
        fh.write("sweep,shutter_angle_deg,exposure_s,readout_duration_s,"
                 "smear_footprint_px,row_skew_std_px,rmse_vs_clean\n")
        for sweep_key in ("exposure_window_sweep", "readout_reorder_sweep"):
            for row in report[sweep_key]:
                fh.write(
                    f"{sweep_key},{row['shutter_angle_deg']},"
                    f"{row['exposure_s']:.6f},"
                    f"{row.get('readout_duration_s', 0.0):.6f},"
                    f"{row['smear_footprint_px']},"
                    f"{row['row_skew_std_px']:.5f},"
                    f"{row['rmse_vs_clean']:.6f}\n")
    return report


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "outputs" / "shutter_demo"
    data = run(out)
    print(json.dumps(data["measured_decision"], indent=2))
    print("wrote", out / "exposure_vs_readout.json")

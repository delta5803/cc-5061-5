"""Stdlib HTTP server for the shutter-smear motion-analysis player.

Run:
    python3 webapp/server.py
then open http://127.0.0.1:8000 in a browser.

The control panel is driven by ``ShutterKernel.ui_schema()`` rather than by
hard-coded markup: changing the kernel's control set changes the UI. Invalid
shutter parameters raise ``ShutterError`` and are returned as structured JSON
so the page can surface the failure inside the shutter panel.
"""
from __future__ import annotations

import io
import json
import sys
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shutter_blur import (  # noqa: E402
    METHODS,
    SCENES,
    ShutterError,
    ShutterKernel,
    contour_energy,
    make_scene,
    motion_roi,
    rmse,
)
from shutter_blur.assets import load_asset  # noqa: E402
from shutter_blur.integrators import (  # noqa: E402
    SceneVisibilityProvider,
    segment_count_map,
)
from shutter_blur.io import (  # noqa: E402
    draw_paths_overlay,
    draw_shutter_band,
    to_uint8,
)
from shutter_blur.kernel import DIRECTIONS, PROFILES  # noqa: E402
from shutter_blur.motion import SceneMotionField  # noqa: E402
from shutter_blur.rolling import (  # noqa: E402
    render_rolling,
    rolling_ground_truth,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "outputs" / "shutter_demo" / "assets"
DT = 1.0 / 24.0
NUM_FRAMES = 12
CACHE_LIMIT = 24
GT_SAMPLES = 64


class SceneCache:
    """One scene + its providers, and a small LRU of rendered frame bundles."""

    def __init__(self):
        self._scenes = {name: make_scene(name) for name in SCENES}
        self._motion = {}
        self._provider = {}
        self._frames = OrderedDict()
        self._lock = threading.Lock()

    def scene(self, name):
        if name not in self._scenes:
            raise ShutterError(f"unknown scene {name!r}")
        return self._scenes[name]

    def motion(self, name):
        if name not in self._motion:
            self._motion[name] = SceneMotionField(self._scenes[name])
        return self._motion[name]

    def provider(self, name):
        if name not in self._provider:
            self._provider[name] = SceneVisibilityProvider(self._scenes[name])
        return self._provider[name]

    def render(self, name, frame_index, method, angle, num_samples,
               readout_frac=0.0, readout_direction="top_down",
               exposure_direction="top_down", exposure_profile="box",
               show_paths=True):
        scene = self.scene(name)
        kernel = ShutterKernel.from_angle(
            float(angle), DT, int(num_samples),
            readout_duration=float(readout_frac) * DT,
            readout_direction=readout_direction,
            exposure_direction=exposure_direction,
            exposure_profile=exposure_profile)
        key = (name, frame_index, method, round(angle, 4), num_samples,
               round(readout_frac, 4), readout_direction,
               exposure_direction, exposure_profile)
        with self._lock:
            if key in self._frames:
                self._frames.move_to_end(key)
                bundle = self._frames[key]
            else:
                t = frame_index * DT
                clean, _ = scene.render(t)
                result = _render_method(
                    scene, self.motion(name), self.provider(name),
                    kernel, t, method, clean)
                gt = _rolling_ref(scene, kernel, t)
                roi = motion_roi(gt, clean)
                seg = segment_count_map(
                    self.provider(name), _grid(scene), t, kernel,
                    self.motion(name))
                bundle = {
                    "image": result.image,
                    "clean": clean,
                    "gt": gt,
                    "paths": result.paths,
                    "visibility": result.visibility,
                    "weight_map": result.weight_map,
                    "t_center": t,
                    "kernel": kernel,
                    "rmse": float(rmse(result.image, gt, roi)),
                    "contour_energy": float(
                        contour_energy(result.image, roi)),
                    "segments_multi": int((seg > 1).sum()),
                    "is_rolling": kernel.is_rolling,
                    "sample_weights": kernel.weights(),
                    "row_offsets": kernel.exposure_row_offsets(scene.height),
                }
                self._frames[key] = bundle
                self._frames.move_to_end(key)
                while len(self._frames) > CACHE_LIMIT:
                    self._frames.popitem(last=False)
        return dict(bundle)


def _grid(scene):
    yy, xx = np.mgrid[0: scene.height, 0: scene.width]
    return np.stack([xx + 0.5, yy + 0.5], axis=-1)


def _render_method(scene, motion, provider, kernel, t, method, clean):
    if method == "rolling":
        return render_rolling(clean, motion, kernel, t, provider)
    from shutter_blur.integrators import render_shutter
    return render_shutter(clean, motion, kernel, t, method, provider)


def _rolling_ref(scene, kernel, t):
    # one reference implementation for global and rolling: R=0 reduces it to
    # the fixed-photosite global shutter integral
    return rolling_ground_truth(scene, kernel, t, GT_SAMPLES)


CACHE = SceneCache()


def _png_bytes(rgb) -> bytes:
    u8 = to_uint8(rgb)
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format="PNG")
    return buf.getvalue()


def _build_overlay(bundle, frame_index, show_paths):
    img = bundle["image"]
    if show_paths:
        overlay = draw_paths_overlay(img, bundle["paths"], every=24)
        overlay = draw_shutter_band(
            overlay, bundle["t_center"], bundle["kernel"].exposure, DT,
            [(frame_index - 1) * DT, frame_index * DT,
             (frame_index + 1) * DT])
        return overlay
    return (img * 255).round().astype(np.uint8)


class Handler(BaseHTTPRequestHandler):
    server_version = "ShutterPlayer/1.0"

    def log_message(self, fmt, *args):  # quieter console
        sys.stderr.write("[player] %s\n" % (fmt % args))

    # ------------------------------------------------------------- helpers
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_png(self, rgb):
        body = _png_bytes(rgb)
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body, ctype, filename):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Disposition",
                         f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ----------------------------------------------------------------- GET
    def do_GET(self):
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                return self._serve_file("index.html", "text/html")
            if path == "/app.js":
                return self._serve_file("app.js", "application/javascript")
            if path == "/api/config":
                return self._config()
            if path == "/api/assets":
                return self._assets()
            if path == "/api/render":
                return self._render(qs)
            if path == "/api/frame":
                return self._frame(qs)
            if path == "/api/export":
                return self._export(qs)
            self._send_json({"error": "not found"}, 404)
        except ShutterError as exc:
            self._send_json({"error": str(exc), "type": "ShutterError"}, 400)
        except (ValueError, KeyError) as exc:
            self._send_json({"error": f"bad parameter: {exc}"}, 400)

    def _serve_file(self, name, ctype):
        target = STATIC_DIR / name
        if not target.is_file():
            return self._send_json({"error": f"{name} missing"}, 404)
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _one(self, qs, key, default):
        vals = qs.get(key, [default])
        return vals[0]

    def _config(self):
        sample_kernel = ShutterKernel.from_angle(180.0, DT, 32)
        self._send_json({
            "scenes": sorted(SCENES),
            "methods": ["rolling", *METHODS],
            "profiles": list(PROFILES),
            "directions": list(DIRECTIONS),
            "frame_interval": DT,
            "num_frames": NUM_FRAMES,
            "ui_schema": sample_kernel.ui_schema(),
            "defaults": {
                "scene": "dial",
                "frame_index": 4,
                "method": "rolling",
                "shutter_angle_deg": 180.0,
                "num_samples": 32,
                "readout_frac": 0.9,
                "readout_direction": "top_down",
                "exposure_direction": "top_down",
                "exposure_profile": "trapezoid",
                "show_paths": True,
            },
        })

    def _params(self, qs):
        return {
            "scene": self._one(qs, "scene", "dial"),
            "frame_index": int(self._one(qs, "frame", 4)),
            "method": self._one(qs, "method", "rolling"),
            "angle": float(self._one(qs, "angle", 180.0)),
            "num_samples": int(self._one(qs, "samples", 32)),
            "readout_frac": float(self._one(qs, "readout", 0.9)),
            "readout_direction": self._one(
                qs, "readout_dir", "top_down"),
            "exposure_direction": self._one(
                qs, "exposure_dir", "top_down"),
            "exposure_profile": self._one(qs, "profile", "trapezoid"),
            "show_paths": self._one(qs, "paths", "1") != "0",
        }

    def _validate(self, p):
        if not 0 <= p["frame_index"] < NUM_FRAMES:
            raise ShutterError(
                f"frame index {p['frame_index']} outside [0, {NUM_FRAMES - 1}]")
        all_methods = ["rolling", *METHODS]
        if p["method"] not in all_methods:
            raise ShutterError(f"unknown method {p['method']!r}")
        if p["readout_direction"] not in DIRECTIONS:
            raise ShutterError(
                f"readout_direction must be one of {DIRECTIONS}")
        if p["exposure_direction"] not in DIRECTIONS:
            raise ShutterError(
                f"exposure_direction must be one of {DIRECTIONS}")
        if p["exposure_profile"] not in PROFILES:
            raise ShutterError(
                f"exposure_profile must be one of {PROFILES}")
        if not 0.0 <= p["readout_frac"] <= 1.0:
            raise ShutterError("readout fraction must lie in [0, 1]")
        kernel = ShutterKernel.from_angle(
            p["angle"], DT, p["num_samples"],
            readout_duration=p["readout_frac"] * DT,
            readout_direction=p["readout_direction"],
            exposure_direction=p["exposure_direction"],
            exposure_profile=p["exposure_profile"])  # raises ShutterError
        return kernel

    def _render(self, qs):
        p = self._params(qs)
        kernel = self._validate(p)
        bundle = CACHE.render(
            p["scene"], p["frame_index"], p["method"], p["angle"],
            p["num_samples"], p["readout_frac"], p["readout_direction"],
            p["exposure_direction"], p["exposure_profile"], p["show_paths"])
        self._send_json({
            "image_png": "/api/frame?" + parsed_render_query(qs, p),
            "metrics": {
                "rmse_vs_gt": round(bundle["rmse"], 6),
                "contour_energy": round(bundle["contour_energy"], 6),
                "pixels_with_multiple_segments": bundle["segments_multi"],
                "mean_sample_weight": float(
                    np.mean(bundle["weight_map"])),
            },
            "kernel": {
                "shutter_angle_deg": kernel.shutter_angle_deg,
                "exposure": kernel.exposure,
                "frame_interval": kernel.frame_interval,
                "num_samples": kernel.num_samples,
                "readout_duration": kernel.readout_duration,
                "readout_direction": kernel.readout_direction,
                "exposure_direction": kernel.exposure_direction,
                "exposure_profile": kernel.exposure_profile,
                "is_rolling": kernel.is_rolling,
                "row_offset_first": float(bundle["row_offsets"][0]),
                "row_offset_last": float(bundle["row_offsets"][-1]),
            },
            "profile_weights": [float(x) for x in bundle["sample_weights"]],
            "t_center": bundle["t_center"],
            "overlay_ready": True,
        })

    def _frame(self, qs):
        p = self._params(qs)
        self._validate(p)
        bundle = CACHE.render(
            p["scene"], p["frame_index"], p["method"], p["angle"],
            p["num_samples"], p["readout_frac"], p["readout_direction"],
            p["exposure_direction"], p["exposure_profile"], p["show_paths"])
        overlay = _build_overlay(bundle, p["frame_index"], p["show_paths"])
        self._send_png(overlay)

    def _assets(self):
        """List the on-disk shutter demo assets (must physically exist)."""
        items = []
        if ASSET_ROOT.exists():
            for manifest_path in sorted(ASSET_ROOT.glob("*/manifest.json")):
                bundle = load_asset(manifest_path.parent)
                manifest = bundle["manifest"]
                traj = bundle["trajectory"]
                items.append({
                    "name": manifest["spec"]["name"],
                    "dir": str(manifest_path.parent.relative_to(ROOT)),
                    "variants": list(manifest["kernels"].keys()),
                    "trajectory_samples": int(len(traj["time"])),
                    "frames": manifest["files"]["frames"],
                    "hashes_ok": self._verify_hashes(manifest_path.parent,
                                                     manifest),
                })
        self._send_json({"assets": items})

    @staticmethod
    def _verify_hashes(asset_dir, manifest):
        ok = {}
        for name, digest in manifest.get("file_hashes", {}).items():
            if name == "manifest.json":
                continue
            target = Path(asset_dir) / name
            import hashlib
            ok[name] = (target.is_file()
                        and hashlib.sha256(target.read_bytes()).hexdigest()
                        == digest)
        return ok

    def _export(self, qs):
        from shutter_blur.io import write_video
        p = self._params(qs)
        self._validate(p)
        kind = self._one(qs, "kind", "mp4")
        frames = []
        for fi in range(NUM_FRAMES):
            bundle = CACHE.render(
                p["scene"], fi, p["method"], p["angle"], p["num_samples"],
                p["readout_frac"], p["readout_direction"],
                p["exposure_direction"], p["exposure_profile"],
                show_paths=False)
            frames.append(bundle["image"])
        buf = io.BytesIO()
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=f".{kind}",
                                         delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            write_video(frames, tmp_path, int(round(1 / DT)), kind)
            body = tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)
        ctype = ("video/mp4" if kind == "mp4" else "image/gif")
        self._send_bytes(
            body, ctype, f"shutter_{p['scene']}_{p['method']}.{kind}")


def parsed_render_query(qs, p):
    from urllib.parse import urlencode
    return urlencode({
        "scene": p["scene"], "frame": p["frame_index"],
        "method": p["method"], "angle": p["angle"],
        "samples": p["num_samples"],
        "readout": p["readout_frac"],
        "readout_dir": p["readout_direction"],
        "exposure_dir": p["exposure_direction"],
        "profile": p["exposure_profile"],
        "paths": "1" if p["show_paths"] else "0",
    })


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    host, port = "127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    httpd = ReusableThreadingHTTPServer((host, port), Handler)
    print(f"Shutter player at http://{host}:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
        httpd.server_close()


if __name__ == "__main__":
    main()

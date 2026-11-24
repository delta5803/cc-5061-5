"""Frame and video IO (PNG via Pillow, MP4/GIF via imageio + static ffmpeg)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB array to uint8, accepting [0,1] float or [0,255] uint8."""
    if np.issubdtype(rgb.dtype, np.integer):
        return rgb.astype(np.uint8)
    return np.clip(rgb, 0.0, 1.0).__mul__(255.0).round().astype(np.uint8)


def save_frame(rgb: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(rgb)).save(path)
    return path


def load_frame(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def write_video(frames: list[np.ndarray], path: str | Path,
                fps: float, kind: str = "mp4") -> Path:
    """Write an MP4 (H.264 via bundled ffmpeg) or animated GIF."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = [to_uint8(f) for f in frames]
    if kind == "gif":
        imageio.mimsave(path, u8, fps=fps, loop=0)
    else:
        writer = imageio.get_writer(
            path, fps=fps, codec="libx264", quality=7,
            macro_block_size=None, output_params=["-pix_fmt", "yuv420p"])
        for frame in u8:
            writer.append_data(frame)
        writer.close()
    return path


def draw_paths_overlay(frame: np.ndarray, paths: np.ndarray,
                       sample_step: int = 2,
                       color=(255, 80, 60),
                       every: int = 24) -> np.ndarray:
    """Draw shutter sampling paths on top of a frame for the result panel.

    ``paths`` has shape (N, H, W, 2); one polyline every ``every`` pixels.
    """
    canvas = to_uint8(frame).copy()
    n, h, w, _ = paths.shape
    color = np.array(color, dtype=np.uint8)
    for y in range(every // 2, h, every):
        for x in range(every // 2, w, every):
            line = paths[::sample_step, y, x]
            for p0, p1 in zip(line[:-1], line[1:]):
                _draw_line(canvas, p0, p1, color)
            for p in line:
                _draw_dot(canvas, p, color)
    return canvas


def draw_shutter_band(canvas: np.ndarray, t_center: float,
                      exposure: float, frame_interval: float,
                      frame_times: list[float]) -> np.ndarray:
    """Annotate the exposure band against frame-centre tick marks."""
    from PIL import ImageDraw

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    h, w = canvas.shape[:2]
    t0, t1 = frame_times[0], frame_times[-1] + frame_interval

    def x_of(t):
        return int(12 + (t - t0) / (t1 - t0) * (w - 24))

    y0, y1 = h - 26, h - 8
    for ft in frame_times:
        x = x_of(ft)
        draw.line([(x, y0 - 4), (x, y1 + 4)], fill=(200, 200, 200))
    draw.rectangle([x_of(t_center - exposure / 2), y0,
                    x_of(t_center + exposure / 2), y1],
                   fill=(255, 120, 60))
    draw.line([(x_of(t_center), y0 - 6), (x_of(t_center), y1 + 6)],
              fill=(255, 255, 255), width=2)
    return np.asarray(img)


def _draw_line(canvas: np.ndarray, p0: np.ndarray, p1: np.ndarray,
               color: np.ndarray) -> None:
    h, w = canvas.shape[:2]
    x0, y0 = float(p0[0]) - 0.5, float(p0[1]) - 0.5
    x1, y1 = float(p1[0]) - 0.5, float(p1[1]) - 0.5
    steps = max(1, int(max(abs(x1 - x0), abs(y1 - y0))) + 1)
    for s in range(steps + 1):
        a = s / steps
        x = int(round(x0 + a * (x1 - x0)))
        y = int(round(y0 + a * (y1 - y0)))
        if 0 <= x < w and 0 <= y < h:
            canvas[y, x] = color


def _draw_dot(canvas: np.ndarray, p: np.ndarray,
              color: np.ndarray) -> None:
    h, w = canvas.shape[:2]
    x, y = int(round(float(p[0]) - 0.5)), int(round(float(p[1]) - 0.5))
    for dy, dx in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        xx, yy = x + dx, y + dy
        if 0 <= xx < w and 0 <= yy < h:
            canvas[yy, xx] = color

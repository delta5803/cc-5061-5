"""Image-domain metrics for motion-blur evaluation."""
from __future__ import annotations

import numpy as np


def rmse(a: np.ndarray, b: np.ndarray, roi: np.ndarray | None = None) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    if roi is not None:
        diff = diff[roi]
    return float(np.sqrt(np.mean(diff ** 2)))


def mean_abs_error(a: np.ndarray, b: np.ndarray,
                   roi: np.ndarray | None = None) -> float:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if roi is not None:
        diff = diff[roi]
    return float(diff.mean())


def _sobel_magnitude(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.mean(axis=2) if rgb.ndim == 3 else rgb
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:-1] = gray[2:] - gray[:-2]
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    return np.hypot(gx, gy)


def contour_energy(rgb: np.ndarray, roi: np.ndarray | None = None) -> float:
    """Total edge energy (Sobel magnitude) inside the region of interest."""
    mag = _sobel_magnitude(rgb)
    if roi is not None:
        mag = mag[roi]
    return float(mag.sum())


def motion_roi(gt: np.ndarray, clean: np.ndarray,
               threshold: float = 0.02) -> np.ndarray:
    """ROI covering every pixel the exposure actually changed (dilated)."""
    change = np.abs(gt - clean).mean(axis=2) > threshold
    roi = change.copy()
    roi[1:] |= change[:-1]
    roi[:-1] |= change[1:]
    roi[:, 1:] |= change[:, :-1]
    roi[:, :-1] |= change[:, 1:]
    return roi


def blur_profile_width(image: np.ndarray, row: int, x_center: int,
                       axis: int = 0) -> float:
    """Width (px) of the moving-edge profile at 10%-90% intensity transition."""
    gray = image.mean(axis=2)
    profile = gray[row, :] if axis == 0 else gray[:, x_center]
    lo, hi = profile.min(), profile.max()
    if hi - lo < 1e-6:
        return 0.0
    low = lo + 0.1 * (hi - lo)
    high = lo + 0.9 * (hi - lo)
    idx = np.arange(len(profile))
    try:
        x1 = idx[profile >= low][0]
        x2 = idx[profile >= high][0]
    except IndexError:
        return 0.0
    return float(abs(x2 - x1))


def energy_ratio(blurred: np.ndarray, clean: np.ndarray,
                 static_roi: np.ndarray) -> float:
    """Mean brightness ratio blurred/clean in a static region (ideal == 1)."""
    b = blurred[static_roi].mean()
    c = clean[static_roi].mean()
    if c < 1e-8:
        return float("inf")
    return float(b / c)


def static_roi(clean_a: np.ndarray, clean_b: np.ndarray,
               margin: int = 8) -> np.ndarray:
    """Corner strips that never move between two clean frames."""
    h, w = clean_a.shape[:2]
    moving = np.abs(clean_a - clean_b).mean(axis=2) > 0.01
    static = ~moving
    static[margin:h - margin, margin:w - margin] = False
    return static

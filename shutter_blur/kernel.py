"""Shutter kernel: exposure time, frame interval and normalized sample weights.

The exposure duration E and the frame interval dt are represented separately.
Pixel displacement over a shutter must never be confused with exposure time:
a pixel's displacement is velocity * E (a length), while E is a duration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ShutterError(ValueError):
    """Raised when shutter parameters are physically invalid."""


@dataclass(frozen=True)
class ShutterKernel:
    """A centered shutter opening on a temporal grid.

    Parameters
    ----------
    exposure:
        Exposure duration E in seconds (time the shutter is open).
    frame_interval:
        Time dt between two consecutive frame centres, in seconds.
    num_samples:
        Number of point samples placed inside the exposure interval.
    shutter_angle_deg:
        Film-camera shutter angle theta with E/dt = theta/360.  Accepted as an
        alternative to ``exposure``; the two are never silently mixed.
    """

    exposure: float
    frame_interval: float
    num_samples: int = 16

    def __post_init__(self) -> None:
        if self.frame_interval <= 0:
            raise ShutterError("frame_interval must be positive")
        if self.exposure <= 0:
            raise ShutterError("exposure duration must be positive")
        if self.exposure > self.frame_interval + 1e-12:
            raise ShutterError(
                f"exposure {self.exposure:g}s exceeds frame interval "
                f"{self.frame_interval:g}s (shutter angle > 360 deg)"
            )
        if self.num_samples < 2:
            raise ShutterError("need at least 2 samples across the shutter")

    @classmethod
    def from_angle(
        cls, angle_deg: float, frame_interval: float, num_samples: int = 16
    ) -> "ShutterKernel":
        if not 0 < angle_deg <= 360:
            raise ShutterError("shutter angle must lie in (0, 360] degrees")
        return cls(
            exposure=angle_deg / 360.0 * frame_interval,
            frame_interval=frame_interval,
            num_samples=num_samples,
        )

    @property
    def shutter_angle_deg(self) -> float:
        return 360.0 * self.exposure / self.frame_interval

    def sample_offsets(self) -> np.ndarray:
        """Temporal offsets tau in [-E/2, E/2] of every sample (in seconds)."""
        n = self.num_samples
        return self.exposure * (np.arange(n, dtype=np.float64) / (n - 1) - 0.5)

    def weights(self) -> np.ndarray:
        """Normalized sample weights summing to one.

        Equal weights are used, but normalization is explicit so static areas
        keep the same brightness whether 8 or 256 samples are taken.
        """
        w = np.full(self.num_samples, 1.0 / self.num_samples, dtype=np.float64)
        return w

    def ui_schema(self) -> list[dict]:
        """Schema consumed by the web player to build its controls."""
        return [
            {
                "name": "shutter_angle_deg",
                "label": "Shutter angle (deg)",
                "min": 1.0,
                "max": 360.0,
                "step": 1.0,
                "value": round(self.shutter_angle_deg, 3),
                "unit": "deg",
            },
            {
                "name": "exposure",
                "label": "Exposure duration",
                "min": 1e-4,
                "max": round(self.frame_interval, 6),
                "step": round(self.frame_interval / 360.0, 6),
                "value": round(self.exposure, 6),
                "unit": "s",
            },
            {
                "name": "frame_interval",
                "label": "Frame interval (fixed)",
                "min": self.frame_interval,
                "max": self.frame_interval,
                "step": 0.0,
                "value": self.frame_interval,
                "unit": "s",
                "readonly": True,
            },
            {
                "name": "num_samples",
                "label": "Samples across shutter",
                "min": 2,
                "max": 128,
                "step": 1,
                "value": self.num_samples,
                "unit": "",
            },
        ]

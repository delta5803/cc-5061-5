"""Shutter kernel: exposure time, frame interval, readout timing and weights.

The exposure duration E and the frame interval dt are represented separately.
Pixel displacement over a shutter must never be confused with exposure time:
a pixel's displacement is velocity * E (a length), while E is a duration.

Two independent time axes are modelled:

* **exposure** - when each row's shutter is open (duration ``exposure``), with
  a selectable exposure *profile* (box / trapezoid / gaussian) that sets the
  temporal weight of every sample;
* **readout** - when each row is sampled relative to the frame centre
  (``readout_duration`` spread top-to-bottom or bottom-to-top).  A zero
  readout duration is a pure global shutter; any positive value makes the
  per-row exposure windows overlap only partially, which is what produces
  rolling-shutter skew as distinct from motion smear.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ShutterError(ValueError):
    """Raised when shutter parameters are physically invalid."""


PROFILES = ("box", "trapezoid", "gaussian")
DIRECTIONS = ("top_down", "bottom_up")


def profile_window(profile: str, u: np.ndarray) -> np.ndarray:
    """Exposure profile weight on the normalized window coordinate u in [-1, 1].

    Outside the window the weight is exactly zero.  ``box`` is the classic
    instantaneous open/close shutter; ``trapezoid`` ramps the opening over
    20% of the window on each side; ``gaussian`` is a 2-sigma bell whose tails
    are cut at the window edges.
    """
    u = np.asarray(u, dtype=np.float64)
    if profile == "box":
        return np.where(np.abs(u) <= 1.0, 1.0, 0.0)
    if profile == "trapezoid":
        ramp = 0.2
        return np.clip((1.0 - np.abs(u)) / ramp, 0.0, 1.0)
    if profile == "gaussian":
        sigma = 0.5
        return np.where(np.abs(u) <= 1.0,
                        np.exp(-0.5 * (u / sigma) ** 2), 0.0)
    raise ShutterError(f"unknown exposure profile {profile!r}; "
                       f"choose from {PROFILES}")


@dataclass(frozen=True)
class ShutterKernel:
    """A centered shutter opening on a temporal grid.

    Parameters
    ----------
    exposure:
        Exposure duration E in seconds (time the shutter is open per row).
    frame_interval:
        Time dt between two consecutive frame centres, in seconds.
    num_samples:
        Number of point samples placed inside each exposure interval.
    shutter_angle_deg:
        Film-camera shutter angle theta with E/dt = theta/360.  Accepted as an
        alternative to ``exposure``; the two are never silently mixed.
    readout_duration:
        Wall-clock time R between the first and last row's exposure centre.
        R = 0 is a global shutter; 0 < R <= dt is a rolling shutter.
    readout_direction:
        ``"top_down"`` reads row 0 first; ``"bottom_up"`` reads the last row
        first.  Reorders readout timing without changing the exposure window.
    exposure_direction:
        Direction in which the per-row exposure windows sweep the frame.
        Kept separate from the readout direction so the two time axes can be
        flipped independently in the demo material.
    exposure_profile:
        Temporal sensitivity curve of the open shutter (see
        :func:`profile_window`).
    """

    exposure: float
    frame_interval: float
    num_samples: int = 16
    readout_duration: float = 0.0
    readout_direction: str = "top_down"
    exposure_direction: str = "top_down"
    exposure_profile: str = "box"

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
        if self.readout_duration < 0.0:
            raise ShutterError("readout_duration must be non-negative")
        if self.readout_duration > self.frame_interval + 1e-12:
            raise ShutterError(
                f"readout duration {self.readout_duration:g}s exceeds frame "
                f"interval {self.frame_interval:g}s")
        if self.readout_direction not in DIRECTIONS:
            raise ShutterError(
                f"readout_direction must be one of {DIRECTIONS}")
        if self.exposure_direction not in DIRECTIONS:
            raise ShutterError(
                f"exposure_direction must be one of {DIRECTIONS}")
        if self.exposure_profile not in PROFILES:
            raise ShutterError(
                f"exposure_profile must be one of {PROFILES}")

    @classmethod
    def from_angle(
        cls, angle_deg: float, frame_interval: float, num_samples: int = 16,
        readout_duration: float = 0.0, readout_direction: str = "top_down",
        exposure_direction: str = "top_down",
        exposure_profile: str = "box",
    ) -> "ShutterKernel":
        if not 0 < angle_deg <= 360:
            raise ShutterError("shutter angle must lie in (0, 360] degrees")
        return cls(
            exposure=angle_deg / 360.0 * frame_interval,
            frame_interval=frame_interval,
            num_samples=num_samples,
            readout_duration=readout_duration,
            readout_direction=readout_direction,
            exposure_direction=exposure_direction,
            exposure_profile=exposure_profile,
        )

    @property
    def shutter_angle_deg(self) -> float:
        return 360.0 * self.exposure / self.frame_interval

    @property
    def is_rolling(self) -> bool:
        return self.readout_duration > 0.0

    def sample_offsets(self) -> np.ndarray:
        """Temporal offsets tau in [-E/2, E/2] of every sample (in seconds)."""
        n = self.num_samples
        return self.exposure * (np.arange(n, dtype=np.float64) / (n - 1) - 0.5)

    def row_offsets(self, height: int) -> np.ndarray:
        """Exposure-centre offset (s) of every image row.

        Row ordering follows the readout direction: the first row digitised
        gets -R/2 and the last gets +R/2.  The independent exposure-direction
        axis is applied on top via :meth:`exposure_row_offsets`.
        """
        if height <= 1 or self.readout_duration == 0.0:
            return np.zeros(height, dtype=np.float64)
        frac = np.arange(height, dtype=np.float64) / (height - 1)
        if self.readout_direction == "bottom_up":
            frac = frac[::-1]
        return self.readout_duration * (frac - 0.5)

    def exposure_row_offsets(self, height: int) -> np.ndarray:
        """Per-row centre offset including the independent exposure sweep.

        The base sign comes from the readout direction (the reset/read order);
        the exposure-direction axis then either keeps that sweep or reverses
        only the exposure windows while the digitisation order in
        :meth:`readout_order` is untouched.
        """
        offsets = self.row_offsets(height)
        if self.readout_duration > 0.0 and \
                self.exposure_direction == "bottom_up":
            offsets = -offsets
        return offsets

    def readout_order(self, height: int) -> np.ndarray:
        """Row indices in the order they are digitised by the sensor."""
        rows = np.arange(height)
        if self.readout_direction == "bottom_up":
            return rows[::-1]
        return rows

    def profile_weights(self, offsets: np.ndarray | None = None) -> np.ndarray:
        """Normalized temporal sample weights implied by the exposure profile.

        The weights always sum to one so static regions keep identical
        brightness whatever profile or sample count is used.
        """
        if offsets is None:
            offsets = self.sample_offsets()
        u = np.asarray(offsets, dtype=np.float64) / (self.exposure / 2.0)
        w = profile_window(self.exposure_profile, u)
        total = w.sum()
        if total <= 0.0:
            raise ShutterError("exposure profile gave every sample zero weight")
        return w / total

    def weights(self) -> np.ndarray:
        """Normalized sample weights summing to one."""
        return self.profile_weights()

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

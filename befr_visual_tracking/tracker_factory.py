"""Factory for configurable visual tracking filters."""

from __future__ import annotations

from typing import Literal

from befr_visual_tracking.camera_model import Camera
from befr_visual_tracking.ekf_tracker import EkfTracker
from befr_visual_tracking.kalman_tracker import KalmanTracker
from befr_visual_tracking.tracker_base import TrackerBase
from befr_visual_tracking.ukf_tracker import UkfTracker

FilterType = Literal["kf", "ekf", "ukf"]


def _default_pixel_noise_std(cameras_by_id: dict[str, Camera]) -> float:
    camera = next(iter(cameras_by_id.values()))
    return 0.5 * (
        camera.noise_model.pixel_noise_std_u + camera.noise_model.pixel_noise_std_v
    )


def create_tracker(
    filter_type: str,
    cameras_by_id: dict[str, Camera],
    *,
    active_cameras: list[str] | None = None,
    acceleration_noise_std: float = 0.5,
    pixel_noise_std: float | None = None,
    innovation_gate: float = 9.21,
    monte_carlo_samples: int = 500,
    random_seed: int = 42,
) -> TrackerBase:
    """Create the selected tracker implementation."""
    normalized = filter_type.strip().lower()
    common_kwargs = {
        "acceleration_noise_std": acceleration_noise_std,
        "active_cameras": active_cameras,
        "monte_carlo_samples": monte_carlo_samples,
    }

    if normalized == "kf":
        import numpy as np

        resolved_pixel_noise = (
            pixel_noise_std
            if pixel_noise_std is not None
            else _default_pixel_noise_std(cameras_by_id)
        )
        return KalmanTracker(
            cameras_by_id,
            pixel_noise_std=resolved_pixel_noise,
            rng=np.random.default_rng(random_seed),
            **common_kwargs,
        )
    if normalized == "ekf":
        import numpy as np

        return EkfTracker(
            cameras_by_id,
            pixel_noise_std=pixel_noise_std,
            innovation_gate=innovation_gate,
            rng=np.random.default_rng(random_seed),
            **common_kwargs,
        )
    if normalized == "ukf":
        import numpy as np

        return UkfTracker(
            cameras_by_id,
            pixel_noise_std=pixel_noise_std,
            innovation_gate=innovation_gate,
            rng=np.random.default_rng(random_seed),
            **common_kwargs,
        )

    raise ValueError(f"Unsupported filter_type: {filter_type!r}. Expected 'kf', 'ekf', or 'ukf'.")

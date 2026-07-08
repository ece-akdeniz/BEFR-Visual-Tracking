"""Common tracker interface for KF, EKF, and UKF."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from befr_visual_tracking.motion_model import (
    STATE_DIM,
    normalize_initial_covariance,
    position_from_state,
    predict_state,
    state_from_position_velocity,
    velocity_from_state,
)


@dataclass(frozen=True)
class TrackerDetection:
    """Synchronized pixel detection passed to a tracker update step."""

    timestamp: float
    camera_id: str
    u: float
    v: float


class TrackerBase(ABC):
    """
    Shared interface for all visual tracking filters.

    State vector:
        x = [p_x, p_y, p_z, v_x, v_y, v_z]^T
    """

    def __init__(self, acceleration_noise_std: float = 0.5) -> None:
        self._acceleration_noise_std = float(acceleration_noise_std)
        self._state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._timestamp: float | None = None

    @property
    def acceleration_noise_std(self) -> float:
        return self._acceleration_noise_std

    @property
    def timestamp(self) -> float | None:
        return self._timestamp

    @property
    def is_initialized(self) -> bool:
        return self._state is not None and self._covariance is not None and self._timestamp is not None

    def initialize(
        self,
        position: np.ndarray,
        covariance: np.ndarray,
        timestamp: float,
        velocity: np.ndarray | None = None,
    ) -> None:
        """Initialise position, optional velocity, and state covariance."""
        self._state = state_from_position_velocity(position, velocity)
        self._covariance = normalize_initial_covariance(covariance)
        self._timestamp = float(timestamp)

    def predict(self, timestamp: float) -> None:
        """Advance the state to ``timestamp`` using the constant-velocity model."""
        self._ensure_initialized()
        assert self._state is not None
        assert self._covariance is not None
        assert self._timestamp is not None

        target_time = float(timestamp)
        dt = target_time - self._timestamp
        if dt < 0.0:
            raise ValueError("Cannot predict backwards in time")

        if dt == 0.0:
            return

        self._state, self._covariance = predict_state(
            self._state,
            self._covariance,
            dt,
            self._acceleration_noise_std,
        )
        self._timestamp = target_time

    @abstractmethod
    def update(self, detections: list[TrackerDetection]) -> None:
        """Apply measurement update(s) at the current timestamp."""

    def get_state(self) -> np.ndarray:
        self._ensure_initialized()
        assert self._state is not None
        return self._state.copy()

    def get_covariance(self) -> np.ndarray:
        self._ensure_initialized()
        assert self._covariance is not None
        return self._covariance.copy()

    def get_position(self) -> np.ndarray:
        return position_from_state(self.get_state())

    def get_velocity(self) -> np.ndarray:
        return velocity_from_state(self.get_state())

    def _ensure_initialized(self) -> None:
        if not self.is_initialized:
            raise RuntimeError("Tracker must be initialized before use")

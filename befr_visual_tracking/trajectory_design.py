"""Synthetic target trajectories for dataset generation and experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class TrajectorySample:
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray


@dataclass(frozen=True)
class TrajectoryDefinition:
    name: str
    duration: float
    sample_rate_hz: float
    position_fn: Callable[[float], np.ndarray]
    velocity_fn: Callable[[float], np.ndarray]


def slow_circle_trajectory(
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 1.5),
    radius: float = 1.0,
    angular_rate: float = 0.3,
    duration: float | None = None,
    sample_rate_hz: float = 30.0,
) -> TrajectoryDefinition:
    """
    Horizontal circle at constant altitude.

    Default parameters produce a slow loop that remains visible from all
    configured cameras around the flight volume centre.
    """
    center_arr = np.asarray(center, dtype=float)
    period = 2.0 * np.pi / angular_rate
    if duration is None:
        duration = period

    def position_fn(t: float) -> np.ndarray:
        return center_arr + np.array(
            [radius * np.cos(angular_rate * t), radius * np.sin(angular_rate * t), 0.0],
            dtype=float,
        )

    def velocity_fn(t: float) -> np.ndarray:
        return np.array(
            [
                -radius * angular_rate * np.sin(angular_rate * t),
                radius * angular_rate * np.cos(angular_rate * t),
                0.0,
            ],
            dtype=float,
        )

    return TrajectoryDefinition(
        name="slow_circle",
        duration=float(duration),
        sample_rate_hz=float(sample_rate_hz),
        position_fn=position_fn,
        velocity_fn=velocity_fn,
    )


def stationary_hover_trajectory(
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 1.5),
    duration: float = 20.0,
    sample_rate_hz: float = 30.0,
) -> TrajectoryDefinition:
    """Constant position — weak parallax for calibration observability tests."""
    center_arr = np.asarray(center, dtype=float)
    zero = np.zeros(3, dtype=float)

    def position_fn(_t: float) -> np.ndarray:
        return center_arr.copy()

    def velocity_fn(_t: float) -> np.ndarray:
        return zero.copy()

    return TrajectoryDefinition(
        name="stationary_hover",
        duration=float(duration),
        sample_rate_hz=float(sample_rate_hz),
        position_fn=position_fn,
        velocity_fn=velocity_fn,
    )


def straight_line_trajectory(
    *,
    start: tuple[float, float, float] = (-1.5, -1.0, 1.2),
    end: tuple[float, float, float] = (1.5, 1.0, 1.8),
    duration: float = 20.0,
    sample_rate_hz: float = 30.0,
) -> TrajectoryDefinition:
    """Linear motion — often degenerate for relative-pose estimation."""
    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    direction = end_arr - start_arr

    def position_fn(t: float) -> np.ndarray:
        alpha = min(max(t / duration, 0.0), 1.0)
        return start_arr + alpha * direction

    def velocity_fn(_t: float) -> np.ndarray:
        return direction / duration

    return TrajectoryDefinition(
        name="straight_line",
        duration=float(duration),
        sample_rate_hz=float(sample_rate_hz),
        position_fn=position_fn,
        velocity_fn=velocity_fn,
    )


def square_trajectory(
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 1.5),
    half_extent: float = 0.8,
    period: float = 20.0,
    sample_rate_hz: float = 30.0,
) -> TrajectoryDefinition:
    """Horizontal square path at constant altitude."""
    center_arr = np.asarray(center, dtype=float)
    corners = [
        np.array([half_extent, half_extent, 0.0]),
        np.array([-half_extent, half_extent, 0.0]),
        np.array([-half_extent, -half_extent, 0.0]),
        np.array([half_extent, -half_extent, 0.0]),
    ]
    segment_duration = period / 4.0

    def position_fn(t: float) -> np.ndarray:
        t_mod = t % period
        segment = int(t_mod // segment_duration) % 4
        alpha = (t_mod % segment_duration) / segment_duration
        start = corners[segment]
        end = corners[(segment + 1) % 4]
        return center_arr + (1.0 - alpha) * start + alpha * end

    def velocity_fn(t: float) -> np.ndarray:
        dt = 1e-4
        return (position_fn(t + dt) - position_fn(t - dt)) / (2.0 * dt)

    return TrajectoryDefinition(
        name="square",
        duration=float(period),
        sample_rate_hz=float(sample_rate_hz),
        position_fn=position_fn,
        velocity_fn=velocity_fn,
    )


def planar_figure_eight_trajectory(
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 1.5),
    scale_x: float = 1.0,
    scale_y: float = 0.7,
    angular_rate: float = 0.25,
    duration: float | None = None,
    sample_rate_hz: float = 30.0,
) -> TrajectoryDefinition:
    """Planar lemniscate-style figure-eight at constant altitude."""
    center_arr = np.asarray(center, dtype=float)
    period = 2.0 * np.pi / angular_rate
    if duration is None:
        duration = 2.0 * period

    def position_fn(t: float) -> np.ndarray:
        wt = angular_rate * t
        return center_arr + np.array(
            [scale_x * np.sin(wt), scale_y * np.sin(wt) * np.cos(wt), 0.0],
            dtype=float,
        )

    def velocity_fn(t: float) -> np.ndarray:
        wt = angular_rate * t
        return np.array(
            [
                scale_x * angular_rate * np.cos(wt),
                scale_y * angular_rate * (np.cos(wt) ** 2 - np.sin(wt) ** 2),
                0.0,
            ],
            dtype=float,
        )

    return TrajectoryDefinition(
        name="planar_figure_eight",
        duration=float(duration),
        sample_rate_hz=float(sample_rate_hz),
        position_fn=position_fn,
        velocity_fn=velocity_fn,
    )


def calibration_figure_eight_3d_trajectory(
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 1.5),
    scale_x: float = 1.2,
    scale_y: float = 0.9,
    vertical_amplitude: float = 0.45,
    angular_rate: float = 0.22,
    duration: float | None = None,
    sample_rate_hz: float = 30.0,
) -> TrajectoryDefinition:
    """
    Rich calibration manoeuvre: horizontal figure-eight with climb/descend.

    Designed to create parallax, altitude variation, and diagonal passes through
    overlapping camera fields of view.
    """
    center_arr = np.asarray(center, dtype=float)
    period = 2.0 * np.pi / angular_rate
    if duration is None:
        duration = 2.5 * period

    def position_fn(t: float) -> np.ndarray:
        wt = angular_rate * t
        altitude = vertical_amplitude * np.sin(0.65 * wt) + 0.35 * vertical_amplitude * np.sin(
            1.35 * wt + 0.4
        )
        return center_arr + np.array(
            [
                scale_x * np.sin(wt),
                scale_y * np.sin(wt) * np.cos(wt),
                altitude,
            ],
            dtype=float,
        )

    def velocity_fn(t: float) -> np.ndarray:
        wt = angular_rate * t
        dx = scale_x * angular_rate * np.cos(wt)
        dy = scale_y * angular_rate * (np.cos(wt) ** 2 - np.sin(wt) ** 2)
        dz = vertical_amplitude * 0.65 * angular_rate * np.cos(0.65 * wt) + (
            0.35 * vertical_amplitude * 1.35 * angular_rate * np.cos(1.35 * wt + 0.4)
        )
        return np.array([dx, dy, dz], dtype=float)

    return TrajectoryDefinition(
        name="calibration_figure_eight_3d",
        duration=float(duration),
        sample_rate_hz=float(sample_rate_hz),
        position_fn=position_fn,
        velocity_fn=velocity_fn,
    )


TRAJECTORY_BUILDERS = {
    "slow_circle": slow_circle_trajectory,
    "square": square_trajectory,
    "stationary_hover": stationary_hover_trajectory,
    "straight_line": straight_line_trajectory,
    "planar_figure_eight": planar_figure_eight_trajectory,
    "calibration_figure_eight_3d": calibration_figure_eight_3d_trajectory,
}


def build_trajectory(name: str, **kwargs) -> TrajectoryDefinition:
    """Instantiate a named trajectory definition."""
    try:
        builder = TRAJECTORY_BUILDERS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(TRAJECTORY_BUILDERS))
        raise ValueError(f"Unsupported trajectory {name!r}. Expected one of: {supported}.") from exc
    return builder(**kwargs)

def sample_trajectory(definition: TrajectoryDefinition) -> list[TrajectorySample]:
    """Sample a trajectory at a fixed rate."""
    dt = 1.0 / definition.sample_rate_hz
    timestamps = np.arange(0.0, definition.duration + 0.5 * dt, dt)
    samples: list[TrajectorySample] = []
    for timestamp in timestamps:
        if timestamp > definition.duration + 1e-9:
            break
        samples.append(
            TrajectorySample(
                timestamp=float(timestamp),
                position=definition.position_fn(float(timestamp)),
                velocity=definition.velocity_fn(float(timestamp)),
            )
        )
    return samples

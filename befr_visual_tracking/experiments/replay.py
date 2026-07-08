"""Instrumented offline filter replay for experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from befr_visual_tracking.experiments.metrics import StepRecord
from befr_visual_tracking.kalman_tracker import group_detections_by_timestamp, load_ground_truth_by_timestamp
from befr_visual_tracking.tracker_base import TrackerDetection
from befr_visual_tracking.tracker_factory import create_tracker


@dataclass
class TrackingReplayResult:
    filter_type: str
    steps: list[StepRecord] = field(default_factory=list)
    tracker_stats: dict = field(default_factory=dict)
    initialization_timestamp: float | None = None


def _stats_to_dict(stats) -> dict:
    return {
        key: getattr(stats, key)
        for key in dir(stats)
        if not key.startswith("_") and isinstance(getattr(stats, key), (int, float))
    }


def replay_tracker_on_detections(
    *,
    filter_type: str,
    cameras_by_id: dict,
    detections_by_time: dict[float, list[TrackerDetection]],
    ground_truth_by_time: dict[float, tuple[np.ndarray, np.ndarray]],
    active_cameras: list[str] | None = None,
    acceleration_noise_std: float = 0.5,
    pixel_noise_std: float | None = 1.0,
    innovation_gate: float = 9.21,
    monte_carlo_samples: int = 500,
    random_seed: int = 42,
) -> TrackingReplayResult:
    """Replay detections through the selected filter with timing diagnostics."""
    normalized = filter_type.strip().lower()
    tracker = create_tracker(
        filter_type,
        cameras_by_id,
        active_cameras=active_cameras,
        acceleration_noise_std=acceleration_noise_std,
        pixel_noise_std=pixel_noise_std if normalized == "kf" else None,
        innovation_gate=innovation_gate,
        monte_carlo_samples=monte_carlo_samples,
        random_seed=random_seed,
    )

    result = TrackingReplayResult(filter_type=normalized)
    for timestamp in sorted(detections_by_time.keys()):
        detections = detections_by_time[timestamp]
        visible_count = len(detections)

        start = time.perf_counter()
        tracker.process_timestep(timestamp, detections)
        elapsed = time.perf_counter() - start

        if not tracker.is_initialized:
            continue

        if result.initialization_timestamp is None:
            result.initialization_timestamp = timestamp

        gt = ground_truth_by_time.get(timestamp)
        if gt is None:
            continue

        nis_values = list(getattr(tracker, "last_step_nis", []))
        result.steps.append(
            StepRecord(
                timestamp=timestamp,
                state=tracker.get_state(),
                covariance=tracker.get_covariance(),
                ground_truth_position=gt[0],
                ground_truth_velocity=gt[1],
                num_visible_cameras=visible_count,
                update_time_sec=elapsed,
                nis_values=nis_values,
            )
        )

    if hasattr(tracker, "stats"):
        result.tracker_stats = _stats_to_dict(tracker.stats)
    return result


def load_detections_for_replay(
    experiment_dir: str | Path,
    *,
    use_ideal_detections: bool = False,
) -> dict[float, list[TrackerDetection]]:
    from befr_visual_tracking.result_writer import load_detections_csv

    rows = load_detections_csv(Path(experiment_dir) / "detections.csv")
    if use_ideal_detections:
        for row in rows:
            if row.get("u_ideal") and row.get("v_ideal"):
                row["u"] = row["u_ideal"]
                row["v"] = row["v_ideal"]
            elif row.get("visible", "").lower() == "true":
                row["u"] = row.get("u", "")
                row["v"] = row.get("v", "")

    return group_detections_by_timestamp(rows)


def load_ground_truth_for_replay(experiment_dir: str | Path):
    return load_ground_truth_by_timestamp(Path(experiment_dir) / "ground_truth.csv")

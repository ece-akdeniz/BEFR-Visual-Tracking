"""Dropout schedules for Experiment 4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from befr_visual_tracking.dataset_generator import DetectionRow


@dataclass(frozen=True)
class DropoutPhase:
    name: str
    num_cameras: int
    duration_fraction: float


DEFAULT_DROPOUT_PHASES = (
    DropoutPhase("all_cameras", 4, 0.20),
    DropoutPhase("three_cameras", 3, 0.15),
    DropoutPhase("two_cameras", 2, 0.15),
    DropoutPhase("one_camera", 1, 0.15),
    DropoutPhase("zero_cameras", 0, 0.10),
    DropoutPhase("recovery_all_cameras", 4, 0.25),
)


def build_dropout_mask(
    timestamps: list[float],
    camera_ids: list[str],
    phases: tuple[DropoutPhase, ...] = DEFAULT_DROPOUT_PHASES,
) -> dict[float, set[str]]:
    """
    Return active camera IDs for each timestamp following a planned dropout sequence.

    Cameras are dropped in fixed ID order when reducing count (camera_3 first, then
    camera_2, ...) so the scenario is reproducible.
    """
    if not timestamps:
        return {}

    sorted_ids = sorted(camera_ids)
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0.0:
        return {timestamp: set(sorted_ids) for timestamp in timestamps}

    phase_boundaries: list[tuple[float, DropoutPhase]] = []
    elapsed_fraction = 0.0
    for phase in phases:
        elapsed_fraction += phase.duration_fraction
        phase_boundaries.append((elapsed_fraction, phase))

    masks: dict[float, set[str]] = {}
    for timestamp in timestamps:
        progress = (timestamp - timestamps[0]) / duration
        selected_phase = phase_boundaries[-1][1]
        for boundary, phase in phase_boundaries:
            if progress <= boundary + 1e-9:
                selected_phase = phase
                break

        if selected_phase.num_cameras <= 0:
            masks[timestamp] = set()
        else:
            keep = max(1, min(selected_phase.num_cameras, len(sorted_ids)))
            masks[timestamp] = set(sorted_ids[:keep])
    return masks


def apply_dropout_to_detections(
    detections: tuple[DetectionRow, ...],
    *,
    camera_ids: list[str],
    phases: tuple[DropoutPhase, ...] = DEFAULT_DROPOUT_PHASES,
) -> tuple[DetectionRow, ...]:
    """Mark detections invisible when their camera is dropped at that timestamp."""
    timestamps = sorted({row.timestamp for row in detections})
    masks = build_dropout_mask(timestamps, camera_ids, phases)
    modified: list[DetectionRow] = []

    for row in detections:
        active = masks.get(row.timestamp, set(camera_ids))
        if row.camera_id not in active:
            modified.append(
                DetectionRow(
                    timestamp=row.timestamp,
                    camera_id=row.camera_id,
                    u=None,
                    v=None,
                    u_ideal=row.u_ideal,
                    v_ideal=row.v_ideal,
                    visible=False,
                    gt_x=row.gt_x,
                    gt_y=row.gt_y,
                    gt_z=row.gt_z,
                    gt_vx=row.gt_vx,
                    gt_vy=row.gt_vy,
                    gt_vz=row.gt_vz,
                )
            )
        else:
            modified.append(row)

    return tuple(modified)

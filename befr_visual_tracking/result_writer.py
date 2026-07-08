"""Save experiment datasets in the standard results format."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from befr_visual_tracking.dataset_generator import CanonicalDataset, DetectionRow, GroundTruthRow


def _float_or_empty(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8f}"


def write_detections_csv(path: Path, rows: tuple[DetectionRow, ...]) -> None:
    fieldnames = [
        "timestamp",
        "camera_id",
        "u",
        "v",
        "visible",
        "gt_x",
        "gt_y",
        "gt_z",
        "gt_vx",
        "gt_vy",
        "gt_vz",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp": f"{row.timestamp:.8f}",
                    "camera_id": row.camera_id,
                    "u": _float_or_empty(row.u),
                    "v": _float_or_empty(row.v),
                    "visible": str(row.visible).lower(),
                    "gt_x": f"{row.gt_x:.8f}",
                    "gt_y": f"{row.gt_y:.8f}",
                    "gt_z": f"{row.gt_z:.8f}",
                    "gt_vx": f"{row.gt_vx:.8f}",
                    "gt_vy": f"{row.gt_vy:.8f}",
                    "gt_vz": f"{row.gt_vz:.8f}",
                }
            )


def write_detections_ideal_csv(path: Path, rows: tuple[DetectionRow, ...]) -> None:
    fieldnames = ["timestamp", "camera_id", "u_ideal", "v_ideal", "visible"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp": f"{row.timestamp:.8f}",
                    "camera_id": row.camera_id,
                    "u_ideal": _float_or_empty(row.u_ideal),
                    "v_ideal": _float_or_empty(row.v_ideal),
                    "visible": str(row.visible).lower(),
                }
            )


def write_ground_truth_csv(path: Path, rows: tuple[GroundTruthRow, ...]) -> None:
    fieldnames = ["timestamp", "gt_x", "gt_y", "gt_z", "gt_vx", "gt_vy", "gt_vz"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp": f"{row.timestamp:.8f}",
                    "gt_x": f"{row.gt_x:.8f}",
                    "gt_y": f"{row.gt_y:.8f}",
                    "gt_z": f"{row.gt_z:.8f}",
                    "gt_vx": f"{row.gt_vx:.8f}",
                    "gt_vy": f"{row.gt_vy:.8f}",
                    "gt_vz": f"{row.gt_vz:.8f}",
                }
            )


def write_metadata_json(path: Path, metadata: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def copy_camera_config(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def save_canonical_dataset(
    output_dir: str | Path,
    dataset: CanonicalDataset,
    *,
    metadata: dict[str, Any],
    camera_config_path: str | Path,
) -> Path:
    """Write the canonical dataset folder and return the output path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    write_detections_csv(output_path / "detections.csv", dataset.detections)
    write_detections_ideal_csv(output_path / "detections_ideal.csv", dataset.detections)
    write_ground_truth_csv(output_path / "ground_truth.csv", dataset.ground_truth)
    copy_camera_config(Path(camera_config_path), output_path / "camera_config.yaml")

    full_metadata = dict(metadata)
    full_metadata["random_seed"] = dataset.random_seed
    full_metadata["num_detection_rows"] = len(dataset.detections)
    full_metadata["num_ground_truth_rows"] = len(dataset.ground_truth)
    write_metadata_json(output_path / "metadata.json", full_metadata)

    return output_path


def load_detections_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_ground_truth_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_metadata_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def allocate_experiment_dir(results_root: str | Path) -> Path:
    """Return the next available results/experiment_NNN directory."""
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in root.glob("experiment_*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("experiment_")
        if suffix.isdigit():
            numbers.append(int(suffix))
    next_id = max(numbers, default=0) + 1
    allocated = root / f"experiment_{next_id:03d}"
    allocated.mkdir(parents=True, exist_ok=False)
    return allocated


def write_estimates_csv(path: Path, steps) -> None:
    """Write per-timestep filter estimates and diagnostics."""
    fieldnames = [
        "timestamp",
        "est_x",
        "est_y",
        "est_z",
        "est_vx",
        "est_vy",
        "est_vz",
        "pos_err",
        "vel_err",
        "num_visible_cameras",
        "update_time_sec",
        "nis_mean",
        "cov_px",
        "cov_py",
        "cov_pz",
        "cov_vx",
        "cov_vy",
        "cov_vz",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in steps:
            pos_err = float(np.linalg.norm(step.state[:3] - step.ground_truth_position))
            vel_err = float(np.linalg.norm(step.state[3:] - step.ground_truth_velocity))
            cov = step.covariance
            nis_mean = (
                float(np.mean(step.nis_values)) if step.nis_values else ""
            )
            writer.writerow(
                {
                    "timestamp": f"{step.timestamp:.8f}",
                    "est_x": f"{step.state[0]:.8f}",
                    "est_y": f"{step.state[1]:.8f}",
                    "est_z": f"{step.state[2]:.8f}",
                    "est_vx": f"{step.state[3]:.8f}",
                    "est_vy": f"{step.state[4]:.8f}",
                    "est_vz": f"{step.state[5]:.8f}",
                    "pos_err": f"{pos_err:.8f}",
                    "vel_err": f"{vel_err:.8f}",
                    "num_visible_cameras": str(step.num_visible_cameras),
                    "update_time_sec": f"{step.update_time_sec:.8f}",
                    "nis_mean": f"{nis_mean:.8f}" if nis_mean != "" else "",
                    "cov_px": f"{cov[0, 0]:.8f}",
                    "cov_py": f"{cov[1, 1]:.8f}",
                    "cov_pz": f"{cov[2, 2]:.8f}",
                    "cov_vx": f"{cov[3, 3]:.8f}",
                    "cov_vy": f"{cov[4, 4]:.8f}",
                    "cov_vz": f"{cov[5, 5]:.8f}",
                }
            )


def write_metrics_json(path: Path, metrics: dict[str, Any]) -> None:
    write_metadata_json(path, metrics)


def save_tracking_experiment(
    output_dir: str | Path,
    *,
    dataset: CanonicalDataset | None,
    metadata: dict[str, Any],
    steps,
    metrics: dict[str, Any],
    camera_config_path: str | Path,
    source_dataset_dir: str | Path | None = None,
) -> Path:
    """
    Write the standard experiment folder consumed by the dashboard.

    If ``dataset`` is None, detections and ground truth are copied from
    ``source_dataset_dir``.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if dataset is not None:
        write_detections_csv(output_path / "detections.csv", dataset.detections)
        write_ground_truth_csv(output_path / "ground_truth.csv", dataset.ground_truth)
    elif source_dataset_dir is not None:
        source = Path(source_dataset_dir)
        shutil.copy2(source / "detections.csv", output_path / "detections.csv")
        shutil.copy2(source / "ground_truth.csv", output_path / "ground_truth.csv")
    else:
        raise ValueError("Either dataset or source_dataset_dir must be provided")

    copy_camera_config(Path(camera_config_path), output_path / "camera_config.yaml")
    write_estimates_csv(output_path / "estimates.csv", steps)
    write_metrics_json(output_path / "metrics.json", metrics)

    full_metadata = dict(metadata)
    if dataset is not None:
        full_metadata["random_seed"] = dataset.random_seed
        full_metadata["num_detection_rows"] = len(dataset.detections)
        full_metadata["num_ground_truth_rows"] = len(dataset.ground_truth)
    full_metadata["num_estimate_rows"] = len(steps)
    write_metadata_json(output_path / "metadata.json", full_metadata)
    return output_path

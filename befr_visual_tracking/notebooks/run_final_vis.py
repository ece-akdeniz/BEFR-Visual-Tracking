#!/usr/bin/env python3
"""Run final_vis.ipynb for a given results folder."""

from __future__ import annotations

import argparse
import re
import sys
from copy import deepcopy
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def patch_controls_source(source: str, *, results_rel: str, output_name: str, timestep_index: int) -> str:
    source = re.sub(
        r'RESULTS_DIR = REPO_ROOT / "results" / "[^"]+"',
        f'RESULTS_DIR = REPO_ROOT / "{results_rel}"',
        source,
        count=1,
    )
    source = re.sub(
        r"STATIC_TIMESTEP_INDEX = \d+",
        f"STATIC_TIMESTEP_INDEX = {timestep_index}",
        source,
        count=1,
    )
    source = re.sub(
        r'STATIC_HTML_OUTPUT_DIR = REPO_ROOT / "befr_visual_tracking" / "notebooks" / "final_vis_figures" / RESULTS_DIR\.name',
        f'STATIC_HTML_OUTPUT_DIR = REPO_ROOT / "befr_visual_tracking" / "notebooks" / "final_vis_figures" / "{output_name}"',
        source,
        count=1,
    )
    return source


def patch_data_load_source(source: str, active_cameras: list[str] | None) -> str:
    if active_cameras is None:
        return source
    cameras_literal = ", ".join(f'"{camera_id}"' for camera_id in active_cameras)
    marker = "cameras_by_id = {camera.name: camera for camera in cameras}"
    if marker not in source:
        return source
    return source.replace(
        marker,
        marker
        + f"\nACTIVE_CAMERA_IDS = [{cameras_literal}]  # advanced-task override\n"
        + "selected_cameras = [cameras_by_id[camera_id] for camera_id in ACTIVE_CAMERA_IDS]\n",
        1,
    )


def find_timestep_index_for_time(results_dir: Path, target_time_s: float) -> int:
    sys.path.insert(0, str(results_dir.parents[1]))
    from befr_visual_tracking.experiments.replay import load_detections_for_replay, load_ground_truth_for_replay

    detections_by_time = load_detections_for_replay(results_dir)
    ground_truth_by_time = load_ground_truth_for_replay(results_dir)
    timestamps = sorted(ts for ts in detections_by_time if ts in ground_truth_by_time)
    if not timestamps:
        return 0
    best_ts = min(timestamps, key=lambda ts: abs(ts - target_time_s))
    return timestamps.index(best_ts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results/three_camera_takeoff_circles4",
        help="Results folder relative to repo root",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Subfolder under final_vis_figures/ (defaults to results folder name)",
    )
    parser.add_argument("--timestep-index", type=int, default=None)
    parser.add_argument("--target-time-s", type=float, default=None)
    parser.add_argument(
        "--active-cameras",
        nargs="+",
        default=None,
        help="Optional camera subset, e.g. camera_0 camera_1 camera_2",
    )
    parser.add_argument("--kernel", default="befr", help="Jupyter kernel name")
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Notebook execution timeout in seconds",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / args.results_dir
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    output_name = args.output_name or results_dir.name
    timestep_index = args.timestep_index
    if timestep_index is None:
        if args.target_time_s is not None:
            timestep_index = find_timestep_index_for_time(results_dir, args.target_time_s)
        else:
            timestep_index = 73

    notebook_path = repo_root / "befr_visual_tracking" / "notebooks" / "final_vis.ipynb"
    nb = nbformat.read(notebook_path, as_version=4)
    patched = deepcopy(nb)

    for cell in patched.cells:
        if cell.cell_type != "code":
            continue
        source = "".join(cell.source)
        if "RESULTS_DIR = REPO_ROOT" in source and "STATIC_TIMESTEP_INDEX" in source:
            cell.source = patch_controls_source(
                source,
                results_rel=args.results_dir,
                output_name=output_name,
                timestep_index=timestep_index,
            )
        if "cameras_by_id = {camera.name: camera for camera in cameras}" in source:
            cell.source = patch_data_load_source(source, args.active_cameras)

    print(f"Executing {notebook_path.name}")
    print(f"  results: {results_dir}")
    print(f"  output:  befr_visual_tracking/notebooks/final_vis_figures/{output_name}")
    print(f"  timestep index: {timestep_index}")
    if args.active_cameras:
        print(f"  active cameras: {', '.join(args.active_cameras)}")

    client = NotebookClient(
        patched,
        timeout=args.timeout,
        kernel_name=args.kernel,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        print(exc, file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

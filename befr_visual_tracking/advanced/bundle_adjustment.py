"""Batch joint optimisation of camera poses and target trajectory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from befr_visual_tracking.advanced.observations import TimestampObservations
from befr_visual_tracking.camera_model import Camera


@dataclass
class BundleAdjustmentProblem:
    camera_ids: list[str]
    reference_camera_id: str
    timestamps: list[float]
    observations: list[TimestampObservations]
    intrinsics_k: np.ndarray
    initial_camera_rotvecs: dict[str, np.ndarray]
    initial_camera_translations: dict[str, np.ndarray]
    initial_positions: np.ndarray


@dataclass
class BundleAdjustmentResult:
    camera_rotvecs: dict[str, np.ndarray]
    camera_translations: dict[str, np.ndarray]
    positions: np.ndarray
    timestamps: list[float]
    cost: float
    success: bool
    nfev: int
    reprojection_rmse: float


def project_point(
    R_world_to_camera: np.ndarray,
    t_world_to_camera: np.ndarray,
    intrinsics_k: np.ndarray,
    point_world: np.ndarray,
) -> np.ndarray:
    point_camera = R_world_to_camera @ point_world + t_world_to_camera
    z = point_camera[2]
    if z <= 1e-6:
        return np.array([np.nan, np.nan], dtype=float)
    homogeneous = intrinsics_k @ point_camera
    return homogeneous[:2] / z


def pack_parameters(
    camera_ids: list[str],
    reference_camera_id: str,
    camera_rotvecs: dict[str, np.ndarray],
    camera_translations: dict[str, np.ndarray],
    positions: np.ndarray,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for camera_id in camera_ids:
        if camera_id == reference_camera_id:
            continue
        blocks.append(np.asarray(camera_rotvecs[camera_id], dtype=float).reshape(3))
        blocks.append(np.asarray(camera_translations[camera_id], dtype=float).reshape(3))
    blocks.append(np.asarray(positions, dtype=float).reshape(-1))
    return np.concatenate(blocks)


def unpack_parameters(
    params: np.ndarray,
    camera_ids: list[str],
    reference_camera_id: str,
    num_timesteps: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    offset = 0
    rotvecs: dict[str, np.ndarray] = {reference_camera_id: np.zeros(3, dtype=float)}
    translations: dict[str, np.ndarray] = {reference_camera_id: np.zeros(3, dtype=float)}

    for camera_id in camera_ids:
        if camera_id == reference_camera_id:
            continue
        rotvecs[camera_id] = params[offset : offset + 3].copy()
        offset += 3
        translations[camera_id] = params[offset : offset + 3].copy()
        offset += 3

    positions = params[offset : offset + 3 * num_timesteps].reshape(num_timesteps, 3)
    return rotvecs, translations, positions


def build_problem(
    observations: list[TimestampObservations],
    cameras_by_id: dict[str, Camera],
    initial_positions: np.ndarray,
    *,
    reference_camera_id: str = "camera_0",
    active_cameras: list[str] | None = None,
) -> BundleAdjustmentProblem:
    camera_ids = active_cameras or sorted(cameras_by_id.keys())
    if reference_camera_id not in camera_ids:
        raise ValueError("Reference camera must be included in active_cameras")

    template = next(iter(cameras_by_id.values()))
    initial_rotvecs: dict[str, np.ndarray] = {}
    initial_translations: dict[str, np.ndarray] = {}
    for camera_id in camera_ids:
        camera = cameras_by_id[camera_id]
        R_wc = camera.extrinsics.R_world_to_camera
        initial_rotvecs[camera_id] = Rotation.from_matrix(R_wc).as_rotvec()
        initial_translations[camera_id] = camera.extrinsics.t_world_to_camera.copy()

    timestamps = [observation.timestamp for observation in observations]
    return BundleAdjustmentProblem(
        camera_ids=camera_ids,
        reference_camera_id=reference_camera_id,
        timestamps=timestamps,
        observations=observations,
        intrinsics_k=template.intrinsics.K,
        initial_camera_rotvecs=initial_rotvecs,
        initial_camera_translations=initial_translations,
        initial_positions=np.asarray(initial_positions, dtype=float).reshape(-1, 3),
    )


def _residuals(
    params: np.ndarray,
    problem: BundleAdjustmentProblem,
    smoothness_weight: float,
) -> np.ndarray:
    rotvecs, translations, positions = unpack_parameters(
        params,
        problem.camera_ids,
        problem.reference_camera_id,
        len(problem.timestamps),
    )

    residuals: list[float] = []
    for time_index, observation in enumerate(problem.observations):
        point = positions[time_index]
        for measurement in observation.measurements:
            camera_id = measurement.camera_id
            R = Rotation.from_rotvec(rotvecs[camera_id]).as_matrix()
            t = translations[camera_id]
            projected = project_point(R, t, problem.intrinsics_k, point)
            if np.any(np.isnan(projected)):
                residuals.extend([100.0, 100.0])
            else:
                residuals.append(projected[0] - measurement.u)
                residuals.append(projected[1] - measurement.v)

    if smoothness_weight > 0.0 and len(positions) >= 3:
        for index in range(1, len(positions) - 1):
            smooth = positions[index + 1] - 2.0 * positions[index] + positions[index - 1]
            residuals.extend((smoothness_weight * smooth).tolist())

    return np.asarray(residuals, dtype=float)


def run_bundle_adjustment(
    problem: BundleAdjustmentProblem,
    *,
    smoothness_weight: float = 0.05,
    max_nfev: int = 200,
) -> BundleAdjustmentResult:
    """Jointly optimise camera poses and trajectory positions."""
    initial = pack_parameters(
        problem.camera_ids,
        problem.reference_camera_id,
        problem.initial_camera_rotvecs,
        problem.initial_camera_translations,
        problem.initial_positions,
    )
    initial_residuals = _residuals(initial, problem, smoothness_weight)
    num_reprojection = int(np.sum(np.isfinite(initial_residuals)))

    result = least_squares(
        _residuals,
        initial,
        args=(problem, smoothness_weight),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=max_nfev,
    )

    rotvecs, translations, positions = unpack_parameters(
        result.x,
        problem.camera_ids,
        problem.reference_camera_id,
        len(problem.timestamps),
    )

    final_residuals = _residuals(result.x, problem, smoothness_weight)
    reproj = final_residuals[:num_reprojection]
    reprojection_rmse = float(np.sqrt(np.mean(reproj**2))) if len(reproj) else float("inf")

    return BundleAdjustmentResult(
        camera_rotvecs=rotvecs,
        camera_translations=translations,
        positions=positions,
        timestamps=problem.timestamps.copy(),
        cost=float(result.cost),
        success=bool(result.success),
        nfev=int(result.nfev),
        reprojection_rmse=reprojection_rmse,
    )


def bundle_adjustment_to_cameras(
    result: BundleAdjustmentResult,
    template_cameras: dict[str, Camera],
    *,
    reference_camera_id: str = "camera_0",
) -> dict[str, Camera]:
    """Convert BA result back to Camera objects."""
    from befr_visual_tracking.advanced.gauge import camera_to_extrinsics

    cameras: dict[str, Camera] = {}
    for camera_id, template in template_cameras.items():
        if camera_id not in result.camera_rotvecs:
            continue
        R = Rotation.from_rotvec(result.camera_rotvecs[camera_id]).as_matrix()
        t = result.camera_translations[camera_id]
        cameras[camera_id] = Camera(
            name=template.name,
            frame_id=template.frame_id,
            intrinsics=template.intrinsics,
            extrinsics=camera_to_extrinsics(R, t),
            noise_model=template.noise_model,
            near_clip=template.near_clip,
            far_clip=template.far_clip,
            frame_convention=template.frame_convention,
        )
    if reference_camera_id not in cameras:
        cameras[reference_camera_id] = template_cameras[reference_camera_id]
    return cameras

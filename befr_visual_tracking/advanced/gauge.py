"""Gauge fixing with camera 0 as the reference frame."""

from __future__ import annotations

import copy

import numpy as np

from befr_visual_tracking.camera_config import quaternion_from_rotation_matrix
from befr_visual_tracking.camera_model import Camera, CameraExtrinsics


def world_to_camera_transform(extrinsics: CameraExtrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Return (R_world_to_camera, t_world_to_camera)."""
    R_wc = extrinsics.R_world_to_camera
    t_wc = extrinsics.t_world_to_camera
    return R_wc.copy(), t_wc.copy()


def transform_point_to_reference(
    point_world: np.ndarray,
    reference_extrinsics: CameraExtrinsics,
) -> np.ndarray:
    """Express a world point in the camera_0 reference frame."""
    R_ref_w, t_ref_w = world_to_camera_transform(reference_extrinsics)
    return R_ref_w @ np.asarray(point_world, dtype=float).reshape(3) + t_ref_w


def relative_world_to_camera(
    camera_extrinsics: CameraExtrinsics,
    reference_extrinsics: CameraExtrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Express a camera pose in the reference frame where camera_0 is identity.

    Returns world-to-camera rotation and translation in the reference frame.
    """
    R_i_w, t_i_w = world_to_camera_transform(camera_extrinsics)
    R_ref_cw = reference_extrinsics.R_camera_to_world
    R_ref_w = reference_extrinsics.R_world_to_camera
    t_ref_w = reference_extrinsics.t_world_to_camera

    R_new = R_i_w @ R_ref_cw
    t_new = t_i_w - R_i_w @ R_ref_w @ t_ref_w
    return R_new, t_new


def camera_to_extrinsics(
    R_world_to_camera: np.ndarray,
    t_world_to_camera: np.ndarray,
) -> CameraExtrinsics:
    """Build CameraExtrinsics from a world-to-camera transform."""
    R_wc = np.asarray(R_world_to_camera, dtype=float).reshape(3, 3)
    t_wc = np.asarray(t_world_to_camera, dtype=float).reshape(3)
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    orientation = quaternion_from_rotation_matrix(R_cw)
    return CameraExtrinsics(position=t_cw, orientation=orientation)


def apply_gauge_fix_to_cameras(
    cameras_by_id: dict[str, Camera],
    reference_camera_id: str = "camera_0",
) -> dict[str, Camera]:
    """Return a copy of cameras expressed in the camera_0 reference gauge."""
    if reference_camera_id not in cameras_by_id:
        raise KeyError(f"Reference camera {reference_camera_id!r} not found")

    reference = cameras_by_id[reference_camera_id]
    fixed: dict[str, Camera] = {}
    for camera_id, camera in cameras_by_id.items():
        R_wc, t_wc = relative_world_to_camera(camera.extrinsics, reference.extrinsics)
        fixed[camera_id] = Camera(
            name=camera.name,
            frame_id=camera.frame_id,
            intrinsics=copy.deepcopy(camera.intrinsics),
            extrinsics=camera_to_extrinsics(R_wc, t_wc),
            noise_model=copy.deepcopy(camera.noise_model),
            near_clip=camera.near_clip,
            far_clip=camera.far_clip,
            frame_convention=camera.frame_convention,
        )
    return fixed


def apply_gauge_fix_to_points(
    points: np.ndarray,
    reference_extrinsics: CameraExtrinsics,
) -> np.ndarray:
    """Transform Nx3 world points into the camera_0 reference frame."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return np.array(
        [transform_point_to_reference(point, reference_extrinsics) for point in points],
        dtype=float,
    )

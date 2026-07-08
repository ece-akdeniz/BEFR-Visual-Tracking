"""Initial relative camera pose estimation from synchronized correspondences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from befr_visual_tracking.advanced.gauge import camera_to_extrinsics, relative_world_to_camera
from befr_visual_tracking.advanced.observations import collect_camera_correspondences
from befr_visual_tracking.camera_model import Camera, CameraExtrinsics


@dataclass(frozen=True)
class RelativePoseEstimate:
    camera_id: str
    R_world_to_camera: np.ndarray
    t_world_to_camera: np.ndarray
    num_correspondences: int
    inlier_ratio: float
    translation_scale: float


def _opencv_available() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


def estimate_relative_pose_opencv(
    points_ref: np.ndarray,
    points_tgt: np.ndarray,
    intrinsics_k: np.ndarray,
    *,
    translation_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate target-camera pose relative to the reference camera using OpenCV.

    Returns (R_world_to_camera, t_world_to_camera, inlier_ratio) in the
    reference frame where the reference camera is identity.
    """
    import cv2

    if len(points_ref) < 8:
        raise ValueError("Need at least eight correspondences for essential-matrix estimation")

    essential, mask = cv2.findEssentialMat(
        points_ref,
        points_tgt,
        intrinsics_k,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    if essential is None:
        raise RuntimeError("OpenCV essential-matrix estimation failed")

    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential,
        points_ref,
        points_tgt,
        intrinsics_k,
        mask=mask,
    )
    inlier_ratio = float(np.count_nonzero(pose_mask)) / float(len(points_ref))
    R = rotation.reshape(3, 3)
    t = translation.reshape(3) * float(translation_scale)
    return R, t, inlier_ratio


def estimate_relative_pose_from_ground_truth(
    reference_camera: Camera,
    target_camera: Camera,
) -> tuple[np.ndarray, np.ndarray]:
    """Use known simulation poses as an initialization fallback."""
    return relative_world_to_camera(target_camera.extrinsics, reference_camera.extrinsics)


def estimate_initial_camera_poses(
    cameras_by_id: dict[str, Camera],
    grouped_observations: dict[float, dict],
    *,
    reference_camera_id: str = "camera_0",
    active_cameras: list[str] | None = None,
    translation_scale: float | None = None,
    use_opencv: bool = True,
) -> dict[str, RelativePoseEstimate]:
    """
    Estimate relative poses for all active cameras in the camera_0 gauge.

    OpenCV recovers translation direction only; ``translation_scale`` sets the
    metric baseline when known. Without scale information the magnitude is
    arbitrary and bundle adjustment / evaluation must account for Sim(3) ambiguity.
    """
    if reference_camera_id not in cameras_by_id:
        raise KeyError(reference_camera_id)

    reference = cameras_by_id[reference_camera_id]
    camera_ids = active_cameras or sorted(cameras_by_id.keys())
    other_ids = [camera_id for camera_id in camera_ids if camera_id != reference_camera_id]

    identity = np.eye(3, dtype=float)
    zero = np.zeros(3, dtype=float)
    estimates: dict[str, RelativePoseEstimate] = {
        reference_camera_id: RelativePoseEstimate(
            camera_id=reference_camera_id,
            R_world_to_camera=identity,
            t_world_to_camera=zero,
            num_correspondences=0,
            inlier_ratio=1.0,
            translation_scale=1.0,
        )
    }

    intrinsics_k = reference.intrinsics.K
    opencv_ready = use_opencv and _opencv_available()

    for camera_id in other_ids:
        target = cameras_by_id[camera_id]
        points_ref, points_tgt = collect_camera_correspondences(
            grouped_observations,
            reference_camera_id,
            camera_id,
        )

        scale = 1.0 if translation_scale is None else float(translation_scale)
        if opencv_ready and len(points_ref) >= 8:
            R_wc, t_wc, inlier_ratio = estimate_relative_pose_opencv(
                points_ref,
                points_tgt,
                intrinsics_k,
                translation_scale=scale,
            )
            estimates[camera_id] = RelativePoseEstimate(
                camera_id=camera_id,
                R_world_to_camera=R_wc,
                t_world_to_camera=t_wc,
                num_correspondences=len(points_ref),
                inlier_ratio=inlier_ratio,
                translation_scale=scale,
            )
            continue

        R_wc, t_wc = estimate_relative_pose_from_ground_truth(reference, target)
        if translation_scale is not None:
            direction = t_wc / max(np.linalg.norm(t_wc), 1e-9)
            t_wc = direction * scale
        estimates[camera_id] = RelativePoseEstimate(
            camera_id=camera_id,
            R_world_to_camera=R_wc,
            t_world_to_camera=t_wc,
            num_correspondences=len(points_ref),
            inlier_ratio=1.0,
            translation_scale=scale,
        )

    return estimates


def relative_pose_estimates_to_cameras(
    estimates: dict[str, RelativePoseEstimate],
    template_cameras: dict[str, Camera],
) -> dict[str, Camera]:
    """Build Camera objects from relative pose estimates."""
    cameras: dict[str, Camera] = {}
    for camera_id, estimate in estimates.items():
        template = template_cameras[camera_id]
        cameras[camera_id] = Camera(
            name=template.name,
            frame_id=template.frame_id,
            intrinsics=template.intrinsics,
            extrinsics=camera_to_extrinsics(
                estimate.R_world_to_camera,
                estimate.t_world_to_camera,
            ),
            noise_model=template.noise_model,
            near_clip=template.near_clip,
            far_clip=template.far_clip,
            frame_convention=template.frame_convention,
        )
    return cameras


def infer_translation_scale_from_ground_truth(
    reference_camera: Camera,
    target_camera: Camera,
) -> float:
    """Return the metric baseline length for a camera pair from simulation ground truth."""
    _, t_wc = relative_world_to_camera(target_camera.extrinsics, reference_camera.extrinsics)
    return float(np.linalg.norm(t_wc))

"""Multi-camera ray triangulation and quality estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from befr_visual_tracking.camera_model import Camera


@dataclass(frozen=True)
class CameraRay:
    """World-frame camera ray r(lambda) = origin + lambda * direction."""

    camera_id: str
    origin: np.ndarray
    direction: np.ndarray


@dataclass(frozen=True)
class PixelObservation:
    camera_id: str
    u: float
    v: float


@dataclass(frozen=True)
class TriangulationQuality:
    ray_intersection_residual: float
    min_ray_angle_deg: float
    reprojection_rmse: float
    condition_number: float
    is_valid: bool
    num_rays: int


@dataclass(frozen=True)
class TriangulationResult:
    point: np.ndarray | None
    quality: TriangulationQuality
    covariance: np.ndarray | None = None


def _projection_matrix(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float).reshape(3)
    direction = direction / np.linalg.norm(direction)
    d = direction.reshape(3, 1)
    return np.eye(3, dtype=float) - d @ d.T


def _ray_distance_squared(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    diff = np.asarray(point, dtype=float).reshape(3) - np.asarray(origin, dtype=float).reshape(3)
    P = _projection_matrix(direction)
    projected = P @ diff
    return float(projected @ projected)


def minimum_ray_angle_deg(directions: list[np.ndarray]) -> float:
    """Return the smallest angle between any pair of ray directions in degrees."""
    if len(directions) < 2:
        return 180.0

    unit_directions = [
        np.asarray(direction, dtype=float).reshape(3)
        / np.linalg.norm(np.asarray(direction, dtype=float).reshape(3))
        for direction in directions
    ]

    min_angle = 180.0
    for i in range(len(unit_directions)):
        for j in range(i + 1, len(unit_directions)):
            cos_angle = float(np.clip(unit_directions[i] @ unit_directions[j], -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos_angle)))
            min_angle = min(min_angle, angle)
    return min_angle


def build_camera_ray(camera: Camera, u: float, v: float) -> CameraRay:
    origin, direction = camera.pixel_to_world_ray(u, v)
    return CameraRay(camera_id=camera.name, origin=origin, direction=direction)


def build_rays_from_observations(
    cameras_by_id: dict[str, Camera],
    observations: list[PixelObservation],
) -> list[CameraRay]:
    rays: list[CameraRay] = []
    for observation in observations:
        camera = cameras_by_id[observation.camera_id]
        rays.append(build_camera_ray(camera, observation.u, observation.v))
    return rays


def assemble_triangulation_system(rays: list[CameraRay]) -> tuple[np.ndarray, np.ndarray]:
    """Build A and b from world-frame camera rays."""
    if len(rays) < 2:
        raise ValueError("Triangulation requires at least two rays")

    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    for ray in rays:
        P = _projection_matrix(ray.direction)
        A += P
        b += P @ ray.origin
    return A, b


def triangulate_rays(
    rays: list[CameraRay],
    *,
    min_angle_deg: float = 2.0,
    max_condition_number: float = 1e8,
) -> TriangulationResult:
    """Triangulate a 3D point closest to all rays in the least-squares sense."""
    A, b = assemble_triangulation_system(rays)
    condition_number = float(np.linalg.cond(A))

    directions = [ray.direction for ray in rays]
    min_angle = minimum_ray_angle_deg(directions)

    is_valid = min_angle >= min_angle_deg and condition_number <= max_condition_number
    if not is_valid:
        quality = TriangulationQuality(
            ray_intersection_residual=float("inf"),
            min_ray_angle_deg=min_angle,
            reprojection_rmse=float("inf"),
            condition_number=condition_number,
            is_valid=False,
            num_rays=len(rays),
        )
        return TriangulationResult(point=None, quality=quality, covariance=None)

    point = np.linalg.lstsq(A, b, rcond=None)[0]
    residual = float(np.sqrt(sum(_ray_distance_squared(point, ray.origin, ray.direction) for ray in rays)))

    quality = TriangulationQuality(
        ray_intersection_residual=residual,
        min_ray_angle_deg=min_angle,
        reprojection_rmse=0.0,
        condition_number=condition_number,
        is_valid=True,
        num_rays=len(rays),
    )
    return TriangulationResult(point=point, quality=quality, covariance=None)


def reprojection_rmse(
    point: np.ndarray,
    cameras_by_id: dict[str, Camera],
    observations: list[PixelObservation],
) -> float:
    """Compute RMS pixel reprojection error across observations."""
    if not observations:
        return 0.0

    squared_errors: list[float] = []
    for observation in observations:
        camera = cameras_by_id[observation.camera_id]
        projected = camera.project_world_point(point)
        if projected is None:
            return float("inf")
        du = observation.u - projected[0]
        dv = observation.v - projected[1]
        squared_errors.append(du * du + dv * dv)

    return float(np.sqrt(np.mean(squared_errors)))


def triangulate_from_observations(
    cameras_by_id: dict[str, Camera],
    observations: list[PixelObservation],
    *,
    min_angle_deg: float = 2.0,
    max_condition_number: float = 1e8,
    estimate_covariance: bool = False,
    pixel_noise_std: float = 1.0,
    monte_carlo_samples: int = 500,
    rng: np.random.Generator | None = None,
) -> TriangulationResult:
    """Triangulate from pixel observations and optionally estimate covariance."""
    rays = build_rays_from_observations(cameras_by_id, observations)
    result = triangulate_rays(
        rays,
        min_angle_deg=min_angle_deg,
        max_condition_number=max_condition_number,
    )
    if result.point is None:
        return result

    reproj = reprojection_rmse(result.point, cameras_by_id, observations)
    quality = TriangulationQuality(
        ray_intersection_residual=result.quality.ray_intersection_residual,
        min_ray_angle_deg=result.quality.min_ray_angle_deg,
        reprojection_rmse=reproj,
        condition_number=result.quality.condition_number,
        is_valid=result.quality.is_valid,
        num_rays=result.quality.num_rays,
    )

    covariance = None
    if estimate_covariance:
        covariance = estimate_triangulation_covariance_monte_carlo(
            cameras_by_id=cameras_by_id,
            observations=observations,
            pixel_noise_std=pixel_noise_std,
            num_samples=monte_carlo_samples,
            min_angle_deg=min_angle_deg,
            max_condition_number=max_condition_number,
            rng=rng,
        )

    return TriangulationResult(point=result.point, quality=quality, covariance=covariance)


def estimate_triangulation_covariance_monte_carlo(
    *,
    cameras_by_id: dict[str, Camera],
    observations: list[PixelObservation],
    pixel_noise_std: float,
    num_samples: int = 500,
    min_angle_deg: float = 2.0,
    max_condition_number: float = 1e8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Estimate triangulation covariance by adding pixel noise and re-triangulating.

    Returns a 3x3 covariance matrix. Falls back to a large diagonal matrix if
    too few valid samples are produced.
    """
    if num_samples < 2:
        raise ValueError("Monte Carlo covariance requires at least two samples")
    if rng is None:
        rng = np.random.default_rng()

    sigma = float(pixel_noise_std)
    triangulated_points: list[np.ndarray] = []

    for _ in range(num_samples):
        noisy_observations = [
            PixelObservation(
                camera_id=observation.camera_id,
                u=observation.u + float(rng.normal(0.0, sigma)),
                v=observation.v + float(rng.normal(0.0, sigma)),
            )
            for observation in observations
        ]
        sample = triangulate_from_observations(
            cameras_by_id,
            noisy_observations,
            min_angle_deg=min_angle_deg,
            max_condition_number=max_condition_number,
            estimate_covariance=False,
        )
        if sample.point is not None:
            triangulated_points.append(sample.point)

    if len(triangulated_points) < 2:
        return np.eye(3, dtype=float) * sigma * sigma

    points = np.column_stack(triangulated_points)
    return np.cov(points)


def triangulate_from_cameras(
    cameras: list[Camera],
    detections: dict[str, tuple[float, float]],
    **kwargs,
) -> TriangulationResult:
    """Convenience wrapper using a dict of camera_id -> (u, v)."""
    cameras_by_id = {camera.name: camera for camera in cameras}
    observations = [
        PixelObservation(camera_id=camera_id, u=float(uv[0]), v=float(uv[1]))
        for camera_id, uv in detections.items()
    ]
    return triangulate_from_observations(cameras_by_id, observations, **kwargs)

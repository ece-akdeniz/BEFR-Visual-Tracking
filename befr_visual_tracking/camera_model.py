"""
Pinhole camera geometry for multi-camera visual tracking.

Pure mathematical model — no ROS or plotting dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from befr_visual_tracking.camera_config import (
    CameraSystemConfig,
    load_camera_system_config,
    normalize_quaternion,
    rotation_matrix_from_quaternion_xyzw,
)


class CameraFrameConvention(Enum):
    """
    Coordinate-frame convention used by the camera projection model.

    OPTICAL_FRAME:
        +x points right in the image
        +y points down in the image
        +z points forward along the optical axis

        Projection:
            u = fx * (x / z) + cx
            v = fy * (y / z) + cy
    """

    OPTICAL_FRAME = "optical_frame"


@dataclass
class CameraIntrinsics:
    width_px: int
    height_px: int
    fx: float
    fy: float
    cx: float | None = None
    cy: float | None = None

    @property
    def principal_point(self) -> tuple[float, float]:
        cx = self.cx if self.cx is not None else self.width_px / 2.0
        cy = self.cy if self.cy is not None else self.height_px / 2.0
        return cx, cy

    @property
    def horizontal_fov(self) -> float:
        return float(np.rad2deg(2 * np.arctan(self.width_px / (2 * self.fx))))

    @property
    def vertical_fov(self) -> float:
        return float(np.rad2deg(2 * np.arctan(self.height_px / (2 * self.fy))))

    @property
    def K(self) -> np.ndarray:
        cx, cy = self.principal_point
        return np.array(
            [[self.fx, 0.0, cx], [0.0, self.fy, cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )


@dataclass
class CameraExtrinsics:
    """
    Pose of the camera optical frame in the world frame.

    Orientation quaternion [x, y, z, w] represents R_camera_to_world.
    """

    position: np.ndarray
    orientation: np.ndarray

    @property
    def R_camera_to_world(self) -> np.ndarray:
        return rotation_matrix_from_quaternion_xyzw(self.orientation)

    @property
    def t_camera_to_world(self) -> np.ndarray:
        return np.asarray(self.position, dtype=float).reshape(3)

    @property
    def R_world_to_camera(self) -> np.ndarray:
        return self.R_camera_to_world.T

    @property
    def t_world_to_camera(self) -> np.ndarray:
        return -self.R_world_to_camera @ self.t_camera_to_world

    @property
    def world_to_camera_matrix(self) -> np.ndarray:
        return np.hstack(
            [self.R_world_to_camera, self.t_world_to_camera.reshape((3, 1))]
        )

    @property
    def camera_to_world_matrix(self) -> np.ndarray:
        return np.vstack(
            [
                np.hstack(
                    [self.R_camera_to_world, self.t_camera_to_world.reshape((3, 1))]
                ),
                np.array([[0.0, 0.0, 0.0, 1.0]]),
            ]
        )


@dataclass
class CameraNoiseModel:
    pixel_noise_std_u: float
    pixel_noise_std_v: float
    detection_probability: float = 1.0
    discretize: bool = True

    @property
    def measurement_noise_covariance(self) -> np.ndarray:
        return np.diag(
            [self.pixel_noise_std_u**2, self.pixel_noise_std_v**2]
        )


@dataclass
class CameraMeasurement:
    stamp: float
    frame_id: str
    camera_name: str
    u: float
    v: float
    covariance: np.ndarray | None = None


@dataclass
class Camera:
    """
    Pinhole camera sensor model using the ROS optical-frame convention.

    Extrinsics describe the pose of the camera optical frame in the world frame.
    """

    name: str
    frame_id: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    noise_model: CameraNoiseModel
    near_clip: float = 0.1
    far_clip: float = np.inf
    frame_convention: CameraFrameConvention = CameraFrameConvention.OPTICAL_FRAME

    def world_to_camera(self, point_world: np.ndarray) -> np.ndarray:
        point_world = np.asarray(point_world, dtype=float).reshape(3)
        return self.extrinsics.R_world_to_camera @ point_world + self.extrinsics.t_world_to_camera

    def project_camera_point(self, point_camera: np.ndarray) -> tuple[float, float] | None:
        if self.frame_convention != CameraFrameConvention.OPTICAL_FRAME:
            raise NotImplementedError(
                f"Projection for frame convention {self.frame_convention} is not implemented."
            )

        x, y, z = np.asarray(point_camera, dtype=float).reshape(3)
        if not (self.near_clip <= z <= self.far_clip):
            return None

        cx, cy = self.intrinsics.principal_point
        u = self.intrinsics.fx * (x / z) + cx
        v = self.intrinsics.fy * (y / z) + cy

        if not (0.0 <= u < self.intrinsics.width_px and 0.0 <= v < self.intrinsics.height_px):
            return None

        return float(u), float(v)

    def project_camera_point_for_filter(
        self,
        point_camera: np.ndarray,
        epsilon: float = 1e-6,
    ) -> tuple[float, float] | None:
        """
        Project a camera-frame point for use inside filters.

        Unlike the sensor model, this does not reject pixels outside the image.
        Projection is rejected only when the point is behind the camera (Z <= eps).
        """
        if self.frame_convention != CameraFrameConvention.OPTICAL_FRAME:
            raise NotImplementedError(
                f"Projection for frame convention {self.frame_convention} is not implemented."
            )

        x, y, z = np.asarray(point_camera, dtype=float).reshape(3)
        if z <= epsilon or z > self.far_clip:
            return None

        cx, cy = self.intrinsics.principal_point
        u = self.intrinsics.fx * (x / z) + cx
        v = self.intrinsics.fy * (y / z) + cy
        return float(u), float(v)

    def project_world_point(self, point_world: np.ndarray) -> tuple[float, float] | None:
        """Project a world-frame 3D point to pixel coordinates."""
        return self.project_camera_point(self.world_to_camera(point_world))

    def project_world_point_for_filter(
        self,
        point_world: np.ndarray,
        epsilon: float = 1e-6,
    ) -> tuple[float, float] | None:
        """Filter-side projection that remains valid slightly outside the image."""
        return self.project_camera_point_for_filter(self.world_to_camera(point_world), epsilon)

    def is_visible(self, point_world: np.ndarray) -> bool:
        """Return True when the world point projects inside the image in front of the camera."""
        return self.project_world_point(point_world) is not None

    def pixel_to_camera_ray(self, u: float, v: float) -> np.ndarray:
        """
        Return a unit direction in the camera optical frame for pixel (u, v).

        d^C = [(u-cx)/fx, (v-cy)/fy, 1]^T, normalized.
        """
        cx, cy = self.intrinsics.principal_point
        direction = np.array(
            [(u - cx) / self.intrinsics.fx, (v - cy) / self.intrinsics.fy, 1.0],
            dtype=float,
        )
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise ValueError("Degenerate camera ray direction")
        return direction / norm

    def pixel_to_world_ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the world-frame ray origin and unit direction for pixel (u, v).

        r(lambda) = origin + lambda * direction
        """
        direction_camera = self.pixel_to_camera_ray(u, v)
        origin = self.extrinsics.t_camera_to_world.copy()
        direction_world = self.extrinsics.R_camera_to_world @ direction_camera
        direction_world /= np.linalg.norm(direction_world)
        return origin, direction_world

    def projection_jacobian(
        self,
        point_world: np.ndarray,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """
        Return the 2x3 Jacobian d(u, v) / d(p_world).

        For camera-frame coordinates (X, Y, Z):

            J_pi = [[fx/Z, 0, -fx X/Z^2],
                    [0, fy/Z, -fy Y/Z^2]]

        and d(u,v)/d(p_world) = J_pi @ R_world_to_camera.
        """
        point_camera = self.world_to_camera(point_world)
        x, y, z = point_camera
        if z <= epsilon:
            raise ValueError("Cannot compute projection Jacobian behind the camera")

        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        z2 = z * z
        J_pi = np.array(
            [[fx / z, 0.0, -fx * x / z2], [0.0, fy / z, -fy * y / z2]],
            dtype=float,
        )
        return J_pi @ self.extrinsics.R_world_to_camera

    def measurement_jacobian(
        self,
        point_world: np.ndarray,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """Return the 2x6 EKF measurement Jacobian [J_pi R_CW | 0]."""
        J_position = self.projection_jacobian(point_world, epsilon=epsilon)
        return np.hstack([J_position, np.zeros((2, 3), dtype=float)])

    def measure(
        self,
        point_world: np.ndarray,
        stamp: float,
        rng: np.random.Generator | None = None,
    ) -> CameraMeasurement | None:
        point_camera = self.world_to_camera(point_world)
        projected = self.project_camera_point(point_camera)
        if projected is None:
            return None

        if rng is None:
            rng = np.random.default_rng()

        noise = rng.multivariate_normal(
            mean=[0.0, 0.0],
            cov=self.noise_model.measurement_noise_covariance,
        )
        u_noisy = projected[0] + noise[0]
        v_noisy = projected[1] + noise[1]

        if self.noise_model.discretize:
            u_noisy = float(np.round(u_noisy))
            v_noisy = float(np.round(v_noisy))
        else:
            u_noisy = float(u_noisy)
            v_noisy = float(v_noisy)

        if not (
            0.0 <= u_noisy < self.intrinsics.width_px
            and 0.0 <= v_noisy < self.intrinsics.height_px
        ):
            return None

        if rng.uniform(0.0, 1.0) > self.noise_model.detection_probability:
            return None

        return CameraMeasurement(
            stamp=stamp,
            frame_id=self.frame_id,
            camera_name=self.name,
            u=u_noisy,
            v=v_noisy,
            covariance=self.noise_model.measurement_noise_covariance,
        )

    def validate_frame_convention(self) -> None:
        if self.frame_convention != CameraFrameConvention.OPTICAL_FRAME:
            raise NotImplementedError(
                f"Validation for frame convention {self.frame_convention} is not implemented."
            )

        cx, cy = self.intrinsics.principal_point
        z = 1.0

        center = self.project_camera_point(np.array([0.0, 0.0, z]))
        right = self.project_camera_point(np.array([0.1, 0.0, z]))
        down = self.project_camera_point(np.array([0.0, 0.1, z]))
        behind = self.project_camera_point(np.array([0.0, 0.0, -z]))

        assert center is not None
        assert right is not None
        assert down is not None
        assert behind is None

        u_center, v_center = center
        u_right, v_right = right
        u_down, v_down = down

        assert np.isclose(u_center, cx)
        assert np.isclose(v_center, cy)
        assert u_right > u_center
        assert np.isclose(v_right, v_center)
        assert np.isclose(u_down, u_center)
        assert v_down > v_center


def default_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(width_px=640, height_px=480, fx=320.0, fy=320.0, cx=320.0, cy=240.0)


def default_noise_model() -> CameraNoiseModel:
    return CameraNoiseModel(pixel_noise_std_u=1.0, pixel_noise_std_v=1.0)


def make_camera(
    name: str = "camera_test",
    frame_id: str = "camera_test_optical_frame",
    position: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_xyzw: np.ndarray | tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    intrinsics: CameraIntrinsics | None = None,
    noise_model: CameraNoiseModel | None = None,
) -> Camera:
    """Build a camera from pose and optional intrinsics/noise settings."""
    return Camera(
        name=name,
        frame_id=frame_id,
        intrinsics=intrinsics or default_intrinsics(),
        extrinsics=CameraExtrinsics(
            position=np.asarray(position, dtype=float),
            orientation=normalize_quaternion(np.asarray(orientation_xyzw, dtype=float)),
        ),
        noise_model=noise_model or default_noise_model(),
    )


def cameras_from_system_config(config: CameraSystemConfig) -> list[Camera]:
    """Instantiate all cameras defined in a loaded system configuration."""
    intrinsics = CameraIntrinsics(
        width_px=config.image.width,
        height_px=config.image.height,
        fx=config.intrinsics.fx,
        fy=config.intrinsics.fy,
        cx=config.intrinsics.cx,
        cy=config.intrinsics.cy,
    )
    noise = CameraNoiseModel(
        pixel_noise_std_u=config.noise.sigma_u,
        pixel_noise_std_v=config.noise.sigma_v,
        discretize=config.noise.quantize,
        detection_probability=config.noise.detection_probability,
    )
    cameras: list[Camera] = []
    for camera_cfg in config.cameras:
        cameras.append(
            Camera(
                name=camera_cfg.id,
                frame_id=camera_cfg.frame_id,
                intrinsics=intrinsics,
                extrinsics=CameraExtrinsics(
                    position=np.array(camera_cfg.position, dtype=float),
                    orientation=np.array(camera_cfg.orientation_xyzw, dtype=float),
                ),
                noise_model=noise,
            )
        )
    return cameras


def cameras_from_yaml(path: str | Path) -> tuple[CameraSystemConfig, list[Camera]]:
    """Load camera system config and build camera models."""
    config = load_camera_system_config(path)
    return config, cameras_from_system_config(config)

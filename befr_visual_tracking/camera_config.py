"""
Camera configuration loading and pose helpers.

World frame (basic Task 1)
--------------------------
Use the Gazebo ``world`` frame: +X forward, +Y left, +Z up (ENU-style).

All camera ``position`` and ``orientation_xyzw`` values describe the pose of
each camera *optical* frame in that world frame:

    p_world = R_camera_to_world @ p_camera + t_camera_to_world

Optical frame convention (ROS):
    +x right in the image, +y down, +z forward along the optical axis.

Advanced Task 1 reference gauge
-------------------------------
Camera 0 optical frame defines the global reference (identity pose after
gauge fixing). Other camera poses are expressed relative to that frame during
calibration; the basic-task ``world`` frame remains the simulation ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class ImageConfig:
    width: int
    height: int


@dataclass(frozen=True)
class IntrinsicsConfig:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class NoiseConfig:
    sigma_u: float
    sigma_v: float
    quantize: bool = True
    detection_probability: float = 1.0


@dataclass(frozen=True)
class WorldConfig:
    frame_id: str
    flight_volume_center: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    reference_camera_id: str = "camera_0"


@dataclass(frozen=True)
class CameraPoseConfig:
    id: str
    frame_id: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    look_at: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class CameraSystemConfig:
    image: ImageConfig
    intrinsics: IntrinsicsConfig
    noise: NoiseConfig
    world: WorldConfig
    cameras: tuple[CameraPoseConfig, ...]

    def camera_by_id(self, camera_id: str) -> CameraPoseConfig:
        for camera in self.cameras:
            if camera.id == camera_id:
                return camera
        raise KeyError(f"Unknown camera id: {camera_id!r}")


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion has zero length")
    return q / norm


def rotation_matrix_from_quaternion_xyzw(q: np.ndarray) -> np.ndarray:
    """Return R_camera_to_world for quaternion [x, y, z, w]."""
    x, y, z, w = normalize_quaternion(q)
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=float,
    )


def quaternion_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to quaternion [x, y, z, w]."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return normalize_quaternion(np.array([x, y, z, w]))


def look_at_orientation_xyzw(
    position: np.ndarray | tuple[float, float, float],
    target: np.ndarray | tuple[float, float, float],
    world_up: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """
    Compute camera-to-world orientation for a ROS optical frame.

    The optical +z axis points from ``position`` toward ``target``.
    +x points right and +y points down in the image (right-handed).

    Returns:
        Quaternion [x, y, z, w] representing R_camera_to_world.
    """
    pos = np.asarray(position, dtype=float).reshape(3)
    tgt = np.asarray(target, dtype=float).reshape(3)
    up = np.asarray(world_up, dtype=float).reshape(3)

    z_axis = tgt - pos
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-9:
        raise ValueError("Camera position and look-at target are coincident")
    z_axis /= z_norm

    x_axis = np.cross(up, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        fallback_up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(fallback_up, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            raise ValueError("Cannot construct look-at frame: degenerate axes")
    x_axis /= x_norm

    y_axis = np.cross(z_axis, x_axis)
    R_camera_to_world = np.column_stack([x_axis, y_axis, z_axis])
    return quaternion_from_rotation_matrix(R_camera_to_world)


def resolve_camera_orientation(
    camera_entry: dict[str, Any],
    default_target: tuple[float, float, float],
    world_up: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Return orientation from YAML, or compute it via look-at."""
    if "orientation_xyzw" in camera_entry:
        q = normalize_quaternion(np.asarray(camera_entry["orientation_xyzw"], dtype=float))
        return tuple(float(v) for v in q)

    target = tuple(camera_entry.get("look_at", default_target))
    q = look_at_orientation_xyzw(
        position=camera_entry["position"],
        target=target,
        world_up=world_up,
    )
    return tuple(float(v) for v in q)


def _as_tuple3(values: list[float] | tuple[float, float, float]) -> tuple[float, float, float]:
    arr = tuple(float(v) for v in values)
    if len(arr) != 3:
        raise ValueError(f"Expected 3 values, got {len(arr)}")
    return arr


def load_camera_system_config(path: str | Path) -> CameraSystemConfig:
    """Load the multi-camera configuration from YAML."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    image = ImageConfig(**raw["image"])
    intrinsics = IntrinsicsConfig(**raw["intrinsics"])
    noise = NoiseConfig(**raw.get("noise", {}))

    world_raw = raw["world"]
    world = WorldConfig(
        frame_id=world_raw["frame_id"],
        flight_volume_center=_as_tuple3(world_raw["flight_volume_center"]),
        up=_as_tuple3(world_raw.get("up", (0.0, 0.0, 1.0))),
        reference_camera_id=world_raw.get("reference_camera_id", "camera_0"),
    )

    cameras: list[CameraPoseConfig] = []
    for entry in raw["cameras"]:
        orientation = resolve_camera_orientation(entry, world.flight_volume_center, world.up)
        look_at = entry.get("look_at")
        cameras.append(
            CameraPoseConfig(
                id=entry["id"],
                frame_id=entry["frame_id"],
                position=_as_tuple3(entry["position"]),
                orientation_xyzw=orientation,
                look_at=_as_tuple3(look_at) if look_at is not None else None,
            )
        )

    return CameraSystemConfig(
        image=image,
        intrinsics=intrinsics,
        noise=noise,
        world=world,
        cameras=tuple(cameras),
    )


def default_cameras_yaml_path() -> Path:
    """Return the package-default cameras.yaml path."""
    return Path(__file__).resolve().parents[1] / "config" / "cameras.yaml"

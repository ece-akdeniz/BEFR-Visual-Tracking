"""Multi-camera visual tracking for BEFR quadcopter state estimation."""

from befr_visual_tracking.camera_model import (
    Camera,
    CameraExtrinsics,
    CameraIntrinsics,
    CameraMeasurement,
    CameraNoiseModel,
    make_camera,
)

__all__ = [
    "Camera",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "CameraMeasurement",
    "CameraNoiseModel",
    "make_camera",
]
__version__ = "0.1.0"

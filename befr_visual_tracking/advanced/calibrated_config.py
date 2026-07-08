"""Save and load calibrated camera configurations."""

from __future__ import annotations

from pathlib import Path

import yaml

from befr_visual_tracking.camera_config import CameraSystemConfig, load_camera_system_config
from befr_visual_tracking.camera_model import Camera


def save_calibrated_cameras(
    output_path: str | Path,
    template_config: CameraSystemConfig,
    calibrated_cameras: dict[str, Camera],
    *,
    reference_camera_id: str = "camera_0",
    metadata: dict | None = None,
    fallback_cameras: dict[str, Camera] | None = None,
) -> Path:
    """Write calibrated camera poses to YAML in the camera_0 gauge."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    camera_entries = []
    for camera_cfg in template_config.cameras:
        if camera_cfg.id in calibrated_cameras:
            camera = calibrated_cameras[camera_cfg.id]
        elif fallback_cameras is not None and camera_cfg.id in fallback_cameras:
            camera = fallback_cameras[camera_cfg.id]
        else:
            continue
        entry = {
            "id": camera_cfg.id,
            "frame_id": camera_cfg.frame_id,
            "position": [float(v) for v in camera.extrinsics.t_camera_to_world],
            "orientation_xyzw": [float(v) for v in camera.extrinsics.orientation],
        }
        camera_entries.append(entry)

    payload = {
        "image": {
            "width": template_config.image.width,
            "height": template_config.image.height,
        },
        "intrinsics": {
            "fx": template_config.intrinsics.fx,
            "fy": template_config.intrinsics.fy,
            "cx": template_config.intrinsics.cx,
            "cy": template_config.intrinsics.cy,
        },
        "noise": {
            "sigma_u": template_config.noise.sigma_u,
            "sigma_v": template_config.noise.sigma_v,
            "quantize": template_config.noise.quantize,
            "detection_probability": template_config.noise.detection_probability,
        },
        "world": {
            "frame_id": template_config.world.frame_id,
            "flight_volume_center": list(template_config.world.flight_volume_center),
            "up": list(template_config.world.up),
            "reference_camera_id": reference_camera_id,
        },
        "calibration": metadata or {},
        "cameras": camera_entries,
    }

    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return output


def load_calibrated_cameras(path: str | Path) -> tuple[CameraSystemConfig, dict[str, Camera]]:
    """Load calibrated camera YAML through the standard camera loader."""
    from befr_visual_tracking.camera_model import cameras_from_yaml

    config_path = Path(path)
    system_config, cameras = cameras_from_yaml(config_path)
    return system_config, {camera.name: camera for camera in cameras}

"""RViz marker helpers for the camera sensor simulator demo."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from befr_visual_tracking.camera_model import Camera, CameraMeasurement

try:
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:  # pragma: no cover - optional outside ROS environment
    Marker = MarkerArray = None  # type: ignore[misc, assignment]


CAMERA_COLORS = {
    "camera_0": (0.2, 0.4, 1.0, 1.0),
    "camera_1": (1.0, 0.25, 0.25, 1.0),
    "camera_2": (1.0, 0.85, 0.1, 1.0),
    "camera_3": (0.85, 0.2, 0.85, 1.0),
}
INACTIVE_COLOR = (0.55, 0.55, 0.55, 0.8)
TARGET_COLOR = (0.1, 0.95, 0.2, 1.0)


def _color_for_camera(camera_name: str, visible: bool) -> tuple[float, float, float, float]:
    if not visible:
        return INACTIVE_COLOR
    return CAMERA_COLORS.get(camera_name, (0.3, 0.8, 0.9, 1.0))


def _set_color(marker, rgba: tuple[float, float, float, float]) -> None:
    marker.color.r = float(rgba[0])
    marker.color.g = float(rgba[1])
    marker.color.b = float(rgba[2])
    marker.color.a = float(rgba[3])


def _point(x: float, y: float, z: float):
    from geometry_msgs.msg import Point

    point = Point()
    point.x = float(x)
    point.y = float(y)
    point.z = float(z)
    return point


def build_sensor_markers(
    *,
    world_frame_id: str,
    stamp,
    cameras: Sequence[Camera],
    target_position: np.ndarray,
    measurements: dict[str, CameraMeasurement],
    ray_length: float = 6.0,
    axis_length: float = 1.0,
) -> MarkerArray:
    """
    Build RViz markers for camera poses, viewing axes, rays, and detections.

    Visible cameras are highlighted; inactive cameras are drawn in gray.
    """
    if Marker is None or MarkerArray is None:
        raise ImportError("visualization_msgs is required to build RViz markers")

    target = np.asarray(target_position, dtype=float).reshape(3)
    markers = MarkerArray()
    marker_id = 0

    target_marker = Marker()
    target_marker.header.frame_id = world_frame_id
    target_marker.header.stamp = stamp
    target_marker.ns = "target"
    target_marker.id = marker_id
    marker_id += 1
    target_marker.type = Marker.SPHERE
    target_marker.action = Marker.ADD
    target_marker.pose.position.x = float(target[0])
    target_marker.pose.position.y = float(target[1])
    target_marker.pose.position.z = float(target[2])
    target_marker.pose.orientation.w = 1.0
    target_marker.scale.x = 0.25
    target_marker.scale.y = 0.25
    target_marker.scale.z = 0.25
    _set_color(target_marker, TARGET_COLOR)
    markers.markers.append(target_marker)

    for index, camera in enumerate(cameras):
        visible = camera.name in measurements
        color = _color_for_camera(camera.name, visible)
        position = camera.extrinsics.t_camera_to_world
        forward = camera.extrinsics.R_camera_to_world @ np.array([0.0, 0.0, 1.0])

        body = Marker()
        body.header.frame_id = world_frame_id
        body.header.stamp = stamp
        body.ns = "camera_body"
        body.id = marker_id
        marker_id += 1
        body.type = Marker.SPHERE
        body.action = Marker.ADD
        body.pose.position.x = float(position[0])
        body.pose.position.y = float(position[1])
        body.pose.position.z = float(position[2])
        body.pose.orientation.w = 1.0
        body.scale.x = 0.18
        body.scale.y = 0.18
        body.scale.z = 0.18
        _set_color(body, color)
        markers.markers.append(body)

        axis = Marker()
        axis.header.frame_id = world_frame_id
        axis.header.stamp = stamp
        axis.ns = "camera_axis"
        axis.id = marker_id
        marker_id += 1
        axis.type = Marker.ARROW
        axis.action = Marker.ADD
        axis.points = [
            _point(position[0], position[1], position[2]),
            _point(
                position[0] + axis_length * forward[0],
                position[1] + axis_length * forward[1],
                position[2] + axis_length * forward[2],
            ),
        ]
        axis.scale.x = 0.06
        axis.scale.y = 0.12
        axis.scale.z = 0.12
        _set_color(axis, color)
        markers.markers.append(axis)

        label = Marker()
        label.header.frame_id = world_frame_id
        label.header.stamp = stamp
        label.ns = "camera_label"
        label.id = marker_id
        marker_id += 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(position[0])
        label.pose.position.y = float(position[1])
        label.pose.position.z = float(position[2] + 0.35)
        label.pose.orientation.w = 1.0
        label.scale.z = 0.18
        status = "visible" if visible else "no detection"
        label.text = f"{camera.name}: {status}"
        _set_color(label, color)
        markers.markers.append(label)

        if visible:
            measurement = measurements[camera.name]
            origin, direction = camera.pixel_to_world_ray(measurement.u, measurement.v)
            distance = float(np.linalg.norm(target - origin))
            length = min(ray_length, max(distance, 0.5))
            end = origin + length * direction

            ray = Marker()
            ray.header.frame_id = world_frame_id
            ray.header.stamp = stamp
            ray.ns = "camera_ray"
            ray.id = marker_id
            marker_id += 1
            ray.type = Marker.LINE_STRIP
            ray.action = Marker.ADD
            ray.points = [
                _point(origin[0], origin[1], origin[2]),
                _point(end[0], end[1], end[2]),
            ]
            ray.scale.x = 0.04
            _set_color(ray, color)
            markers.markers.append(ray)

            pixel_label = Marker()
            pixel_label.header.frame_id = world_frame_id
            pixel_label.header.stamp = stamp
            pixel_label.ns = "pixel_label"
            pixel_label.id = marker_id
            marker_id += 1
            pixel_label.type = Marker.TEXT_VIEW_FACING
            pixel_label.action = Marker.ADD
            midpoint = 0.5 * (origin + end)
            pixel_label.pose.position.x = float(midpoint[0])
            pixel_label.pose.position.y = float(midpoint[1])
            pixel_label.pose.position.z = float(midpoint[2] + 0.15)
            pixel_label.pose.orientation.w = 1.0
            pixel_label.scale.z = 0.14
            pixel_label.text = f"({measurement.u:.0f}, {measurement.v:.0f})"
            _set_color(pixel_label, color)
            markers.markers.append(pixel_label)

        _ = index

    return markers

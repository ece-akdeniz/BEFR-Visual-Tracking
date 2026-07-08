"""Publish nav_msgs/Odometry from a tracker state estimate."""

from __future__ import annotations

from nav_msgs.msg import Odometry

from befr_visual_tracking.state_estimate_export import state_estimate_from_tracker
from befr_visual_tracking.tracker_base import TrackerBase


def odometry_from_tracker(
    tracker: TrackerBase,
    *,
    frame_id: str,
    child_frame_id: str,
    stamp,
    orientation_covariance_value: float = 1e6,
) -> Odometry:
    """Build a nav_msgs/Odometry message from the current tracker state."""
    export = state_estimate_from_tracker(
        tracker,
        orientation_covariance_value=orientation_covariance_value,
    )

    message = Odometry()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.child_frame_id = child_frame_id

    message.pose.pose.position.x = float(export.position[0])
    message.pose.pose.position.y = float(export.position[1])
    message.pose.pose.position.z = float(export.position[2])
    message.pose.pose.orientation.x = 0.0
    message.pose.pose.orientation.y = 0.0
    message.pose.pose.orientation.z = 0.0
    message.pose.pose.orientation.w = 1.0
    message.pose.covariance = export.pose_covariance.tolist()

    message.twist.twist.linear.x = float(export.velocity[0])
    message.twist.twist.linear.y = float(export.velocity[1])
    message.twist.twist.linear.z = float(export.velocity[2])
    message.twist.covariance = export.twist_covariance.tolist()

    return message

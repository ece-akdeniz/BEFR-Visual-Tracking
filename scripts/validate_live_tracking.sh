#!/bin/bash
# Quick live validation: compare ground truth vs state estimate (run while stack is up).
set -eo pipefail

# ROS setup.bash uses unset vars; do not enable nounset (-u) before sourcing.
source /opt/ros/humble/setup.bash
source ~/BEFR_SS_2026/install/setup.bash
source ~/BEFR-Project/install/setup.bash

check_topic_hz() {
  local topic="$1"
  local warning="$2"
  local out
  out="$(timeout 3 ros2 topic hz "$topic" 2>&1 || true)"
  echo "$out"
  if ! echo "$out" | grep -q "average rate"; then
    echo "WARNING: $warning"
  fi
}

echo "=== Clock (sim must be running) ==="
check_topic_hz /clock "/clock not publishing — press Play in Gazebo"

echo ""
echo "=== Motor controllers ==="
ros2 control list_controllers -c /rrbot/controller_manager 2>/dev/null || echo "WARNING: controller_manager not available"

echo ""
echo "=== Ground truth (one sample) ==="
timeout 5 ros2 topic echo /model/rrbot/odometry --once 2>/dev/null || echo "WARNING: no odometry"

echo ""
echo "=== Tracking nodes ==="
NODES="$(ros2 node list 2>&1 || true)"
if echo "$NODES" | grep -q '/camera_simulator'; then
  echo "/camera_simulator"
else
  echo "WARNING: camera_simulator missing"
fi
if echo "$NODES" | grep -q '/visual_tracker'; then
  echo "/visual_tracker"
else
  echo "WARNING: visual_tracker missing — check tracking terminal for crash (Ctrl+C in Terminal 2 and relaunch)"
fi

echo ""
echo "=== State estimate (one sample) ==="
timeout 5 ros2 topic echo /visual_tracking/state_estimate --once 2>/dev/null || echo "WARNING: no state estimate — is tracking launch running?"

echo ""
echo "=== Detection rate (3 s) ==="
check_topic_hz /visual_tracking/camera_0/detection "no detections — is tracking launch running?"

echo ""
echo "Target: GT z ≈ 3–8 m (square_ctrl); estimate should be within ~1–2 m of GT."
echo "Cameras use config/cameras_gazebo.yaml (center z=5 m) by default."

#!/bin/bash
# One-glance live stack health check (run while Gazebo + square_ctrl + tracking are up).
set -eo pipefail

source /opt/ros/humble/setup.bash
source ~/BEFR_SS_2026/install/setup.bash
source ~/BEFR-Project/install/setup.bash

pass() { echo "OK: $1"; }
fail() { echo "FAIL: $1"; }

echo "=== ROS nodes ==="
NODES="$(ros2 node list 2>&1 || true)"
echo "$NODES"
echo "$NODES" | grep -q '/camera_simulator' && pass camera_simulator || fail "camera_simulator missing — restart tracking.launch.py"
echo "$NODES" | grep -q '/visual_tracker' && pass visual_tracker || fail "visual_tracker missing — check Terminal 2 for Python crash (often 'Cannot predict backwards in time')"

echo ""
echo "=== Rates (3 s) ==="
for topic in /clock /visual_tracking/camera_0/detection /visual_tracking/state_estimate; do
  out="$(timeout 3 ros2 topic hz "$topic" 2>&1 || true)"
  if echo "$out" | grep -q "average rate"; then
    rate="$(echo "$out" | grep "average rate" | tail -1)"
    pass "$topic — $rate"
  else
    fail "$topic — not publishing"
  fi
done

echo ""
echo "=== Position compare (one sample each) ==="
GT="$(timeout 3 ros2 topic echo /model/rrbot/odometry --once 2>/dev/null || true)"
EST="$(timeout 3 ros2 topic echo /visual_tracking/state_estimate --once 2>/dev/null || true)"

gt_x="$(echo "$GT" | awk '/^    x:/{print $2; exit}')"
gt_y="$(echo "$GT" | awk '/^    y:/{print $2; exit}')"
gt_z="$(echo "$GT" | awk '/^    z:/{print $2; exit}')"
est_x="$(echo "$EST" | awk '/^    x:/{print $2; exit}')"
est_y="$(echo "$EST" | awk '/^    y:/{print $2; exit}')"
est_z="$(echo "$EST" | awk '/^    z:/{print $2; exit}')"

if [[ -n "$gt_x" && -n "$est_x" ]]; then
  echo "GT:        x=$gt_x  y=$gt_y  z=$gt_z"
  echo "Estimate:  x=$est_x  y=$est_y  z=$est_z"
  python3 - <<PY
import math
gt = (${gt_x:-0}, ${gt_y:-0}, ${gt_z:-0})
est = (${est_x:-0}, ${est_y:-0}, ${est_z:-0})
err = math.dist(gt, est)
print(f"3D error: {err:.2f} m")
if err < 2.0:
    print("OK: tracking accuracy looks good")
elif err < 5.0:
    print("WARN: moderate error — restart tracking after drone is at z≈5–7 m")
else:
    print("FAIL: large error — visual_tracker likely crashed or initialized on ground")
PY
else
  fail "could not read GT and/or estimate"
fi

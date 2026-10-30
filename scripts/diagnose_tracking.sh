#!/bin/bash
# Diagnose why live tracking nodes fail to start or publish on the VM.
set -eo pipefail

source /opt/ros/humble/setup.bash
source ~/BEFR_SS_2026/install/setup.bash
source ~/BEFR-Project/install/setup.bash

echo "========== 1. Package install =========="
ros2 pkg prefix befr_visual_tracking 2>&1 || echo "FAIL: package not found"
echo ""
echo "Executables:"
ls -la ~/BEFR-Project/install/befr_visual_tracking/lib/befr_visual_tracking/ 2>&1 || echo "FAIL: libexec/lib dir missing"
echo ""
echo "cameras_gazebo.yaml:"
ls -la ~/BEFR-Project/install/befr_visual_tracking/share/befr_visual_tracking/config/cameras_gazebo.yaml 2>&1 || echo "FAIL: config missing"

echo ""
echo "========== 2. Sim / clock =========="
timeout 3 ros2 topic hz /clock 2>&1 || true

echo ""
echo "========== 3. Ground truth =========="
timeout 3 ros2 topic echo /model/rrbot/odometry --once 2>&1 | head -20 || echo "FAIL: no odometry"

echo ""
echo "========== 4. Running nodes =========="
ros2 node list 2>&1 || true

echo ""
echo "========== 5. Tracking topics =========="
ros2 topic list 2>&1 | grep visual_tracking || echo "No /visual_tracking/* topics"

echo ""
echo "========== 6. Try camera_simulator alone (5 s) =========="
echo "If this crashes, read the Python traceback below:"
timeout 5 ros2 run befr_visual_tracking camera_simulator \
  --ros-args \
  -p use_sim_time:=true \
  -p ground_truth_topic:=/model/rrbot/odometry \
  -p cameras_config_path:=/home/student/BEFR-Project/install/befr_visual_tracking/share/befr_visual_tracking/config/cameras_gazebo.yaml \
  2>&1 || true

echo ""
echo "========== 7. Python deps =========="
python3 -c "import numpy; import yaml; import scipy; print('numpy', numpy.__version__)" 2>&1 || echo "FAIL: Python import error"

echo ""
echo "========== Done =========="
echo "Paste this full output when asking for help."

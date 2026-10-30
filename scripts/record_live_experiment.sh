#!/bin/bash
# Record a live Gazebo tracking dataset (run while sim + square_ctrl are active).
set -eo pipefail

# ROS setup.bash uses unset vars; do not enable nounset (-u) before sourcing.
source /opt/ros/humble/setup.bash
source ~/BEFR_SS_2026/install/setup.bash
source ~/BEFR-Project/install/setup.bash

OUTPUT_DIR="${1:-$HOME/BEFR-Project/results/live_slow_circle}"
DURATION="${2:-21.0}"

mkdir -p "$OUTPUT_DIR"

echo "Recording to $OUTPUT_DIR for ${DURATION}s"
echo "Ensure: Gazebo playing, square_ctrl running, drone near z≈1.5 m"

ros2 launch befr_visual_tracking record_experiment.launch.py \
    use_sim_time:=true \
    ground_truth_topic:=/model/rrbot/odometry \
    output_dir:="$OUTPUT_DIR" \
    record_duration_s:="$DURATION" \
    record_bag:=true

echo "Done. Copy to Mac if needed:"
echo "  scp -r student@<vm-ip>:$OUTPUT_DIR ./results/"

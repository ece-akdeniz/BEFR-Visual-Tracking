# BEFR Visual Tracking — VM Startup Cheat Sheet

Ubuntu 22.04 VM, ROS 2 Humble, Ignition Gazebo 6.16.

Two folders:

| Folder | Purpose |
|--------|---------|
| `~/BEFR_SS_2026` | Course Gazebo sim (rrbot) |
| `~/BEFR-Project` | Visual tracking package |

## One-time setup (already done if built)

```bash
# Course sim
cd ~/BEFR_SS_2026
git submodule update --init --recursive
sudo apt install -y ros-humble-gz-ros2-control-demos ros-humble-tf-transformations
pip3 install 'numpy<2' --user
colcon build --symlink-install

# Visual tracking
cd ~/BEFR-Project
pip3 install -r requirements.txt
colcon build --packages-select befr_visual_tracking --symlink-install
```

## Every terminal — source order

```bash
source /opt/ros/humble/setup.bash
source ~/BEFR_SS_2026/install/setup.bash
source ~/BEFR-Project/install/setup.bash   # only for tracking terminals
```

## 4-terminal live demo

### Terminal 1 — Gazebo (leave running)

```bash
source /opt/ros/humble/setup.bash
source ~/BEFR_SS_2026/install/setup.bash
ros2 launch rrbot_gazebo rrbot_world.launch.py
```

Wait until all 4 motor controllers show **Configured and activated**. Then press **Play** in Gazebo.

### Terminal 2 — Visual tracking (start after Gazebo Play)

```bash
source /opt/ros/humble/setup.bash
source ~/BEFR_SS_2026/install/setup.bash
source ~/BEFR-Project/install/setup.bash
ros2 topic hz /clock          # must show steady rate before continuing

ros2 launch befr_visual_tracking tracking.launch.py \
    filter_type:=ekf \
    launch_camera_simulator:=true \
    use_sim_time:=true \
    ground_truth_topic:=/model/rrbot/odometry
```

Uses **`cameras_gazebo.yaml`** by default (flight volume center z = 5 m, wider 8 m camera ring) so tracking works at square_ctrl altitudes (~3–8 m).

### Terminal 3 — Fly the drone

```bash
source /opt/ros/humble/setup.bash
source ~/BEFR_SS_2026/install/setup.bash
ros2 run square_ctrl square_ctrl
```

No need to chase z ≈ 1.5 m anymore — start tracking first, then launch square_ctrl.

### Terminal 4 — Monitor

```bash
source /opt/ros/humble/setup.bash
source ~/BEFR_SS_2026/install/setup.bash
source ~/BEFR-Project/install/setup.bash

ros2 topic hz /visual_tracking/camera_0/detection
ros2 topic echo /model/rrbot/odometry --field pose.pose.position
ros2 topic echo /visual_tracking/state_estimate --field pose.pose.position
```

## Live dataset recording

With Gazebo + flying drone + stable tracking:

```bash
source /opt/ros/humble/setup.bash
source ~/BEFR_SS_2026/install/setup.bash
source ~/BEFR-Project/install/setup.bash

mkdir -p ~/BEFR-Project/results/live_slow_circle

ros2 launch befr_visual_tracking record_experiment.launch.py \
    use_sim_time:=true \
    ground_truth_topic:=/model/rrbot/odometry \
    output_dir:=~/BEFR-Project/results/live_slow_circle \
    record_duration_s:=21.0 \
    record_bag:=true
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Package 'square_ctrl' not found` | `colcon build` in `~/BEFR_SS_2026`, then `source install/setup.bash` |
| `libexec directory ... does not exist` | Ensure `setup.cfg` exists in `~/BEFR-Project`, rebuild project |
| `Cannot predict backwards in time` | Restart tracking after `/clock` is publishing; never start tracking before Gazebo Play |
| Motor controller errors | Kill all Gazebo: `killall -9 gz ruby ros2`; relaunch once |
| Odometry empty | Apply Ignition plugin names in `rrbot.xacro`, rebuild `rrbot_gazebo` |
| NumPy / tf_transformations crash | `pip3 install 'numpy<2' --user` |
| Estimate diverges badly | Confirm `cameras_gazebo.yaml` is used (default in launch); check GT with `ros2 topic echo /model/rrbot/odometry` |

## Clean restart (when controllers break)

```bash
killall -9 gz gzserver gzclient ruby ros2 rviz2 2>/dev/null
sleep 2
# then start Terminal 1 again
```

## Offline work (Mac or VM — no Gazebo)

```bash
cd ~/BEFR-Project
PYTHONPATH=. python -m befr_visual_tracking.run_experiments --list
PYTHONPATH=. python -m befr_visual_tracking.run_experiments --experiment 0
PYTHONPATH=. python -m befr_visual_tracking.run_experiments --experiment 1
PYTHONPATH=. streamlit run dashboard/streamlit_app.py
```

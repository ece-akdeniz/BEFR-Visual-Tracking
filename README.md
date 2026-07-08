# BEFR Visual Tracking

Multi-camera visual tracking for a simulated quadcopter (Task 1 + advanced camera calibration).

**Authors:** Ece Akdeniz, Lars Wolfgramm-Kuntz  
**Course:** Bayesian Estimation for Flight Robotics (SoSe 2026)  
**Repository:** [github.com/ece-akdeniz/BEFR-Visual-Tracking](https://github.com/ece-akdeniz/BEFR-Visual-Tracking)

## Contents

| Path | Description |
|------|-------------|
| `befr_visual_tracking/` | Core package: camera model, triangulation, KF/EKF/UKF, calibration |
| `dashboard/` | Streamlit app for offline experiments and plots |
| `results/` | Saved experiments and calibration datasets |
| `presentation/` | Slides (`presentation.pptx`) and figure exports |
| `launch/` | ROS 2 launch files |
| `tests/` | Unit tests |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Dashboard

```bash
source .venv/bin/activate
PYTHONPATH=. streamlit run dashboard/streamlit_app.py
```

Open **http://localhost:8501** in a browser.

- **Interactive tracker** — load saved runs from `results/experiment_NNN/` or rerun filters offline
- **3D visualisation / Time plots** — trajectory and error over time
- **Filter comparison** — KF vs EKF vs UKF (e.g. `experiment_005`, `006`, `007`)
- **Advanced calibration** — bundle adjustment on calibration manoeuvres

## Offline experiments

```bash
PYTHONPATH=. python -m befr_visual_tracking.run_experiments --experiment 1
PYTHONPATH=. python -m befr_visual_tracking.run_calibration results/calibration_figure_eight_3d
```

See `OFFLINE_RESULTS.md` for a short summary of experiment runs.

## Presentation figures

| Location | Figures |
|----------|---------|
| `presentation/plots/advanced/` | Advanced task 3D frames (unknown poses, BA, …) |
| `presentation/plots/observability/` | Exp 101 manoeuvre comparison |
| `befr_visual_tracking/notebooks/final_vis_figures/three_camera_takeoff_circles4/` | Task 1 filter visualisations (Plotly HTML) |
| `befr_visual_tracking/notebooks/final_vis_figures/calibration_figure_eight_3d/` | Calibration task visualisations |
| `befr_visual_tracking/notebooks/final_vis_figures/calibration_figure_eight_3d_advanced/` | Advanced calibration figures |

Regenerate Plotly figures:

```bash
python befr_visual_tracking/notebooks/run_final_vis.py \
  --results-dir results/three_camera_takeoff_circles4

python befr_visual_tracking/notebooks/generate_advanced_lars_figures.py
python befr_visual_tracking/notebooks/generate_observability_manoeuvres_figure.py
```

Notebook source: `befr_visual_tracking/notebooks/final_vis.ipynb`

## ROS 2 (optional)

```bash
colcon build --packages-select befr_visual_tracking
source install/setup.bash
ros2 launch befr_visual_tracking tracking.launch.py
```

Camera and tracking configs are generated per dataset under `results/*/camera_config.yaml`.

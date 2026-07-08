from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from befr_visual_tracking.advanced.calibration_pipeline import run_calibration_pipeline
from befr_visual_tracking.advanced.gauge import apply_gauge_fix_to_cameras
from befr_visual_tracking.camera_model import cameras_from_yaml
from befr_visual_tracking.generate_canonical_dataset import default_results_dir, generate_canonical_dataset
from dashboard.lib.session import init_session_state

init_session_state()
st.title("Advanced camera calibration")

st.markdown(
    """
Only **Camera 0** pose is fixed. Other camera poses and the calibration trajectory are estimated
from synchronized pixels using essential-matrix initialisation and batch bundle adjustment.
"""
)

MANOEUVRE_TO_EXPERIMENT = {
    "3D figure-eight": "calibration_figure_eight_3d",
    "planar figure-eight": "calibration_planar_figure_eight",
    "straight line": "calibration_straight_line",
    "hover": "calibration_stationary_hover",
}

manoeuvre = st.selectbox("Calibration manoeuvre", list(MANOEUVRE_TO_EXPERIMENT.keys()))
active_cameras = st.multiselect(
    "Active cameras",
    ["camera_0", "camera_1", "camera_2", "camera_3"],
    default=["camera_0", "camera_1", "camera_2"],
)

if st.button("Run calibration", type="primary"):
    experiment_name = MANOEUVRE_TO_EXPERIMENT[manoeuvre]
    results_root = default_results_dir()
    folder_names = {
        "calibration_figure_eight_3d": "calibration_figure_eight_3d",
        "calibration_planar_figure_eight": "calibration_planar_figure_eight",
        "calibration_straight_line": "calibration_straight_line",
        "calibration_stationary_hover": "calibration_stationary_hover",
    }
    dataset_dir = results_root / folder_names[experiment_name]

    if not (dataset_dir / "detections.csv").is_file():
        with st.spinner("Generating calibration dataset..."):
            generate_canonical_dataset(experiment_name=experiment_name, output_dir=results_root)

    with st.spinner("Running bundle adjustment..."):
        result = run_calibration_pipeline(
            dataset_dir,
            active_cameras=active_cameras,
            use_opencv_init=False,
            use_metric_scale_from_ground_truth=True,
            subsample_stride=5,
            max_timesteps=80,
        )
    st.session_state.calibration_result = result
    st.success(f"Calibration complete: {result.calibrated_config_path}")

if "calibration_result" not in st.session_state:
    st.info("Run calibration to compare true and optimised camera poses.")
    st.stop()

result = st.session_state.calibration_result
before = result.before_optimization
after = result.after_optimization

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trajectory RMSE aligned (before)", f"{before.trajectory_rmse_aligned:.3f} m")
c2.metric("Trajectory RMSE aligned (after)", f"{after.trajectory_rmse_aligned:.3f} m")
c3.metric("Camera translation RMSE (after)", f"{after.camera_translation_rmse:.3f} m")
c4.metric("Reprojection RMSE (after)", f"{after.reprojection_rmse:.3f} px")

_, true_cameras = cameras_from_yaml(result.experiment_dir / "camera_config.yaml")
true_gauge = apply_gauge_fix_to_cameras({camera.name: camera for camera in true_cameras})
_, est_cameras = cameras_from_yaml(result.calibrated_config_path)
est_by_id = {camera.name: camera for camera in est_cameras}

fig = go.Figure()
for label, cameras, color in (("True", true_gauge, "green"), ("Optimised", est_by_id, "orange")):
    fig.add_trace(
        go.Scatter3d(
            x=[cam.extrinsics.t_camera_to_world[0] for cam in cameras.values()],
            y=[cam.extrinsics.t_camera_to_world[1] for cam in cameras.values()],
            z=[cam.extrinsics.t_camera_to_world[2] for cam in cameras.values()],
            mode="markers+text",
            marker={"size": 5, "color": color},
            text=list(cameras.keys()),
            name=label,
        )
    )
fig.update_layout(scene={"aspectmode": "data"}, height=600, title="Camera positions in camera_0 gauge")
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Richer manoeuvres (3D figure-eight) improve parallax. "
    f"Estimated similarity scale vs ground truth: {after.scale_estimate:.3f}."
)

import numpy as np
import streamlit as st

from dashboard.lib.plots_3d import build_scene_figure
from dashboard.lib.session import current_cameras, current_steps, init_session_state

init_session_state()
st.title("3D visualisation")

steps = current_steps()
cameras = current_cameras()
if not steps or not cameras:
    st.info("Run the interactive tracker or load a saved experiment first.")
    st.stop()

index = st.slider("Timestep", 0, len(steps) - 1, st.session_state.selected_timestep_index)
st.session_state.selected_timestep_index = index
step = steps[index]

ground_truth = np.array([s.ground_truth_position for s in steps], dtype=float)
estimate = np.array([s.state[:3] for s in steps], dtype=float)

detections = {}
if st.session_state.interactive_result is not None:
    detections_at_time = st.session_state.interactive_result.detections_by_time.get(step.timestamp, [])
    detections = {d.camera_id: (d.u, d.v) for d in detections_at_time}

active_ids = list(detections.keys()) if detections else list(cameras.keys())
fig = build_scene_figure(
    cameras_by_id=cameras,
    ground_truth_positions=ground_truth,
    estimated_positions=estimate,
    current_step=step,
    active_camera_ids=active_ids,
    detections=detections or None,
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"t = {step.timestamp:.2f} s, visible cameras = {step.num_visible_cameras}, "
    f"position error = {np.linalg.norm(step.state[:3] - step.ground_truth_position):.3f} m"
)

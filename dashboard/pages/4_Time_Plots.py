import streamlit as st

from dashboard.lib.plots_time import build_time_series_figure
from dashboard.lib.session import current_metrics, current_steps, init_session_state

init_session_state()
st.title("Time plots")

steps = current_steps()
if not steps:
    st.info("Run the interactive tracker or load a saved experiment first.")
    st.stop()

metrics = current_metrics()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Position RMSE", f"{metrics.get('position_rmse', 0.0):.4f}")
col2.metric("Velocity RMSE", f"{metrics.get('velocity_rmse', 0.0):.4f}")
col3.metric("Max position error", f"{metrics.get('position_max_error', 0.0):.4f}")
col4.metric("Filter rate [Hz]", f"{metrics.get('filter_frequency_hz', 0.0):.1f}")

st.plotly_chart(build_time_series_figure(steps), use_container_width=True)

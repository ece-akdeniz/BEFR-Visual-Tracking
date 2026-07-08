"""BEFR Visual Tracking Dashboard."""

from __future__ import annotations

import streamlit as st

from dashboard.lib.session import init_session_state

st.set_page_config(
    page_title="BEFR Visual Tracking",
    page_icon="🛸",
    layout="wide",
)

init_session_state()

st.title("BEFR Visual Tracking Dashboard")
st.caption("Multi-camera Bayesian state estimation for quadcopter tracking")

st.markdown(
    """
This dashboard reuses the same pure-Python camera and filter modules as the ROS nodes.

**Mode A — Saved experiment viewer:** load completed `results/experiment_NNN/` folders.

**Mode B — Interactive offline rerun:** regenerate noisy detections and replay KF / EKF / UKF instantly.
"""
)

col1, col2 = st.columns(2)
with col1:
    st.info(f"Current mode: **{st.session_state.dashboard_mode}**")
with col2:
    steps = st.session_state.get("loaded_experiment") or st.session_state.get("interactive_result")
    st.success("Data loaded" if steps else "Use Interactive Tracker or load a saved experiment")

st.page_link("pages/1_Project_Overview.py", label="Project overview", icon="📘")
st.page_link("pages/2_Interactive_Tracker.py", label="Interactive tracker", icon="🎛️")
st.page_link("pages/3_3D_Visualisation.py", label="3D visualisation", icon="🧊")
st.page_link("pages/4_Time_Plots.py", label="Time plots", icon="📈")
st.page_link("pages/5_Filter_Comparison.py", label="Filter comparison", icon="⚖️")
st.page_link("pages/6_Methods.py", label="How each method works", icon="🧠")
st.page_link("pages/7_Advanced_Calibration.py", label="Advanced calibration", icon="📷")

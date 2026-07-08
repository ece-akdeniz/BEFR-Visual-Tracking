import streamlit as st

from dashboard.lib.session import init_session_state

init_session_state()
st.title("Project overview")

st.markdown(
    """
## Task 1 — Multi-camera visual tracking

Estimate the 6D quadcopter state from noisy pixel detections:

\\[
\\mathbf{x}_k = [p_x, p_y, p_z, v_x, v_y, v_z]^T
\\]

### Camera setup
- Four fixed pinhole cameras around the flight volume
- Known intrinsics, synchronized detections at 30 Hz
- Camera 0 defines the reference frame for the advanced task

### Three estimation methods
| Method | Measurement model | Update |
|---|---|---|
| **KF** | Triangulated 3D position | Linear Kalman filter |
| **EKF** | Raw pixels | Extended Kalman filter with projection Jacobian |
| **UKF** | Raw pixels | Unscented Kalman filter with sigma points |

### Basic versus advanced task
**Basic task:** camera poses are known. Compare KF, EKF, and UKF under noise, dropout, and geometry changes.

**Advanced task:** only Camera 0 pose and all intrinsics are known. Jointly estimate other camera poses and the calibration trajectory using batch bundle adjustment, then reuse the online filters with calibrated extrinsics.
"""
)

st.subheader("Pipeline")
st.code(
    """
cameras.yaml + trajectory
        ↓
detections.csv + ground_truth.csv
        ↓
KF / EKF / UKF  (same modules as ROS visual_tracker)
        ↓
estimates.csv + metrics.json
    """,
    language="text",
)

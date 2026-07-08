import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.lib.experiment_loader import list_saved_experiments, load_saved_experiment
from dashboard.lib.offline_rerun import InteractiveRunConfig, compare_filters
from dashboard.lib.session import init_session_state

init_session_state()
st.title("Filter comparison")

source = st.radio("Comparison source", ["Interactive rerun", "Saved experiments"], horizontal=True)

rows = []
if source == "Interactive rerun":
    col1, col2 = st.columns(2)
    with col1:
        trajectory = st.selectbox(
            "Trajectory",
            ["circle", "square", "planar figure-eight", "3d figure-eight"],
            key="cmp_traj",
        )
        pixel_noise = st.slider("Pixel noise σ [px]", 0.0, 5.0, 1.0, 0.1, key="cmp_noise")
    with col2:
        active_cameras = st.multiselect(
            "Active cameras",
            ["camera_0", "camera_1", "camera_2", "camera_3"],
            default=["camera_0", "camera_1", "camera_2"],
            key="cmp_cams",
        )
        dropout = st.selectbox(
            "Dropout",
            ["none", "one camera", "multiple cameras", "all cameras temporarily"],
            key="cmp_drop",
        )

    if st.button("Compare KF / EKF / UKF"):
        config = InteractiveRunConfig(
            filter_type="ekf",
            trajectory=trajectory,
            active_cameras=active_cameras,
            pixel_noise_std=pixel_noise,
            dropout_mode=dropout,
        )
        results = compare_filters(config, ["kf", "ekf", "ukf"])
        st.session_state.comparison_results = results
        for result in results:
            rows.append(
                {
                    "Filter": result.config.filter_type.upper(),
                    "Position RMSE": result.metrics["position_rmse"],
                    "Velocity RMSE": result.metrics["velocity_rmse"],
                    "Mean runtime [ms]": 1e3 * result.metrics["mean_update_time_sec"],
                    "Recovery [s]": result.metrics.get("recovery_time_sec"),
                }
            )
else:
    experiments = list_saved_experiments()
    selected = st.multiselect("Saved runs", [p.name for p in experiments])
    for name in selected:
        exp = load_saved_experiment(experiments[[p.name for p in experiments].index(name)])
        rows.append(
            {
                "Filter": exp.metadata.get("filter", "n/a").upper(),
                "Position RMSE": exp.metrics.get("position_rmse"),
                "Velocity RMSE": exp.metrics.get("velocity_rmse"),
                "Mean runtime [ms]": 1e3 * exp.metrics.get("mean_update_time_sec", 0.0),
                "Recovery [s]": exp.metrics.get("recovery_time_sec"),
            }
        )

if rows:
    table = pd.DataFrame(rows)
    st.dataframe(table, use_container_width=True)
    fig = px.bar(table, x="Filter", y="Position RMSE", color="Filter", title="Position RMSE by filter")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.bar(
        table,
        x="Filter",
        y="Mean runtime [ms]",
        color="Filter",
        title="Mean update runtime by filter",
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Configure a scenario and run the comparison.")

st.subheader("Experiment sweep plots")
saved = list_saved_experiments()
if saved:
    sweep_rows = []
    for path in saved:
        exp = load_saved_experiment(path)
        sweep_rows.append(
            {
                "experiment": path.name,
                "filter": exp.metadata.get("filter", "n/a"),
                "pixel_noise_std": exp.metadata.get("pixel_noise_std"),
                "num_cameras": len(exp.metadata.get("active_cameras", [])),
                "position_rmse": exp.metrics.get("position_rmse"),
            }
        )
    sweep = pd.DataFrame(sweep_rows)
    if sweep["pixel_noise_std"].notna().any():
        st.plotly_chart(
            px.scatter(
                sweep,
                x="pixel_noise_std",
                y="position_rmse",
                color="filter",
                title="Error versus pixel noise",
            ),
            use_container_width=True,
        )
    if sweep["num_cameras"].notna().any():
        st.plotly_chart(
            px.scatter(
                sweep,
                x="num_cameras",
                y="position_rmse",
                color="filter",
                title="Error versus camera count",
            ),
            use_container_width=True,
        )

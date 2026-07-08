import streamlit as st

from dashboard.lib.experiment_loader import list_saved_experiments, load_saved_experiment
from dashboard.lib.offline_rerun import InteractiveRunConfig, run_interactive_tracking
from dashboard.lib.session import init_session_state, set_interactive_result, set_loaded_experiment

init_session_state()
st.title("Interactive tracker")

mode = st.radio(
    "Dashboard mode",
    ["Interactive offline rerun", "Saved experiment viewer"],
    horizontal=True,
)

if mode == "Saved experiment viewer":
    experiments = list_saved_experiments()
    if not experiments:
        st.warning("No saved experiments found in `results/experiment_NNN/`.")
    else:
        labels = [path.name for path in experiments]
        selected = st.selectbox("Saved experiment", labels)
        if st.button("Load experiment"):
            loaded = load_saved_experiment(experiments[labels.index(selected)])
            set_loaded_experiment(loaded)
            st.success(f"Loaded {loaded.path.name}")
    if st.session_state.loaded_experiment is not None:
        exp = st.session_state.loaded_experiment
        st.json(exp.metadata)
        st.metric("Position RMSE [m]", exp.metrics.get("position_rmse", "n/a"))
        st.metric("Velocity RMSE [m/s]", exp.metrics.get("velocity_rmse", "n/a"))
else:
    col1, col2 = st.columns(2)
    with col1:
        filter_type = st.selectbox("Filter", ["kf", "ekf", "ukf"], index=1)
        trajectory = st.selectbox(
            "Trajectory",
            ["circle", "square", "planar figure-eight", "3d figure-eight"],
        )
        pixel_noise = st.slider("Pixel noise σ [px]", 0.0, 5.0, 1.0, 0.1)
    with col2:
        active_cameras = st.multiselect(
            "Active cameras",
            ["camera_0", "camera_1", "camera_2", "camera_3"],
            default=["camera_0", "camera_1", "camera_2"],
        )
        dropout = st.selectbox(
            "Dropout",
            ["none", "one camera", "multiple cameras", "all cameras temporarily"],
        )
        random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    if st.button("Run offline tracker", type="primary"):
        if len(active_cameras) < 2 and filter_type == "kf":
            st.error("KF requires at least two active cameras.")
        else:
            result = run_interactive_tracking(
                InteractiveRunConfig(
                    filter_type=filter_type,
                    trajectory=trajectory,
                    active_cameras=active_cameras,
                    pixel_noise_std=pixel_noise,
                    dropout_mode=dropout,
                    random_seed=int(random_seed),
                )
            )
            set_interactive_result(result)
            st.success(
                f"Completed {len(result.replay.steps)} steps in "
                f"{result.metrics.get('total_runtime_sec', 0.0):.3f} s"
            )

    if st.session_state.interactive_result is not None:
        metrics = st.session_state.interactive_result.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Position RMSE [m]", f"{metrics.get('position_rmse', 0.0):.4f}")
        c2.metric("Velocity RMSE [m/s]", f"{metrics.get('velocity_rmse', 0.0):.4f}")
        c3.metric("Mean update [ms]", f"{1e3 * metrics.get('mean_update_time_sec', 0.0):.3f}")
        c4.metric("Recovery [s]", metrics.get("recovery_time_sec") or "n/a")

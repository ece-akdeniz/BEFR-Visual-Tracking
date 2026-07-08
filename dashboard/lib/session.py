"""Shared Streamlit session helpers."""

from __future__ import annotations

import streamlit as st

from dashboard.lib.experiment_loader import LoadedExperiment, load_saved_experiment
from dashboard.lib.offline_rerun import InteractiveRunResult


def init_session_state() -> None:
    defaults = {
        "dashboard_mode": "Interactive offline rerun",
        "loaded_experiment": None,
        "interactive_result": None,
        "comparison_results": None,
        "selected_timestep_index": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_steps():
    if st.session_state.loaded_experiment is not None:
        return st.session_state.loaded_experiment.steps
    if st.session_state.interactive_result is not None:
        return st.session_state.interactive_result.replay.steps
    return []


def current_cameras():
    if st.session_state.loaded_experiment is not None:
        return st.session_state.loaded_experiment.cameras_by_id
    if st.session_state.interactive_result is not None:
        return st.session_state.interactive_result.cameras_by_id
    return {}


def current_metrics() -> dict:
    if st.session_state.loaded_experiment is not None:
        return st.session_state.loaded_experiment.metrics
    if st.session_state.interactive_result is not None:
        return st.session_state.interactive_result.metrics
    return {}


def set_loaded_experiment(experiment: LoadedExperiment) -> None:
    st.session_state.dashboard_mode = "Saved experiment viewer"
    st.session_state.loaded_experiment = experiment
    st.session_state.interactive_result = None
    st.session_state.comparison_results = None


def set_interactive_result(result: InteractiveRunResult) -> None:
    st.session_state.dashboard_mode = "Interactive offline rerun"
    st.session_state.interactive_result = result
    st.session_state.loaded_experiment = None
    st.session_state.comparison_results = None

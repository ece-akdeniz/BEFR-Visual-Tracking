"""Time-series Plotly charts for the dashboard."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from befr_visual_tracking.experiments.metrics import StepRecord


def _times(steps: list[StepRecord]) -> list[float]:
    return [step.timestamp for step in steps]


def build_time_series_figure(steps: list[StepRecord]) -> go.Figure:
    if not steps:
        return go.Figure()

    times = _times(steps)
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Position",
            "Velocity",
            "Position error",
            "Velocity error",
            "Position covariance",
            "Active cameras",
        ),
        vertical_spacing=0.08,
    )

    for index, axis in enumerate(("x", "y", "z")):
        gt = [step.ground_truth_position[index] for step in steps]
        est = [step.state[index] for step in steps]
        fig.add_trace(
            go.Scatter(x=times, y=gt, name=f"gt_{axis}", line={"dash": "dash"}),
            row=1,
            col=1,
        )
        fig.add_trace(go.Scatter(x=times, y=est, name=f"est_{axis}"), row=1, col=1)

    for index, axis in enumerate((3, 4, 5)):
        gt = [step.ground_truth_velocity[index - 3] for step in steps]
        est = [step.state[index] for step in steps]
        fig.add_trace(go.Scatter(x=times, y=gt, showlegend=False, line={"dash": "dash"}), row=1, col=2)
        fig.add_trace(go.Scatter(x=times, y=est, showlegend=False), row=1, col=2)

    pos_err = [
        float(((step.state[:3] - step.ground_truth_position) ** 2).sum()) ** 0.5 for step in steps
    ]
    vel_err = [
        float(((step.state[3:] - step.ground_truth_velocity) ** 2).sum()) ** 0.5 for step in steps
    ]
    fig.add_trace(go.Scatter(x=times, y=pos_err, name="position error"), row=2, col=1)
    fig.add_trace(go.Scatter(x=times, y=vel_err, name="velocity error"), row=2, col=2)

    cov_trace = [step.covariance[0, 0] + step.covariance[1, 1] + step.covariance[2, 2] for step in steps]
    fig.add_trace(go.Scatter(x=times, y=cov_trace, name="covariance sum"), row=3, col=1)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=[step.num_visible_cameras for step in steps],
            name="visible cameras",
        ),
        row=3,
        col=2,
    )

    fig.update_layout(height=900, showlegend=False)
    return fig

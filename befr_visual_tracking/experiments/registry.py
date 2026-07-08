"""Experiment plan registry — one variable changed at a time."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrackingRunSpec:
    filter_type: str
    active_cameras: list[str]
    acceleration_noise_std: float = 0.5
    pixel_noise_std: float = 1.0
    innovation_gate: float = 9.21


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    dataset_overrides: dict = field(default_factory=dict)
    tracking_overrides: dict = field(default_factory=dict)
    metadata_overrides: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: int
    title: str
    description: str
    experiment_type: str  # tracking | calibration
    variants: tuple[ExperimentVariant, ...]
    filters: tuple[str, ...] = ("kf", "ekf", "ukf")
    shared_dataset: bool = False


def default_cameras_config(name: str = "cameras.yaml") -> Path:
    return Path(__file__).resolve().parents[2] / "config" / name


EXPERIMENT_PLANS: dict[int, ExperimentPlan] = {
    0: ExperimentPlan(
        experiment_id=0,
        title="Mathematical sanity test",
        description="Zero noise, all cameras, slow trajectory — verify geometry before tuning noise.",
        experiment_type="tracking",
        filters=("kf", "ekf", "ukf"),
        variants=(
            ExperimentVariant(
                name="sanity",
                dataset_overrides={
                    "trajectory": "slow_circle",
                    "use_zero_noise": True,
                    "quantize": False,
                },
                tracking_overrides={"active_cameras": ["camera_0", "camera_1", "camera_2", "camera_3"]},
                metadata_overrides={"dropout_scenario": "none"},
            ),
        ),
    ),
    1: ExperimentPlan(
        experiment_id=1,
        title="Main filter comparison",
        description="Fixed three-camera setup, one-pixel noise, no dropout.",
        experiment_type="tracking",
        filters=("kf", "ekf", "ukf"),
        shared_dataset=True,
        variants=(
            ExperimentVariant(
                name="main_comparison",
                dataset_overrides={"trajectory": "slow_circle", "pixel_noise_std": 1.0},
                tracking_overrides={"active_cameras": ["camera_0", "camera_1", "camera_2"]},
                metadata_overrides={"dropout_scenario": "none"},
            ),
        ),
    ),
    2: ExperimentPlan(
        experiment_id=2,
        title="Number of cameras",
        description="Two, three, and four active cameras on the same detections.",
        experiment_type="tracking",
        filters=("ekf",),
        shared_dataset=True,
        variants=(
            ExperimentVariant(
                name="two_cameras",
                tracking_overrides={"active_cameras": ["camera_0", "camera_1"]},
            ),
            ExperimentVariant(
                name="three_cameras",
                tracking_overrides={"active_cameras": ["camera_0", "camera_1", "camera_2"]},
            ),
            ExperimentVariant(
                name="four_cameras",
                tracking_overrides={"active_cameras": ["camera_0", "camera_1", "camera_2", "camera_3"]},
            ),
        ),
    ),
    3: ExperimentPlan(
        experiment_id=3,
        title="Pixel noise",
        description="Evaluate accuracy and covariance growth for several noise levels.",
        experiment_type="tracking",
        filters=("ekf",),
        variants=(
            ExperimentVariant(name="noise_0p5", dataset_overrides={"pixel_noise_std": 0.5}),
            ExperimentVariant(name="noise_1p0", dataset_overrides={"pixel_noise_std": 1.0}),
            ExperimentVariant(name="noise_2p0", dataset_overrides={"pixel_noise_std": 2.0}),
            ExperimentVariant(name="noise_5p0", dataset_overrides={"pixel_noise_std": 5.0}),
        ),
    ),
    4: ExperimentPlan(
        experiment_id=4,
        title="Camera dropout",
        description="Planned dropout and recovery sequence.",
        experiment_type="tracking",
        filters=("kf", "ekf", "ukf"),
        variants=(
            ExperimentVariant(
                name="planned_dropout",
                dataset_overrides={"dropout_scenario": "planned_sequence"},
                metadata_overrides={"dropout_scenario": "planned_sequence"},
            ),
        ),
    ),
    5: ExperimentPlan(
        experiment_id=5,
        title="Camera geometry",
        description="Wide baseline, narrow baseline, and nearly parallel viewing directions.",
        experiment_type="tracking",
        filters=("ekf",),
        variants=(
            ExperimentVariant(
                name="wide_baseline",
                dataset_overrides={"cameras_config_path": str(default_cameras_config("cameras_wide_baseline.yaml"))},
            ),
            ExperimentVariant(
                name="narrow_baseline",
                dataset_overrides={"cameras_config_path": str(default_cameras_config("cameras_narrow_baseline.yaml"))},
            ),
            ExperimentVariant(
                name="parallel_view",
                dataset_overrides={"cameras_config_path": str(default_cameras_config("cameras_parallel.yaml"))},
            ),
        ),
    ),
    6: ExperimentPlan(
        experiment_id=6,
        title="Motion-model mismatch",
        description="Test the constant-velocity process model on different manoeuvres.",
        experiment_type="tracking",
        filters=("ekf",),
        variants=(
            ExperimentVariant(name="slow_circle", dataset_overrides={"trajectory": "slow_circle"}),
            ExperimentVariant(
                name="planar_figure_eight",
                dataset_overrides={"trajectory": "planar_figure_eight"},
            ),
            ExperimentVariant(name="straight_line", dataset_overrides={"trajectory": "straight_line"}),
            ExperimentVariant(
                name="calibration_figure_eight_3d",
                dataset_overrides={"trajectory": "calibration_figure_eight_3d"},
            ),
        ),
    ),
    7: ExperimentPlan(
        experiment_id=7,
        title="Target distance",
        description="Near, medium, and far motion relative to the camera cluster.",
        experiment_type="tracking",
        filters=("ekf",),
        variants=(
            ExperimentVariant(
                name="near",
                dataset_overrides={"trajectory": "slow_circle", "trajectory_kwargs": {"radius": 0.4}},
            ),
            ExperimentVariant(
                name="medium",
                dataset_overrides={"trajectory": "slow_circle", "trajectory_kwargs": {"radius": 1.0}},
            ),
            ExperimentVariant(
                name="far",
                dataset_overrides={"trajectory": "slow_circle", "trajectory_kwargs": {"radius": 1.8}},
            ),
        ),
    ),
    101: ExperimentPlan(
        experiment_id=101,
        title="Advanced — manoeuvre observability",
        description="Compare camera-pose reconstruction across calibration trajectories.",
        experiment_type="calibration",
        filters=(),
        variants=(
            ExperimentVariant(name="hover", dataset_overrides={"trajectory": "stationary_hover"}),
            ExperimentVariant(name="line", dataset_overrides={"trajectory": "straight_line"}),
            ExperimentVariant(
                name="planar_eight",
                dataset_overrides={"trajectory": "planar_figure_eight"},
            ),
            ExperimentVariant(
                name="figure_eight_3d",
                dataset_overrides={"trajectory": "calibration_figure_eight_3d"},
            ),
        ),
    ),
    102: ExperimentPlan(
        experiment_id=102,
        title="Advanced — calibration camera count",
        description="Two, three, and four cameras for joint calibration.",
        experiment_type="calibration",
        filters=(),
        variants=(
            ExperimentVariant(
                name="two_cameras",
                metadata_overrides={"active_cameras": ["camera_0", "camera_1"]},
            ),
            ExperimentVariant(
                name="three_cameras",
                metadata_overrides={"active_cameras": ["camera_0", "camera_1", "camera_2"]},
            ),
            ExperimentVariant(
                name="four_cameras",
                metadata_overrides={"active_cameras": ["camera_0", "camera_1", "camera_2", "camera_3"]},
            ),
        ),
    ),
    103: ExperimentPlan(
        experiment_id=103,
        title="Advanced — calibration noise",
        description="Pixel-noise levels for joint calibration accuracy.",
        experiment_type="calibration",
        filters=(),
        variants=(
            ExperimentVariant(
                name="noise_0p5",
                dataset_overrides={"pixel_noise_std": 0.5, "trajectory": "calibration_figure_eight_3d"},
            ),
            ExperimentVariant(
                name="noise_1p0",
                dataset_overrides={"pixel_noise_std": 1.0, "trajectory": "calibration_figure_eight_3d"},
            ),
            ExperimentVariant(
                name="noise_2p0",
                dataset_overrides={"pixel_noise_std": 2.0, "trajectory": "calibration_figure_eight_3d"},
            ),
        ),
    ),
    104: ExperimentPlan(
        experiment_id=104,
        title="Advanced — impact on tracking",
        description="Compare tracking with true, initial, and optimised camera poses.",
        experiment_type="advanced_tracking",
        filters=("kf", "ekf", "ukf"),
        shared_dataset=True,
        variants=(
            ExperimentVariant(
                name="true_poses",
                dataset_overrides={"trajectory": "calibration_figure_eight_3d"},
                metadata_overrides={"camera_pose_source": "true"},
            ),
            ExperimentVariant(
                name="initial_poses",
                dataset_overrides={"trajectory": "calibration_figure_eight_3d"},
                metadata_overrides={"camera_pose_source": "initial"},
            ),
            ExperimentVariant(
                name="optimised_poses",
                dataset_overrides={"trajectory": "calibration_figure_eight_3d"},
                metadata_overrides={"camera_pose_source": "optimised"},
            ),
        ),
    ),
}


def get_experiment_plan(experiment_id: int) -> ExperimentPlan:
    try:
        return EXPERIMENT_PLANS[experiment_id]
    except KeyError as exc:
        supported = ", ".join(str(key) for key in sorted(EXPERIMENT_PLANS))
        raise KeyError(f"Unknown experiment id {experiment_id}. Supported: {supported}") from exc

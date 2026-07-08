# Experiment Results

Summary of selected offline runs. Full data is in `results/experiment_NNN/`.

## Task 1

**Experiment 0** (zero pixel noise):

| Run | Filter | Position RMSE [m] |
|-----|--------|-------------------|
| experiment_001 | KF | 0.0014 |
| experiment_002 | EKF | ≈ 0 |
| experiment_003 | UKF | 0.00004 |

**Experiment 1** (1 px noise, 3 cameras):

| Run | Filter | Position RMSE [m] |
|-----|--------|-------------------|
| experiment_005 | KF | 0.0085 |
| experiment_006 | EKF | 0.0087 |
| experiment_007 | UKF | 0.0112 |

## Advanced task

| Experiment | Description |
|------------|-------------|
| 101 | Calibration manoeuvre comparison (hover, straight line, planar/3D figure-eight) |
| 103 | Calibration with 0.5 / 1.0 / 2.0 px noise |
| 104 | Tracking with calibrated vs. true camera poses |

Calibration datasets: `results/calibration_*`

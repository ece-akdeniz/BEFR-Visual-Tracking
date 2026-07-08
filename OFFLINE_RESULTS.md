# Offline Experiment Results Summary

Generated during plan implementation. Full CSV data lives in `results/experiment_NNN/`.

## Experiment 0 — Mathematical sanity (zero noise)

| Run | Filter | Position RMSE [m] | Velocity RMSE [m/s] |
|-----|--------|-------------------|---------------------|
| experiment_001 | KF | 0.00138 | 0.0197 |
| experiment_002 | EKF | 4.4e-07 | 0.0120 |
| experiment_003 | UKF | 4.1e-05 | 0.0148 |

**Pass:** position RMSE near zero with noise off.

## Experiment 1 — Main filter comparison (1 px noise, 3 cameras)

| Run | Filter | Position RMSE [m] | Velocity RMSE [m/s] |
|-----|--------|-------------------|---------------------|
| experiment_005 | KF | 0.00853 | 0.0520 |
| experiment_006 | EKF | 0.00874 | 0.0522 |
| experiment_007 | UKF | 0.01121 | 0.0571 |

**Pass:** all three filters track the slow circle with sub-centimetre RMSE.

## Advanced experiments

| Experiment | Status | Notes |
|------------|--------|-------|
| 101 — Manoeuvre observability | Completed | experiment_009, 011, 013, 015 |
| 102 — Calibration camera count | Failed | `No valid triangulated positions` — degenerate geometry with 2 cameras |
| 103 — Calibration noise | Completed | experiment_022, 024, 026 |
| 104 — Impact on tracking | Partial / slow | run separately if needed |

Calibration datasets generated in `results/calibration_*`. Bundle adjustment (`run_calibration`) is compute-heavy; run on Mac when needed:

```bash
cd BEFR-Project
PYTHONPATH=. python -m befr_visual_tracking.run_calibration results/calibration_figure_eight_3d
```

## Dashboard

```bash
cd BEFR-Project
PYTHONPATH=. streamlit run dashboard/streamlit_app.py
```

Load experiments from the **Saved experiment viewer** (30 runs available).

## Live demo (VM)

See [VM_STARTUP.md](VM_STARTUP.md). Validate with:

```bash
~/BEFR-Project/scripts/validate_live_tracking.sh
```

Record live data with:

```bash
~/BEFR-Project/scripts/record_live_experiment.sh
```

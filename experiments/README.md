Experiments Convention

- Root: `experiments/`
- Each experiment: a folder named `YYYYMMDD_HHMMSS_desc/`
- Contents:
  - `params.json`: Input parameters used (e.g., last_n, hfa, weights, calibration a/b, blend, flags)
  - `results.csv`: Key metrics summary (accuracy, MAE, coverage metrics)
  - Optional: `notes.md`, plots, and links to backtest artifacts under `backtests/`

Lightweight Workflow

- After running calibration or backtests, create a timestamped folder and dump params/results.
- Keep experiments small and committed for reproducibility.
- Reference experiment IDs in PR descriptions when relevant.


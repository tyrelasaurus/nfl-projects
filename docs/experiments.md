Experiments Guide

Overview
- The `experiments/` directory holds lightweight, reproducible experiment artifacts committed to the repo.

Layout
- `experiments/`
  - `YYYYMMDD_HHMMSS_description/`
    - `params.json` — parameters used (last_n, hfa, weights, calibration a/b, blend, flags)
    - `results.csv` — summary metrics (e.g., accuracy, MAE)
    - `notes.md` (optional) — brief observations and next steps
    - links or references to `backtests/` artifacts (per-game CSVs, HTML summaries)

Workflow
- After running a calibration or backtest sweep, create a new folder with a timestamp and short description.
- Save the exact parameters and a concise CSV of results for traceability.
- Reference the experiment folder in PR descriptions or commit messages when tuning parameters.


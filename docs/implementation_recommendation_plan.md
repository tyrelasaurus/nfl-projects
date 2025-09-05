## Implementation Recommendation Plan

- Objective: Improve winner and spread prediction accuracy through calibrated power scores, tuned home-field advantage, and robust validation, while keeping the system simple and transparent.
- Outcomes: Calibrated projections, reproducible backtests, clear dashboards, and a controlled release process with measurable accuracy and MAE gains.

## Tech Stack

- Core: Python 3.12, pandas, numpy
- Modeling/Calibration: scikit-learn (linear regression, isotonic regression), Optuna (or grid search) for lightweight hyperparameter sweeps
- Visualization: Plotly or Altair (static HTML export), optional csv → HTML tables with embedded JS filters (already in place)
- CLI/Automation: Native Python CLIs, Makefiles or simple shell scripts for batch runs
- Validation/Testing: pytest, pytest-cov
- CI/CD: GitHub Actions (lint, tests, quick backtest smoke)
- Experiment Tracking (optional): MLflow or lightweight CSV/JSON artifacts (commit to repo) for parameters/results
- Config: YAML files (parameter versioning: last_n, HFA, calibration a/b, weights)

## Architecture Overview

- Data Layer: ESPNClient (existing) with season/weekly fetchers. Backtests use season events for historical stability.
- Model Layer: PowerRankModel (existing) with last-N across seasons + SOS + rolling stats; add margin scaling and HFA tuning hooks.
- Calibration Layer: Scripts that read per-game backtest CSVs and compute:
  - Linear margin scaling: actual ≈ a + b × projected
  - Optimal HFA: sweep to minimize MAE
  - Probability calibration: map projected margin → P(win) via isotonic
- Evaluation Layer: Backtests (winners and spreads) over multiple seasons; dashboards (HTML) with filters and summary tables.
- Orchestration: Runner scripts for daily/weekly runs and parameterized backtests.

## Implementation Phases

### 1) Data & Backtest Foundation (stabilize and extend)
- Enhance backtest outputs (done): actual margin, final score, coverage vs predicted spread, HTML filters (week/team/position/predicted side/coverage).
- Add multi-season runs (2021–2024), script to aggregate results across seasons.
- Deliverables:
  - backtest_winners for seasons list
  - Folded summary CSV (season-level rows) + consolidated HTML index

### 2) Calibration v1: Margin Scaling & HFA Tuning
- Implement calibrate_margins.py
  - Input: per-game winners backtest CSVs
  - Output: a, b for linear scaling; residual diagnostics
- Implement tune_hfa.py
  - Input: per-game winners backtest CSVs
  - Output: optimal HFA (global; optional per-stadium/team-over-time later)
- Integrate calibrated parameters into runtime:
  - New config fields: calibration.margin.a, calibration.margin.b, model.hfa
  - Apply y’ = a + b × projected before display/decisions
- Acceptance:
  - MAE reduction vs baseline; stable across seasons (no overfit)

### 3) Hyperparameter Tuning: last-N & Weights
- Implement sweep_params.py
  - Sweep last_n ∈ {8, 10, 12, 14, 17}
  - Optional grid: weights for season_avg_margin, rolling_avg_margin, sos, recency_factor
  - Objective: composite (e.g., 70% winners accuracy + 30% MAE reduction)
- Produce ranked summary; write chosen params to config.
- Acceptance:
  - Improvement sustained across multiple seasons; minimal complexity added

### 4) Residual Analytics & Dashboards
- Build residual_analytics.py:
  - Segment residuals by: home/away, spread magnitude buckets, week number, team, stadium, weather proxies (if available later)
  - Output: HTML dashboards with tables/plots (Plotly/Altair)
- Identify and document systematic biases and suggested remedies (e.g., team-specific or early-season adjustments)
- Acceptance:
  - Clear insights + action items; segments with significant bias (p<0.05 or effect size threshold)

### 5) Probability Calibration (Winners)
- Fit isotonic regression mapping |projected margin| → P(home win)
- Validate with Brier score/log loss; produce calibration plots (reliability curves)
- Integrate: produce calibrated win probabilities with projections
- Acceptance:
  - Better probability calibration (lower Brier score), monotonic mapping

### 6) Edge Policy & Thresholds (ATS workflow ready)
- When odds available:
  - Edge bin analysis: edge bins → realized cover rates; pick thresholds >52.4%
  - Add an edge_threshold setting; runner outputs only actionable edges
- When odds unavailable:
  - Use internal threshold vs predicted spread magnitude as a proxy for confidence
- Acceptance:
  - Documented thresholds with historical performance; toggle in runner

### 7) Validation & Release
- Multi-season CV summary; hold-out season confirmation
- Update README with calibrated parameters and results
- CI smoke test: short backtest slice, schema and HTML generation checks
- Feature flag: config.enable_calibration to toggle on/off
- Acceptance:
  - Documented improvements, revert plan if needed, deployment notes

## Deliverables

- Scripts:
  - calibrate_margins.py, tune_hfa.py, sweep_params.py, residual_analytics.py
- Config changes:
  - calibration.margin.{a,b}, model.hfa, model.last_n, model.weights
- Reports:
  - Multi-season winners/spreads backtests
  - Calibration report (scaling/HFA)
  - Residual analytics HTML
- Updated runner and README with new parameters and usage

## Development Workflow

- Branching:
  - feature/calibration, feature/hfa-tuning, feature/sweeps, feature/residual-dashboard
- PR process:
  - Small PRs; include short backtest artifact (1 season subset) for reviewers
  - Code owners for ESPN client and model files
- CI:
  - pytest + coverage
  - Lint (flake8 optional)
  - Quick backtest smoke (1 season, 1–2 weeks) to catch schema regressions
- Experiment Tracking:
  - Store CSV/JSON artifacts per experiment in /experiments with metadata:
    - params.json (last_n, hfa, weights, a/b)
    - results.csv (summary)
  - Keep a CHANGELOG.md section for parameter updates
- Parameter Versioning:
  - All tuned params live in YAML with version tag
  - Runner logs parameter set hash in outputs
- Reproducibility:
  - Deterministic seeds where applicable (not much stochasticity here)
  - Pin tool versions in requirements
- Feature Flags:
  - config.enable_calibration, config.use_calibrated_probability
- Environments:
  - Dev (local), CI (GitHub Actions), Prod (scheduled run)

## Timeline (indicative)

- Week 1:
  - Finish calibration v1 (margin scaling, HFA), integrate into runner/config
  - Multi-season winners backtests for baseline + calibrated
- Week 2:
  - last-N and weight sweeps + select parameters
  - Residual analytics report
- Week 3:
  - Probability calibration + calibration plots + integration
  - ATS edge policy (if odds coverage sufficient)
- Week 4:
  - Polish dashboards, documentation, CI guardrails
  - Release calibrated configuration as default

## Risks & Mitigations

- ESPN data gaps (odds/historical inconsistencies):
  - Use winners backtest as primary metric; treat spreads backtest as supplemental
- Overfitting:
  - Multi-season validation; hold-out year; limit model changes to simple scaling + HFA, plus conservative weight tuning
- Complexity creep:
  - Keep scripting simple; avoid heavy frameworks; prefer CSV + HTML artifacts

## Success Metrics

- Winners accuracy: +1–3 pp over baseline across multiple seasons
- MAE vs actual margin: measurable reduction after calibration
- Calibrated probability: improved Brier/log loss
- Edge thresholds: historical cover > 52.4% for selected bins (spreads backtest)


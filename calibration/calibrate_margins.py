#!/usr/bin/env python3
"""
Fit linear margin scaling: actual_margin ≈ a + b * projected_margin.

Inputs: one or more backtest winners CSV files with columns:
  projected_margin, actual_margin

Outputs:
  - Prints a and b
  - Optionally writes to calibration/params.yaml (merge/update)

Usage:
  python -m calibration.calibrate_margins --inputs backtests/backtest_winners_2024_*.csv --write
"""

import argparse
import glob
import os
import sys
import yaml
import pandas as pd
from typing import List

VALID_LEAGUES = ('nfl', 'ncaa')


def load_csvs(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        for fp in glob.glob(p):
            try:
                df = pd.read_csv(fp)
                if 'projected_margin' in df.columns and 'actual_margin' in df.columns:
                    frames.append(df[['projected_margin', 'actual_margin']].copy())
            except Exception:
                continue
    if not frames:
        raise RuntimeError("No valid CSVs with projected_margin and actual_margin found")
    return pd.concat(frames, ignore_index=True).dropna()


def fit_linear(df: pd.DataFrame):
    import numpy as np
    X = df['projected_margin'].values
    y = df['actual_margin'].values
    # Fit a + b x via least squares
    A = np.vstack([np.ones_like(X), X]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef[0], coef[1]
    mae = float(np.mean(np.abs((a + b * X) - y)))
    return float(a), float(b), mae


def resolve_params_path(league: str | None, override: str | None) -> str:
    if override:
        return override
    lg = (league or 'nfl').lower()
    if lg not in VALID_LEAGUES:
        raise ValueError(f"Unsupported league '{league}'. Choose from {VALID_LEAGUES}.")
    return 'calibration/params.yaml' if lg == 'nfl' else f'calibration/{lg}_params.yaml'


def update_params_yaml(a: float, b: float, path: str):
    data = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
    data.setdefault('calibration', {}).setdefault('margin', {})
    data['calibration']['margin']['a'] = float(a)
    data['calibration']['margin']['b'] = float(b)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description='Calibrate margin scaling (actual ≈ a + b * projected)')
    ap.add_argument('--inputs', nargs='+', required=True, help='Glob patterns for backtest winners CSVs')
    ap.add_argument('--league', choices=VALID_LEAGUES, help='League for params convenience (default: nfl)')
    ap.add_argument('--params-path', help='Explicit params YAML path (overrides --league)')
    ap.add_argument('--write', action='store_true', help='Write a,b to the resolved params YAML')
    args = ap.parse_args()

    df = load_csvs(args.inputs)
    a, b, mae = fit_linear(df)
    print(f"Fitted margin scaling: actual ≈ {a:.3f} + {b:.3f} * projected (MAE={mae:.3f})")
    if args.write:
        params_path = resolve_params_path(args.league, args.params_path)
        update_params_yaml(a, b, params_path)
        print(f"Updated {params_path}")


if __name__ == '__main__':
    sys.exit(main())

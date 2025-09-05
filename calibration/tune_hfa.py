#!/usr/bin/env python3
"""
Tune Home Field Advantage (HFA) by minimizing MAE against actual margins.

Inputs: one or more winners backtest CSVs with columns:
  projected_margin, actual_margin
These projected margins were computed with an assumed HFA (default 2.0).

We recover base_diff = projected_margin - hfa_used, then evaluate candidate HFA values
to minimize MAE of (base_diff + hfa_candidate) vs actual_margin.

Usage:
  python -m calibration.tune_hfa --inputs backtests/backtest_winners_*.csv --hfa-used 2.0 --min 1.0 --max 3.5 --step 0.1 --write
"""

import argparse
import glob
import os
import sys
import yaml
import pandas as pd
import numpy as np
from typing import List


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


def sweep_hfa(df: pd.DataFrame, hfa_used: float, hfa_min: float, hfa_max: float, step: float):
    base = df['projected_margin'].values - hfa_used
    y = df['actual_margin'].values
    best_hfa = None
    best_mae = float('inf')
    cur = hfa_min
    while cur <= hfa_max + 1e-9:
        pred = base + cur
        mae = float(np.mean(np.abs(pred - y)))
        if mae < best_mae:
            best_mae, best_hfa = mae, float(cur)
        cur += step
    return best_hfa, best_mae


def update_params_yaml(hfa: float, path: str = 'calibration/params.yaml'):
    data = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
    data.setdefault('model', {})
    data['model']['hfa'] = float(hfa)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description='Tune HFA to minimize MAE vs actual margins')
    ap.add_argument('--inputs', nargs='+', required=True, help='Glob patterns for backtest winners CSVs')
    ap.add_argument('--hfa-used', type=float, default=2.0, help='HFA used to compute projected_margin in inputs')
    ap.add_argument('--min', type=float, default=1.0, help='Min HFA to test')
    ap.add_argument('--max', type=float, default=3.5, help='Max HFA to test')
    ap.add_argument('--step', type=float, default=0.1, help='Step for HFA sweep')
    ap.add_argument('--write', action='store_true', help='Write tuned HFA to calibration/params.yaml')
    args = ap.parse_args()

    df = load_csvs(args.inputs)
    best_hfa, best_mae = sweep_hfa(df, args.hfa_used, args.min, args.max, args.step)
    print(f"Best HFA: {best_hfa:.2f} (MAE={best_mae:.3f})")
    if args.write:
        update_params_yaml(best_hfa)
        print("Updated calibration/params.yaml")


if __name__ == '__main__':
    sys.exit(main())


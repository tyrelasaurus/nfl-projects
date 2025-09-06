#!/usr/bin/env python3
"""
Calibrate winner probabilities from projected margins using isotonic-like mapping.

Reads winners backtest per-game CSVs and fits a monotonic mapping from
projected margin (calibrated) to P(home win). Stores the mapping in
calibration/params.yaml as breakpoints (x) and probabilities (p), which can
be used at runtime to produce calibrated probabilities.

Usage:
  python -m calibration.calibrate_probabilities --inputs backtests/backtest_winners_*.csv --write
"""

import argparse
import glob
import os
import sys
import yaml
import numpy as np
import pandas as pd
from typing import List, Tuple


def load_margin_params(path='calibration/params.yaml') -> Tuple[float, float, float, float]:
    a, b, lo, hi = 0.0, 1.0, 3.0, 7.0
    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
            a = float(cfg.get('calibration', {}).get('margin', {}).get('a', 0.0))
            b = float(cfg.get('calibration', {}).get('margin', {}).get('b', 1.0))
            bl = cfg.get('calibration', {}).get('blend', {})
            lo = float(bl.get('low', 3.0))
            hi = float(bl.get('high', 7.0))
    except Exception:
        pass
    return a, b, lo, hi


def blended_calibration(raw: np.ndarray, a: float, b: float, lo: float, hi: float) -> np.ndarray:
    cal_lin = a + b * raw
    mag = np.abs(raw)
    out = np.empty_like(raw)
    # Regions
    mask_lo = mag <= lo
    mask_hi = mag >= hi
    mask_mid = ~(mask_lo | mask_hi)
    out[mask_lo] = raw[mask_lo]
    out[mask_hi] = cal_lin[mask_hi]
    t = (mag[mask_mid] - lo) / (hi - lo)
    out[mask_mid] = (1 - t) * raw[mask_mid] + t * cal_lin[mask_mid]
    return out


def load_backtests(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        for fp in glob.glob(p):
            try:
                df = pd.read_csv(fp)
                if {'season','week','projected_margin','actual_margin'}.issubset(df.columns):
                    frames.append(df[['season','week','projected_margin','actual_margin']].copy())
                elif {'season','week','projected_margin_raw','projected_margin_cal','actual_margin'}.issubset(df.columns):
                    frames.append(df[['season','week','projected_margin_raw','projected_margin_cal','actual_margin']].copy())
            except Exception:
                continue
    if not frames:
        raise SystemExit('No matching winners per-game CSVs found')
    return pd.concat(frames, ignore_index=True)


def get_proj_cal(df: pd.DataFrame, a: float, b: float, lo: float, hi: float) -> np.ndarray:
    if 'projected_margin_cal' in df.columns:
        return pd.to_numeric(df['projected_margin_cal'], errors='coerce').values
    raw = pd.to_numeric(df.get('projected_margin_raw', df.get('projected_margin')), errors='coerce').values
    return blended_calibration(raw, a, b, lo, hi)


def pav_isotonic(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Pool-Adjacent-Violators for monotone increasing fit y_hat(x)
    # Start with bin-level averages and pool when violation occurs.
    # Returns unique x (bin centers) and fitted p for interpolation.
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    w = w[order]
    # Initialize blocks
    p = y.copy()
    wsum = w.copy()
    i = 0
    while i < len(p) - 1:
        if p[i] <= p[i+1] + 1e-12:
            i += 1
        else:
            # pool i and i+1
            new_p = (p[i]*wsum[i] + p[i+1]*wsum[i+1]) / (wsum[i]+wsum[i+1])
            p[i] = new_p
            wsum[i] = wsum[i] + wsum[i+1]
            # remove i+1 by folding left
            p = np.delete(p, i+1)
            wsum = np.delete(wsum, i+1)
            x = np.delete(x, i+1)
            if i > 0:
                i -= 1
    return x, p


def fit_probability(df: pd.DataFrame, a: float, b: float, lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    proj = get_proj_cal(df, a, b, lo, hi)
    actual = pd.to_numeric(df['actual_margin'], errors='coerce').values
    # y: 1 if home win, 0 otherwise (exclude pushes from fit)
    mask = ~np.isnan(proj) & ~np.isnan(actual) & (actual != 0)
    proj = proj[mask]
    y = (actual[mask] > 0).astype(float)
    # Bin into quantiles to reduce noise
    q = np.linspace(0, 1, 51)
    qs = np.quantile(proj, q)
    # Deduplicate quantiles
    qs = np.unique(qs)
    bins = np.digitize(proj, qs[1:-1], right=False)
    bin_x = []
    bin_y = []
    bin_w = []
    for bidx in range(len(qs)):
        sel = bins == bidx if bidx < len(qs)-1 else (proj >= qs[-1])
        if np.any(sel):
            bin_x.append(np.mean(proj[sel]))
            bin_y.append(np.mean(y[sel]))
            bin_w.append(np.sum(sel))
    bx = np.array(bin_x)
    by = np.array(bin_y)
    bw = np.array(bin_w, dtype=float)
    # Apply PAV isotonic
    xi, pi = pav_isotonic(bx, by, bw)
    return xi, pi


def write_params(x: np.ndarray, p: np.ndarray, path='calibration/params.yaml'):
    data = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
    data.setdefault('calibration', {})['probability'] = {
        'type': 'isotonic',
        'x': [float(v) for v in x.tolist()],
        'p': [float(v) for v in p.tolist()],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description='Calibrate winner probabilities from projected margins')
    ap.add_argument('--inputs', nargs='+', required=True, help='Glob patterns for winners per-game CSVs')
    ap.add_argument('--write', action='store_true', help='Write mapping to calibration/params.yaml')
    args = ap.parse_args()

    a, b, lo, hi = load_margin_params()
    df = load_backtests(args.inputs)
    x, p = fit_probability(df, a, b, lo, hi)
    print('Fitted probability mapping:')
    print('X (breakpoints):', ','.join(f"{v:.2f}" for v in x))
    print('P (probabilities):', ','.join(f"{v:.3f}" for v in p))
    if args.write:
        write_params(x, p)
        print('Updated calibration/params.yaml with probability mapping')


if __name__ == '__main__':
    sys.exit(main())


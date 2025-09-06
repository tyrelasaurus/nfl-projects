#!/usr/bin/env python3
"""
Edge Policy Analysis

Reads spreads backtest per-game CSVs (from backtest/backtest_spreads.py) and
analyzes ATS win rate as a function of absolute edge magnitude. Recommends a
threshold that achieves >= 52.4% ATS win rate with a minimum sample size.

Inputs must include columns: 'edge' and 'correct' (1/0), optionally 'season'.

Usage:
  python -m calibration.edge_policy --inputs backtests/backtest_*_*.csv --min-samples 100 --target 0.524 --write

Outputs:
  - Prints a table of thresholds → n, win%.
  - If --write, updates calibration/params.yaml with policy.edge.threshold and enables it.
  - Writes CSV summary to calibration/edge_policy_summary.csv
"""

import argparse
import glob
import os
import csv
from typing import List, Tuple
import yaml


def load_rows(patterns: List[str]) -> List[dict]:
    rows: List[dict] = []
    for p in patterns:
        for fp in glob.glob(p):
            try:
                with open(fp, 'r') as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        if 'edge' in r and 'correct' in r:
                            rows.append(r)
            except Exception:
                continue
    return rows


def analyze_thresholds(rows: List[dict], step: float = 0.5, max_thr: float = 10.0) -> List[Tuple[float, int, float]]:
    # Returns list of (threshold, n, win_rate) for abs(edge)>=threshold
    def parse_float(x):
        try:
            return float(x)
        except Exception:
            return None
    edges = []
    wins = []
    for r in rows:
        e = parse_float(r.get('edge', ''))
        c = r.get('correct', '')
        if e is None or c == '':
            continue
        try:
            w = 1 if float(c) > 0 else 0
        except Exception:
            continue
        edges.append(abs(e))
        wins.append(w)
    out: List[Tuple[float, int, float]] = []
    for thr in [round(x * step, 3) for x in range(0, int(max_thr / step) + 1)]:
        sel = [i for i, val in enumerate(edges) if val >= thr]
        n = len(sel)
        if n == 0:
            wr = 0.0
        else:
            wr = sum(wins[i] for i in sel) / n
        out.append((thr, n, wr))
    return out


def write_summary(rows: List[Tuple[float, int, float]], path: str) -> str:
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['threshold', 'n', 'win_rate'])
        for thr, n, wr in rows:
            w.writerow([thr, n, f"{wr:.3f}"])
    return path


def update_params_yaml(threshold: float, path: str = 'calibration/params.yaml', enable: bool = True, min_samples: int | None = None):
    data = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
    cal = data.setdefault('calibration', {})
    policy = cal.setdefault('policy', {})
    edge = policy.setdefault('edge', {})
    edge['threshold'] = float(threshold)
    edge['enabled'] = bool(enable)
    if min_samples is not None:
        edge['min_samples'] = int(min_samples)
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description='Analyze ATS edge thresholds and optionally write policy to params')
    ap.add_argument('--inputs', nargs='+', required=True, help='Glob patterns for spreads per-game CSVs')
    ap.add_argument('--min-samples', type=int, default=100, help='Minimum sample size for threshold')
    ap.add_argument('--target', type=float, default=0.524, help='Target ATS win rate (e.g., 0.524 ~ break-even)')
    ap.add_argument('--write', action='store_true', help='Write recommended threshold to calibration/params.yaml')
    args = ap.parse_args()

    rows = load_rows(args.inputs)
    if not rows:
        raise SystemExit('No rows with edge and correct fields found')
    table = analyze_thresholds(rows)

    # Choose smallest threshold meeting target and min-samples to maximize opportunities
    candidates = [(thr, n, wr) for thr, n, wr in table if n >= args.min_samples and wr >= args.target]
    if candidates:
        recommended = min(candidates, key=lambda x: x[0])
    else:
        # fallback to best win rate with min samples
        pool = [t for t in table if t[1] >= args.min_samples]
        recommended = max(pool, key=lambda x: x[2]) if pool else max(table, key=lambda x: x[2])

    # Print summary
    print('Threshold Analysis (abs(edge) >= thr):')
    print('thr\tn\twin%')
    for thr, n, wr in table:
        print(f"{thr:.1f}\t{n}\t{wr:.3f}")
    print('\nRecommended: thr=%.2f, n=%d, win%%=%.3f' % recommended)

    # Write CSV summary
    out = write_summary(table, 'calibration/edge_policy_summary.csv')
    print(f'Wrote {out}')

    if args.write:
        update_params_yaml(recommended[0], enable=True, min_samples=args.min_samples)
        print('Updated calibration/params.yaml with policy.edge.threshold and enabled=true')


if __name__ == '__main__':
    raise SystemExit(main())


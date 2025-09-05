#!/usr/bin/env python3
"""
Pre vs Post Calibration Report for winner/margin metrics.

Reads one or more backtest_winners CSVs (with projected_margin, actual_margin)
and compares metrics using raw projected margins vs calibrated margins
(a + b * projected), using a,b from calibration/params.yaml.

Outputs an HTML summary with per-season and overall stats.

Usage:
  python -m calibration.report_pre_post --inputs backtests/backtest_winners_*.csv --output ./backtests
"""

import argparse
import glob
import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timezone


def load_params(path='calibration/params.yaml'):
    a, b = 0.0, 1.0
    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
            a = float(cfg.get('calibration', {}).get('margin', {}).get('a', 0.0))
            b = float(cfg.get('calibration', {}).get('margin', {}).get('b', 1.0))
    except Exception:
        pass
    return a, b


def load_csvs(patterns):
    frames = []
    for p in patterns:
        for fp in glob.glob(p):
            try:
                df = pd.read_csv(fp)
                if {'season','week','projected_margin','actual_margin'}.issubset(df.columns):
                    frames.append(df[['season','week','projected_margin','actual_margin']].copy())
            except Exception:
                continue
    if not frames:
        raise SystemExit('No matching CSVs with required columns found')
    return pd.concat(frames, ignore_index=True)


def compute_metrics(df_raw: pd.DataFrame, a: float, b: float):
    df = df_raw.copy()
    df['projected_cal'] = a + b * df['projected_margin']
    # Accuracy (winners) pre/post
    pred_pre = np.sign(df['projected_margin'])
    pred_post = np.sign(df['projected_cal'])
    actual = np.sign(df['actual_margin'])
    mask_valid_pre = actual != 0
    mask_valid_post = actual != 0
    acc_pre = float((pred_pre[mask_valid_pre] == actual[mask_valid_pre]).mean()) if mask_valid_pre.any() else 0.0
    acc_post = float((pred_post[mask_valid_post] == actual[mask_valid_post]).mean()) if mask_valid_post.any() else 0.0
    # MAE pre/post
    mae_pre = float(np.abs(df['projected_margin'] - df['actual_margin']).mean())
    mae_post = float(np.abs(df['projected_cal'] - df['actual_margin']).mean())
    return acc_pre, acc_post, mae_pre, mae_post


def main():
    ap = argparse.ArgumentParser(description='Report pre vs post calibration metrics')
    ap.add_argument('--inputs', nargs='+', required=True, help='Glob patterns for backtest_winners CSVs')
    ap.add_argument('--output', default='./backtests', help='Output directory')
    args = ap.parse_args()

    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    a, b = load_params()
    df = load_csvs(args.inputs)

    rows = []
    for season, sdf in df.groupby('season'):
        acc_pre, acc_post, mae_pre, mae_post = compute_metrics(sdf, a, b)
        rows.append({'season': int(season), 'acc_pre': acc_pre, 'acc_post': acc_post, 'mae_pre': mae_pre, 'mae_post': mae_post})
    # Overall
    acc_pre, acc_post, mae_pre, mae_post = compute_metrics(df, a, b)
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out_html = os.path.join(out_dir, f'pre_post_calibration_{ts}.html')
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Pre vs Post Calibration</title>"
                "<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3}</style>"
                "</head><body>")
        f.write("<h1>Pre vs Post Calibration Report</h1>")
        f.write(f"<p>Using a = {a:.3f}, b = {b:.3f}</p>")
        f.write("<table><tr><th>Season</th><th>Accuracy (pre)</th><th>Accuracy (post)</th><th>MAE (pre)</th><th>MAE (post)</th></tr>")
        for r in sorted(rows, key=lambda x: x['season']):
            f.write(f"<tr><td>{r['season']}</td><td>{r['acc_pre']:.3f}</td><td>{r['acc_post']:.3f}</td><td>{r['mae_pre']:.2f}</td><td>{r['mae_post']:.2f}</td></tr>")
        f.write(f"<tr><th>Overall</th><th>{acc_pre:.3f}</th><th>{acc_post:.3f}</th><th>{mae_pre:.2f}</th><th>{mae_post:.2f}</th></tr>")
        f.write("</table></body></html>")
    print("Report written:", out_html)


if __name__ == '__main__':
    sys.exit(main())


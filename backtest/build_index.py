#!/usr/bin/env python3
"""
Build a simple consolidated index HTML in the backtests directory, linking to
per-season winners summaries and recent backtest summary pages.

Usage:
  python -m backtest.build_index --dir ./backtests
"""

import argparse
import glob
import os
from datetime import datetime


def build_index(backtests_dir: str) -> str:
    d = os.path.abspath(os.path.expanduser(backtests_dir))
    os.makedirs(d, exist_ok=True)
    winners_pages = sorted(glob.glob(os.path.join(d, 'backtest_winners_summary_*_*.html')))
    summary_pages = sorted(glob.glob(os.path.join(d, 'backtest_summary_*_*.html')))
    seasons_csv = os.path.join(d, 'winners_seasons_summary.csv')

    def rel(p: str) -> str:
        return os.path.basename(p)

    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Backtests Index</title>",
        "<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3}</style>",
        "</head><body>",
        f"<h1>Backtests Index</h1><p>Generated: {now}</p>",
    ]

    html.append("<h2>Winners (Per-Season) Summaries</h2>")
    if winners_pages:
        html.append("<table><tr><th>Season</th><th>File</th></tr>")
        for p in winners_pages:
            base = rel(p)
            # Expect file like backtest_winners_summary_YYYY_TS.html
            season = base.split('_')[3] if len(base.split('_')) >= 4 else ''
            html.append(f"<tr><td>{season}</td><td><a href='{base}'>{base}</a></td></tr>")
        html.append("</table>")
    else:
        html.append("<p>No winners summaries found.</p>")

    if os.path.exists(seasons_csv):
        html.append("<h2>Folded Season Summary</h2>")
        html.append(f"<p><a href='{os.path.basename(seasons_csv)}'>winners_seasons_summary.csv</a></p>")
        # Render a small table preview
        try:
            import csv as _csv
            rows = []
            with open(seasons_csv, 'r') as _f:
                r = _csv.DictReader(_f)
                for i, row in enumerate(r):
                    if i < 10:
                        rows.append(row)
            html.append("<table><tr><th>Season</th><th>Games</th><th>Wins</th><th>Losses</th><th>Pushes</th><th>Accuracy</th><th>Cover Rate</th></tr>")
            for row in rows:
                html.append(
                    f"<tr><td>{row['season']}</td><td>{row['games']}</td><td>{row['wins']}</td><td>{row['losses']}</td><td>{row['pushes']}</td><td>{float(row['accuracy']):.3f}</td><td>{(float(row['cover_rate']):.3f) if row['cover_rate'] not in ('', None) else ''}</td></tr>"
                )
            html.append("</table>")
        except Exception:
            pass

    html.append("<h2>Recent Backtest Summaries</h2>")
    if summary_pages:
        html.append("<ul>")
        for p in summary_pages[-20:]:  # last 20
            base = rel(p)
            html.append(f"<li><a href='{base}'>{base}</a></li>")
        html.append("</ul>")
    else:
        html.append("<p>No backtest summaries found.</p>")

    html.append("</body></html>")
    out_path = os.path.join(d, 'index.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html))
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Build consolidated backtests index page')
    ap.add_argument('--dir', default='./backtests', help='Backtests directory')
    args = ap.parse_args()
    out = build_index(args.dir)
    print(f"Wrote {out}")


if __name__ == '__main__':
    raise SystemExit(main())

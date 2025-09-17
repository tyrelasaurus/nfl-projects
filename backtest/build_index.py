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
    season_summaries = sorted(glob.glob(os.path.join(d, 'winners_seasons_summary*.csv')))

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
        html.append("<table><tr><th>League</th><th>Season</th><th>File</th></tr>")
        for p in winners_pages:
            base = rel(p)
            parts = base.split('_')
            league = 'nfl'
            season = ''
            if len(parts) >= 5 and parts[0] == 'backtest' and parts[1] == 'winners' and parts[2] == 'summary':
                maybe_league = parts[3]
                if maybe_league.isdigit():
                    season = maybe_league
                else:
                    league = maybe_league
                    if len(parts) >= 6:
                        season = parts[4]
            html.append(f"<tr><td>{league.upper()}</td><td>{season}</td><td><a href='{base}'>{base}</a></td></tr>")
        html.append("</table>")
    else:
        html.append("<p>No winners summaries found.</p>")

    if season_summaries:
        html.append("<h2>Folded Season Summaries</h2>")
        for seasons_csv in season_summaries:
            base_csv = os.path.basename(seasons_csv)
            html.append(f"<h3>{base_csv}</h3>")
            html.append(f"<p><a href='{base_csv}'>{base_csv}</a></p>")
            try:
                import csv as _csv
                rows = []
                with open(seasons_csv, 'r') as _f:
                    r = _csv.DictReader(_f)
                    for i, row in enumerate(r):
                        if i < 10:
                            rows.append(row)
                headers = ['league', 'season', 'games', 'wins', 'losses', 'pushes', 'accuracy', 'cover_rate']
                html.append("<table><tr>" + ''.join(f"<th>{h.title()}</th>" for h in headers) + "</tr>")
                for row in rows:
                    cover_rate_val = ''
                    try:
                        cv = row.get('cover_rate')
                        if cv not in ('', None):
                            cover_rate_val = f"{float(cv):.3f}"
                    except Exception:
                        cover_rate_val = row.get('cover_rate', '')
                    html.append(
                        "<tr>"
                        f"<td>{row.get('league', 'N/A')}</td>"
                        f"<td>{row.get('season', '')}</td>"
                        f"<td>{row.get('games', '')}</td>"
                        f"<td>{row.get('wins', '')}</td>"
                        f"<td>{row.get('losses', '')}</td>"
                        f"<td>{row.get('pushes', '')}</td>"
                        f"<td>{float(row.get('accuracy', 0.0)):.3f}</td>"
                        f"<td>{cover_rate_val}</td>"
                        "</tr>"
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

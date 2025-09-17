#!/usr/bin/env python3
"""
Aggregate per-season winners backtests into a folded CSV.

Scans backtests/backtest_winners_{season}_{timestamp}.csv, picks the latest
per season, computes season-level metrics, and writes
backtests/winners_seasons_summary.csv.

Metrics:
- season, games, pushes, wins, losses, accuracy (wins/(games-pushes))
- cover_yes, cover_no, cover_push, cover_rate (if 'covered' present)

Usage:
  python -m backtest.aggregate_seasons --dir ./backtests
"""

import argparse
import csv
import glob
import os

from typing import List

try:  # Allow running as package or standalone script
    from .backtest_winners import VALID_LEAGUES, parse_league  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - standalone execution
    from backtest_winners import VALID_LEAGUES, parse_league  # type: ignore[import-not-found]


def latest_by_season(backtests_dir: str, league: str) -> dict[int, str]:
    patterns = [os.path.join(backtests_dir, f'backtest_winners_{league}_*_*.csv')]
    # Backward compatibility for historical NFL files without explicit league
    if league == 'nfl':
        patterns.append(os.path.join(backtests_dir, 'backtest_winners_*_*.csv'))

    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))

    latest: dict[int, str] = {}
    for p in files:
        base = os.path.basename(p)
        parts = base.split('_')
        if len(parts) < 4 or parts[0] != 'backtest' or parts[1] != 'winners':
            continue

        # Determine league/season segments
        league_part = parts[2]
        season_idx = 3
        if league_part not in VALID_LEAGUES:
            # Legacy format: backtest_winners_{season}_{ts}.csv
            league_part = 'nfl'
            season_idx = 2

        if league_part != league:
            continue

        if len(parts) <= season_idx:
            continue

        season_part = parts[season_idx]
        # Remove extension if attached to timestamp
        season_str = season_part
        try:
            season = int(season_str)
        except ValueError:
            continue

        latest[season] = p
    return latest


def compute_metrics(csv_path: str) -> dict:
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        games = 0
        pushes = 0
        wins = 0
        cover_yes = 0
        cover_no = 0
        cover_push = 0
        has_cover = False
        for row in reader:
            games += 1
            # pushes: when actual margin is zero, if present
            try:
                actual_margin = float(row.get('actual_margin', '') or 0)
                if abs(actual_margin) < 1e-9:
                    pushes += 1
            except Exception:
                pass
            # wins
            try:
                c = row.get('correct')
                if c is not None and str(c).strip() != '':
                    wins += int(float(c) > 0)
            except Exception:
                pass
            # coverage (optional)
            cov = row.get('covered') or row.get('covered_predicted')
            if cov is not None and cov != '':
                has_cover = True
                cov_norm = str(cov).strip().lower()
                if cov_norm in ('yes', 'y', 'true', '1', 'covered'):
                    cover_yes += 1
                elif cov_norm in ('push', 'p'):
                    cover_push += 1
                else:
                    cover_no += 1
        losses = max(games - pushes - wins, 0)
        denom = max(games - pushes, 1)
        accuracy = wins / denom
        cover_rate = (cover_yes / max(cover_yes + cover_no, 1)) if has_cover else ''
        return {
            'games': games,
            'pushes': pushes,
            'wins': wins,
            'losses': losses,
            'accuracy': accuracy,
            'cover_yes': (cover_yes if has_cover else ''),
            'cover_no': (cover_no if has_cover else ''),
            'cover_push': (cover_push if has_cover else ''),
            'cover_rate': cover_rate,
        }


def write_summary(out_path: str, rows: list[dict]) -> str:
    fieldnames = [
        'league', 'season', 'games', 'pushes', 'wins', 'losses', 'accuracy',
        'cover_yes', 'cover_no', 'cover_push', 'cover_rate'
    ]
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x['season']):
            w.writerow(r)
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Aggregate season-level winners backtests')
    ap.add_argument('--dir', default='./backtests', help='Backtests directory')
    ap.add_argument('--league', choices=VALID_LEAGUES, default='nfl', help='League to aggregate (default: nfl)')
    args = ap.parse_args()

    league = parse_league(args.league)
    d = os.path.abspath(os.path.expanduser(args.dir))
    os.makedirs(d, exist_ok=True)
    latest = latest_by_season(d, league)
    rows = []
    for season, path in latest.items():
        m = compute_metrics(path)
        m['season'] = season
        m['league'] = league
        rows.append(m)
    out = os.path.join(d, f'winners_seasons_summary_{league}.csv')
    write_summary(out, rows)

    # Maintain legacy filename for NFL to avoid breaking existing workflows
    if league == 'nfl':
        legacy_out = os.path.join(d, 'winners_seasons_summary.csv')
        write_summary(legacy_out, rows)
        print(f'Wrote {out} and legacy {legacy_out} with {len(rows)} rows')
    else:
        print(f'Wrote {out} with {len(rows)} rows')


if __name__ == '__main__':
    raise SystemExit(main())

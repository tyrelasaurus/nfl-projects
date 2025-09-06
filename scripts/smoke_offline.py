#!/usr/bin/env python3
"""
Deterministic offline smoke: run spreads using test CSVs.

Uses `test_power.csv` and `test_schedule.csv` to exercise DataLoader and
SpreadCalculator, writing a tiny output CSV under ./output.
"""
import os
import sys
import csv
from typing import Tuple

# Ensure repo root is on sys.path for local package imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nfl_model.spread_model import SpreadCalculator
from nfl_model.data_loader import DataLoader


def run_smoke(week: int = 1) -> Tuple[str, int]:
    root = os.getcwd()
    power_csv = os.path.join(root, 'test_power.csv')
    schedule_csv = os.path.join(root, 'test_schedule.csv')
    out_dir = os.path.join(root, 'output')
    os.makedirs(out_dir, exist_ok=True)

    loader = DataLoader(power_csv, schedule_csv)
    power = loader.load_power_rankings()
    week_df = loader.load_schedule(week)
    matchups = [(r.home_team, r.away_team, getattr(r, 'game_date', '')) for r in week_df.itertuples(index=False)]

    calc = SpreadCalculator(home_field_advantage=2.0)
    results = calc.calculate_week_spreads(matchups, power, week)

    out = os.path.join(out_dir, f'smoke_spreads_week_{week}.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['week','home_team','away_team','projected_spread','betting_line'])
        for r in results:
            w.writerow([r.week, r.home_team, r.away_team, f"{r.projected_spread:+.1f}", calc.format_spread_as_betting_line(r.projected_spread, r.home_team)])
    return out, len(results)


if __name__ == '__main__':
    out, n = run_smoke(1)
    print(f"Wrote {out} with {n} rows")

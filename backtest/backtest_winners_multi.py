#!/usr/bin/env python3
"""
Run winners backtests over multiple seasons and aggregate results.

Usage examples:
  python -m backtest.backtest_winners_multi --seasons 2021-2024 --last-n 17 --hfa 2.0 --output ./backtests
  python -m backtest.backtest_winners_multi --seasons 2022 2023 2024 --last-n 14 --hfa 2.0
"""

import argparse
import os
import sys
import csv
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

from power_ranking.power_ranking.api.espn_client import ESPNClient
from power_ranking.power_ranking.models.power_rankings import PowerRankModel


def ensure_dir(path: str) -> str:
    p = os.path.abspath(os.path.expanduser(path))
    os.makedirs(p, exist_ok=True)
    return p


def get_season_events(client: ESPNClient, season: int) -> List[Dict[str, Any]]:
    data = client.get_season_final_rankings(season)
    return data.get('events', []) or []


def get_prev_season_events(client: ESPNClient, season: int) -> List[Dict[str, Any]]:
    try:
        return client.get_season_final_rankings(season - 1).get('events', []) or []
    except Exception:
        return client.get_last_season_final_rankings().get('events', []) or []


def extract_games_from_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    games: List[Dict[str, Any]] = []
    for event in events:
        for comp in event.get('competitions', []) or []:
            competitors = comp.get('competitors') or []
            home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
            if not home or not away:
                continue
            games.append({
                'event_id': str(event.get('id')),
                'home_id': str(home.get('team', {}).get('id')),
                'away_id': str(away.get('team', {}).get('id')),
                'home_abbr': home.get('team', {}).get('abbreviation'),
                'away_abbr': away.get('team', {}).get('abbreviation'),
                'home_name': home.get('team', {}).get('displayName'),
                'away_name': away.get('team', {}).get('displayName'),
                'home_score': int(home.get('score', 0) or 0),
                'away_score': int(away.get('score', 0) or 0),
                'status': event.get('status', {}).get('type', {}).get('name'),
                'date': (event.get('date') or '').split('T')[0],
                'week': event.get('week_number') or event.get('week', {}).get('number'),
            })
    return games


def build_training_events(prev_events: List[Dict[str, Any]], curr_events: List[Dict[str, Any]], up_to_week_exclusive: int) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    events.extend([e for e in prev_events if e.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL'])
    for e in curr_events:
        wk = e.get('week_number') or e.get('week', {}).get('number')
        if not wk or wk >= up_to_week_exclusive:
            continue
        if e.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL':
            events.append(e)
    # Deduplicate
    seen = set()
    unique: List[Dict[str, Any]] = []
    for e in events:
        eid = str(e.get('id'))
        if eid in seen:
            continue
        unique.append(e)
        seen.add(eid)
    return unique


def run_winners_for_season(season: int, last_n: int, hfa: float, out_dir: str) -> Tuple[Dict[str, Any], str]:
    client = ESPNClient()
    teams = client.get_teams()
    curr_events = get_season_events(client, season)
    prev_events = get_prev_season_events(client, season)
    # Partition by week
    weeks: Dict[int, List[Dict[str, Any]]] = {}
    for e in curr_events:
        wk = e.get('week_number') or e.get('week', {}).get('number')
        if isinstance(wk, int):
            weeks.setdefault(wk, []).append(e)

    model = PowerRankModel()
    total = 0
    correct = 0
    pushes = 0
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    per_game_csv = os.path.join(out_dir, f'backtest_winners_{season}_{ts}.csv')
    with open(per_game_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['season','week','date','home','away','projected_margin','actual_margin','away_score','home_score','final_score','predicted_team','actual_team','correct'])
        for week in range(1, 19):
            wevents = weeks.get(week, [])
            games = extract_games_from_events(wevents)
            if not games:
                continue
            train_events = build_training_events(prev_events, curr_events, week)
            sb_like = {'events': train_events, 'week': {'number': week}, 'season': {'year': season}}
            rankings, comp = model.compute(sb_like, teams, last_n_games=last_n)
            powers: Dict[str, float] = comp.get('power_scores', {})
            for g in games:
                if g['status'] != 'STATUS_FINAL':
                    continue
                home_id, away_id = g['home_id'], g['away_id']
                if home_id not in powers or away_id not in powers:
                    continue
                projected = (powers[home_id] - powers[away_id]) + hfa
                actual_margin = g['home_score'] - g['away_score']
                predicted_team = g['home_name'] if projected > 0 else g['away_name'] if projected < 0 else 'TIE'
                actual_team = g['home_name'] if actual_margin > 0 else g['away_name'] if actual_margin < 0 else 'TIE'
                if predicted_team == 'TIE' or actual_team == 'TIE':
                    pushes += 1
                    is_correct = ''
                else:
                    is_correct = '1' if predicted_team == actual_team else '0'
                    if is_correct == '1':
                        correct += 1
                total += 1
                w.writerow([season, week, g['date'], g['home_abbr'], g['away_abbr'], f"{projected:+.1f}", f"{actual_margin:+.1f}", g['away_score'], g['home_score'], f"{g['away_score']}-{g['home_score']}", predicted_team, actual_team, is_correct])
    # Summary record
    wins = correct
    losses = max(total - pushes - wins, 0)
    acc = (wins / max(total - pushes, 1)) if total else 0.0
    return {
        'season': season,
        'games': total,
        'pushes': pushes,
        'wins': wins,
        'losses': losses,
        'accuracy': acc,
        'per_game_csv': per_game_csv,
    }, per_game_csv


def parse_seasons_arg(values: List[str]) -> List[int]:
    acc: List[int] = []
    for v in values:
        if '-' in v:
            a, b = v.split('-', 1)
            acc.extend(range(int(a), int(b) + 1))
        else:
            acc.append(int(v))
    return sorted(set(acc))


def main():
    ap = argparse.ArgumentParser(description='Run winners backtests over multiple seasons and aggregate results')
    ap.add_argument('--seasons', nargs='+', required=False, help='Seasons list or ranges, e.g., 2021-2024 or 2022 2023 2024')
    ap.add_argument('--last-n', type=int, default=17, help='Last N games per team (default 17)')
    ap.add_argument('--hfa', type=float, default=2.0, help='Home field advantage (default 2.0)')
    ap.add_argument('--output', type=str, default='./backtests', help='Output directory')
    args = ap.parse_args()

    out_dir = ensure_dir(args.output)
    client = ESPNClient()
    if not args.seasons:
        # Default to last 4 seasons ending with last completed
        last = client.get_last_completed_season()
        seasons = [last - 3, last - 2, last - 1, last]
    else:
        seasons = parse_seasons_arg(args.seasons)

    # Aggregate summaries
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    agg_csv = os.path.join(out_dir, f'backtest_winners_aggregate_{ts}.csv')
    index_html = os.path.join(out_dir, f'backtest_winners_index_{ts}.html')

    records = []
    for season in seasons:
        rec, _ = run_winners_for_season(season, args.last_n, args.hfa, out_dir)
        records.append(rec)

    with open(agg_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['season', 'games', 'pushes', 'wins', 'losses', 'accuracy', 'per_game_csv'])
        for r in records:
            w.writerow([r['season'], r['games'], r['pushes'], r['wins'], r['losses'], f"{r['accuracy']:.3f}", r['per_game_csv']])

    # HTML index
    with open(index_html, 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Winners Backtests Index</title>"
                "<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3}</style>"
                "</head><body>")
        f.write("<h1>Winners Backtests (Aggregate)</h1>")
        f.write("<table><tr><th>Season</th><th>Games</th><th>Pushes</th><th>Wins</th><th>Losses</th><th>Accuracy</th><th>CSV</th></tr>")
        for r in records:
            f.write("<tr>" +
                    f"<td>{r['season']}</td>" +
                    f"<td>{r['games']}</td>" +
                    f"<td>{r['pushes']}</td>" +
                    f"<td>{r['wins']}</td>" +
                    f"<td>{r['losses']}</td>" +
                    f"<td>{r['accuracy']:.3f}</td>" +
                    f"<td>{r['per_game_csv']}</td>" +
                    "</tr>")
        f.write("</table></body></html>")

    print("Aggregate winners backtests complete")
    print(f"Seasons: {', '.join(map(str,seasons))}")
    print(f"Aggregate CSV: {agg_csv}")
    print(f"Index HTML:    {index_html}")


if __name__ == '__main__':
    sys.exit(main())


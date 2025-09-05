#!/usr/bin/env python3
"""
Backtesting suite for NFL spread model using Power Rankings (last-N across seasons).

For a given season (default: last completed), iterates weeks 1-18 and:
 - Builds power rankings from prior data only (previous season + weeks < current)
 - Computes projected spreads (HFA configurable)
 - Fetches market lines and final results from ESPN
 - Evaluates ATS accuracy and error metrics
 - Exports per-game CSV and a summary CSV/HTML

Usage examples:
  python -m backtest.backtest_spreads --season 2024 --last-n 17 --hfa 2.0 --output ./backtests
"""

import argparse
import os
import sys
import csv
from typing import Dict, List, Tuple, Any
from datetime import datetime, timezone

from power_ranking.power_ranking.api.espn_client import ESPNClient
from power_ranking.power_ranking.models.power_rankings import PowerRankModel


def ensure_dir(path: str) -> str:
    p = os.path.abspath(os.path.expanduser(path))
    os.makedirs(p, exist_ok=True)
    return p


def fetch_week_data(client: ESPNClient, season: int, week: int) -> Dict[str, Any]:
    return client.get_scoreboard(week=week, season=season)


def fetch_market_from_competition(comp: Dict[str, Any], home_abbr: str, away_abbr: str, hfa: float) -> Tuple[Any, Any]:
    odds_list = comp.get('odds') or []
    market_spread = None
    market_line = None
    if odds_list:
        o = odds_list[0]
        details = o.get('details') or ''  # e.g., "KC -3.5"
        fav_abbr = None
        spread_val = o.get('spread')
        try:
            spread_val = float(spread_val) if spread_val is not None else None
        except Exception:
            spread_val = None
        if details:
            parts = details.split()
            if len(parts) >= 2:
                fav_abbr = parts[0].strip()
                if spread_val is None:
                    try:
                        spread_val = float(parts[1])
                    except Exception:
                        pass
        if spread_val is not None:
            if fav_abbr == home_abbr:
                market_spread = abs(spread_val)
            elif fav_abbr == away_abbr:
                market_spread = -abs(spread_val)
        if market_spread is None:
            hto = o.get('homeTeamOdds') or {}
            ato = o.get('awayTeamOdds') or {}
            if hto.get('favorite') is True and spread_val is not None:
                market_spread = abs(spread_val)
            elif ato.get('favorite') is True and spread_val is not None:
                market_spread = -abs(spread_val)
        # Build a market line string
        if market_spread is not None:
            ml = f"{home_abbr} -{abs(market_spread):.1f}" if market_spread > 0 else (
                 f"{home_abbr} +{abs(market_spread):.1f}" if market_spread < 0 else f"{home_abbr} PK")
            market_line = ml
    return market_spread, market_line


def extract_matchups(week_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    games = []
    for event in week_data.get('events', []) or []:
        for comp in event.get('competitions', []) or []:
            competitors = comp.get('competitors') or []
            home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
            if not home or not away:
                continue
            games.append({
                'event_id': event.get('id'),
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
                'comp': comp,
            })
    return games


def build_training_events(client: ESPNClient, season: int, up_to_week_exclusive: int) -> List[Dict[str, Any]]:
    # Prior season full
    try:
        prev = client.get_season_final_rankings(season - 1)
    except Exception:
        prev = client.get_last_season_final_rankings()
    events = list(prev.get('events', []) or [])
    # Current season prior weeks (1..week-1)
    for w in range(1, up_to_week_exclusive):
        wd = fetch_week_data(client, season, w)
        evs = wd.get('events', []) or []
        # Keep only finals to avoid leakage
        events.extend([e for e in evs if e.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL'])
    # Deduplicate by event id
    seen = set()
    unique = []
    for e in events:
        eid = str(e.get('id'))
        if eid in seen:
            continue
        unique.append(e)
        seen.add(eid)
    return unique


def main():
    parser = argparse.ArgumentParser(description='Backtest NFL spreads against last season actuals')
    parser.add_argument('--season', type=int, help='Season year (default: last completed)')
    parser.add_argument('--last-n', type=int, default=17, help='Last N games per team for power (default: 17)')
    parser.add_argument('--hfa', type=float, default=2.0, help='Home field advantage (default: 2.0)')
    parser.add_argument('--output', type=str, default='./backtests', help='Output directory')
    args = parser.parse_args()

    client = ESPNClient()
    if not args.season:
        args.season = client.get_last_completed_season()

    out_dir = ensure_dir(args.output)
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    per_game_csv = os.path.join(out_dir, f'backtest_{args.season}_{ts}.csv')
    summary_csv = os.path.join(out_dir, f'backtest_summary_{args.season}_{ts}.csv')

    # Fetch team list (for names mapping if needed)
    teams = client.get_teams()

    model = PowerRankModel()
    total_games = 0
    correct = 0
    pushes = 0
    mae_market = []
    mae_actual = []

    with open(per_game_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'season', 'week', 'date', 'home', 'away',
            'projected', 'market', 'edge', 'actual_margin', 'ats_pick', 'ats_result', 'correct'
        ])

        for week in range(1, 19):
            week_data = fetch_week_data(client, args.season, week)
            games = extract_matchups(week_data)
            if not games:
                continue

            # Build training set from historical data (no data from current week)
            merged_events = build_training_events(client, args.season, week)
            scoreboard_like = {'events': merged_events, 'week': {'number': week}, 'season': {'year': args.season}}

            rankings, comp = model.compute(scoreboard_like, teams, last_n_games=args.last_n)
            power_scores: Dict[str, float] = comp.get('power_scores', {})

            for g in games:
                if g['status'] != 'STATUS_FINAL':
                    # Skip non-final (should be all final for completed season)
                    continue
                home_id = g['home_id']
                away_id = g['away_id']
                if home_id not in power_scores or away_id not in power_scores:
                    continue
                home_power = power_scores[home_id]
                away_power = power_scores[away_id]
                projected = (home_power - away_power) + args.hfa
                market_spread, market_line = fetch_market_from_competition(g['comp'], g['home_abbr'], g['away_abbr'], args.hfa)
                # Actual margin
                actual_margin = g['home_score'] - g['away_score']
                # ATS pick and result
                ats_pick = 'home' if projected > (market_spread if market_spread is not None else 0.0) else (
                           'away' if projected < (market_spread if market_spread is not None else 0.0) else 'push')
                ats_result = 'push'
                is_correct = ''
                if market_spread is not None:
                    # Home covers if actual_margin > market_spread
                    if actual_margin > market_spread:
                        ats_result = 'home'
                    elif actual_margin < market_spread:
                        ats_result = 'away'
                    else:
                        ats_result = 'push'
                    if ats_result == 'push' or ats_pick == 'push':
                        pushes += 1
                        is_correct = ''
                    else:
                        is_correct = '1' if ats_pick == ats_result else '0'
                        if is_correct == '1':
                            correct += 1
                    total_games += 1
                    mae_market.append(abs(projected - market_spread))
                # Error vs actual margin
                mae_actual.append(abs(projected - actual_margin))

                writer.writerow([
                    args.season, week, g['date'], g['home_abbr'], g['away_abbr'],
                    f"{projected:+.1f}",
                    (f"{market_spread:+.1f}" if market_spread is not None else ''),
                    (f"{(projected - market_spread):+.1f}" if market_spread is not None else ''),
                    f"{actual_margin:+.1f}", ats_pick, ats_result, is_correct
                ])

    # Summarize
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['season', 'games_scored', 'pushes', 'ats_wins', 'ats_losses', 'ats_win_pct', 'mae_vs_market', 'mae_vs_actual'])
        ats_losses = max(total_games - correct - pushes, 0)
        ats_win_pct = (correct / max(total_games, 1)) if total_games else 0.0
        writer.writerow([
            args.season, total_games, pushes, correct, ats_losses,
            f"{ats_win_pct:.3f}",
            (f"{(sum(mae_market)/len(mae_market)):.2f}" if mae_market else ''),
            (f"{(sum(mae_actual)/len(mae_actual)):.2f}" if mae_actual else '')
        ])

    print(f"Backtest complete for {args.season}")
    print(f"Per-game CSV: {per_game_csv}")
    print(f"Summary CSV:  {summary_csv}")


if __name__ == '__main__':
    sys.exit(main())


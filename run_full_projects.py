#!/usr/bin/env python3
"""
End-to-end runner for NFL Projects Suite.

Executes the Power Rankings suite and then the NFL Spread Model suite, and
prints results to the terminal while also exporting a consolidated CSV and
an HTML summary page.

Usage examples:
  python run_full_projects.py --week 1 --last-n 17 --output ./output
  python run_full_projects.py --week 1 --last-n 17 --schedule test_schedule.csv
"""

import argparse
import os
import sys
import csv
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any
import yaml


def ensure_abs(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def build_team_abbr_map() -> Dict[str, str]:
    """Map full team names to standard abbreviations for nfl_model compatibility."""
    return {
        'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
        'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
        'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
        'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
    }


def run_power_rankings(week: int | None, last_n: int, output_dir: str) -> Tuple[str, List[Tuple[str, str, float]], Dict[str, Any]]:
    """Run power rankings and return (csv_path, rankings, computation_data)."""
    from power_ranking.power_ranking.api.espn_client import ESPNClient
    from power_ranking.power_ranking.models.power_rankings import PowerRankModel
    from power_ranking.power_ranking.export.csv_exporter import CSVExporter

    client = ESPNClient()
    # Load tuned weights if present
    tuned_weights = None
    try:
        with open(os.path.join(os.getcwd(), 'calibration', 'params.yaml'), 'r') as f:
            cfg = yaml.safe_load(f) or {}
            w = cfg.get('model', {}).get('weights')
            if isinstance(w, dict):
                tuned_weights = {
                    'season_avg_margin': float(w.get('season_avg_margin', 0.55)),
                    'rolling_avg_margin': float(w.get('rolling_avg_margin', 0.20)),
                    'sos': float(w.get('sos', 0.20)),
                    'recency_factor': float(w.get('recency_factor', 0.05)),
                }
    except Exception:
        tuned_weights = None

    model = PowerRankModel(weights=tuned_weights) if tuned_weights else PowerRankModel()

    # Determine week
    if week is None:
        week = client.get_current_week()

    teams = client.get_teams()

    # Fetch current and last completed season; merge for last-N
    current = client.get_scoreboard(week=week)
    try:
        last_year = client.get_last_completed_season()
        last_season = client.get_season_final_rankings(last_year)
    except Exception:
        last_season = None

    if not client.has_current_season_games(week=week):
        merged = last_season or current or {'events': []}
    else:
        merged_ids = set()
        merged_events = []
        for ev in (current.get('events') or []):
            eid = str(ev.get('id'))
            if eid not in merged_ids:
                merged_events.append(ev)
                merged_ids.add(eid)
        if last_season and last_season.get('events'):
            for ev in last_season['events']:
                eid = str(ev.get('id'))
                if eid not in merged_ids:
                    merged_events.append(ev)
                    merged_ids.add(eid)
        merged = {'events': merged_events, 'week': current.get('week') or {'number': week}}

    rankings, computation = model.compute(merged, teams, last_n_games=last_n)
    exporter = CSVExporter(output_dir)
    # Use a friendly export name
    export_week = f"week_{week}_last{last_n}"
    csv_path = exporter.export_rankings(rankings, export_week)
    return csv_path, rankings, computation


def make_abbrev_power_csv(full_power_csv: str, output_dir: str) -> str:
    """Convert full-name power CSV to abbreviation-based CSV for nfl_model."""
    import pandas as pd

    abbr_map = build_team_abbr_map()
    df = pd.read_csv(full_power_csv)
    # Expect columns: team_id, team_name, power_score
    df_abbr = df.copy()
    df_abbr['team_name'] = df_abbr['team_name'].map(lambda n: abbr_map.get(n, n))
    out_path = os.path.join(output_dir, 'power_rankings_abbr.csv')
    df_abbr[['team_name', 'power_score']].to_csv(out_path, index=False)
    return out_path


def run_spread_model(power_csv: str, schedule_csv: str, week: int, output_dir: str,
                     odds_map: Dict[Tuple[str, str], Dict[str, Any]] | None = None) -> Tuple[str, List[Dict]]:
    """Run nfl_model spreads using provided power rankings and schedule.

    Returns (spreads_csv_path, results_list_of_dicts)
    """
    import pandas as pd
    from nfl_model.spread_model import SpreadCalculator
    from nfl_model.data_loader import DataLoader

    loader = DataLoader(power_csv, schedule_csv)
    power = loader.load_power_rankings()
    week_df = loader.load_schedule(week)
    matchups = [(r.home_team, r.away_team, getattr(r, 'game_date', '')) for r in week_df.itertuples(index=False)]

    # Load calibration parameters if available
    calib_path = os.path.join(os.getcwd(), 'calibration', 'params.yaml')
    a, b = 0.0, 1.0
    hfa = 2.0
    try:
        if os.path.exists(calib_path):
            with open(calib_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
                a = float(cfg.get('calibration', {}).get('margin', {}).get('a', 0.0))
                b = float(cfg.get('calibration', {}).get('margin', {}).get('b', 1.0))
                hfa = float(cfg.get('model', {}).get('hfa', 2.0))
    except Exception:
        pass

    calc = SpreadCalculator(home_field_advantage=hfa)
    results = calc.calculate_week_spreads(matchups, power, week)

    # Write a CSV
    spreads_csv = os.path.join(output_dir, f'spreads_week_{week}.csv')
    with open(spreads_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['week', 'home_team', 'away_team', 'projected_spread_raw', 'projected_spread', 'betting_line', 'market_spread', 'market_line', 'edge', 'game_date'])
        for r in results:
            mk = None
            ml = None
            edge = None
            if odds_map:
                info = odds_map.get((r.home_team, r.away_team))
                if info:
                    mk = info.get('market_spread')
                    ml = info.get('market_line')
                    if mk is not None:
                        edge = (a + b * r.projected_spread) - mk
            writer.writerow([
                r.week, r.home_team, r.away_team, f"{r.projected_spread:.1f}",
                f"{(a + b * r.projected_spread):.1f}",
                calc.format_spread_as_betting_line((a + b * r.projected_spread), r.home_team),
                (f"{mk:.1f}" if isinstance(mk, (int, float)) else ''),
                (ml or ''),
                (f"{edge:+.1f}" if isinstance(edge, (int, float)) else ''),
                r.game_date
            ])

    # Collect results list
    results_list = [
        {
            'week': r.week,
            'home_team': r.home_team,
            'away_team': r.away_team,
            'projected_spread_raw': r.projected_spread,
            'projected_spread': (a + b * r.projected_spread),
            'betting_line': calc.format_spread_as_betting_line((a + b * r.projected_spread), r.home_team),
            'market_spread': (odds_map or {}).get((r.home_team, r.away_team), {}).get('market_spread'),
            'market_line': (odds_map or {}).get((r.home_team, r.away_team), {}).get('market_line'),
            'edge': ((a + b * r.projected_spread) - (odds_map or {}).get((r.home_team, r.away_team), {}).get('market_spread')
                     if (odds_map or {}).get((r.home_team, r.away_team), {}).get('market_spread') is not None else None),
            'game_date': r.game_date,
        }
        for r in results
    ]

    return spreads_csv, results_list


def fetch_schedule_from_espn(week: int, season: int, output_dir: str) -> Tuple[str, Dict[Tuple[str, str], Dict[str, Any]]]:
    """Fetch Week schedule from ESPN and write a normalized schedule CSV.

    Returns path to the CSV with columns: week,home_team,away_team,game_date
    Teams are written as abbreviations.
    """
    from power_ranking.power_ranking.api.espn_client import ESPNClient

    client = ESPNClient()
    data = client.get_scoreboard(week=week, season=season)

    abbr_map = build_team_abbr_map()
    rows: List[Tuple[int, str, str, str]] = []
    odds_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for event in data.get('events', []) or []:
        comps = (event.get('competitions') or [])
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get('competitors') or []
        home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
        away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
        if not home or not away:
            continue
        home_name = home.get('team', {}).get('displayName') or ''
        away_name = away.get('team', {}).get('displayName') or ''
        home_abbr_full = home.get('team', {}).get('abbreviation') or abbr_map.get(home_name, home_name)
        away_abbr_full = away.get('team', {}).get('abbreviation') or abbr_map.get(away_name, away_name)
        # Map to abbreviations
        home_abbr = abbr_map.get(home_name, home_abbr_full)
        away_abbr = abbr_map.get(away_name, away_abbr_full)
        game_date = (event.get('date') or '').split('T')[0]
        rows.append((week, home_abbr, away_abbr, game_date))

        # Parse odds if present
        odds_list = comp.get('odds') or []
        market_spread = None
        market_line = None
        if odds_list:
            o = odds_list[0]
            try:
                spread_val = float(o.get('spread')) if o.get('spread') is not None else None
            except Exception:
                spread_val = None
            details = o.get('details') or ''  # e.g., "KC -3.5"
            fav_abbr = None
            if details:
                parts = details.split()
                if len(parts) >= 2:
                    fav_abbr = parts[0].strip()
                    try:
                        spread_val = float(parts[1]) if spread_val is None else spread_val
                    except Exception:
                        pass
            # Determine home/away favorite and set sign so positive means home favored
            if spread_val is not None:
                if fav_abbr:
                    # Compare with home/away abbreviations
                    if fav_abbr == home_abbr or fav_abbr == home_abbr_full:
                        market_spread = abs(spread_val)
                    elif fav_abbr == away_abbr or fav_abbr == away_abbr_full:
                        market_spread = -abs(spread_val)
                # Fallback: use homeTeamOdds.favorite if available
                if market_spread is None:
                    hto = o.get('homeTeamOdds') or {}
                    ato = o.get('awayTeamOdds') or {}
                    if hto.get('favorite') is True and spread_val is not None:
                        market_spread = abs(spread_val)
                    elif ato.get('favorite') is True and spread_val is not None:
                        market_spread = -abs(spread_val)
            # Format market line using home perspective
            if market_spread is not None:
                # Use SpreadCalculator-style formatting
                try:
                    from nfl_model.spread_model import SpreadCalculator
                    market_line = SpreadCalculator(home_field_advantage=2.0).format_spread_as_betting_line(market_spread, home_abbr)
                except Exception:
                    # Manual format
                    if market_spread > 0:
                        market_line = f"{home_abbr} -{abs(market_spread):.1f}"
                    elif market_spread < 0:
                        market_line = f"{home_abbr} +{abs(market_spread):.1f}"
                    else:
                        market_line = f"{home_abbr} PK"
        odds_map[(home_abbr, away_abbr)] = {
            'market_spread': market_spread,
            'market_line': market_line,
        }

    if not rows:
        raise RuntimeError(f"No games found from ESPN for season {season} week {week}")

    out = os.path.join(output_dir, f'schedule_week_{week}_{season}.csv')
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['week', 'home_team', 'away_team', 'game_date'])
        writer.writerows(rows)
    return out, odds_map


def write_summary_csv(output_dir: str,
                      rankings: List[Tuple[str, str, float]],
                      computation: Dict[str, Any],
                      spreads: List[Dict],
                      week: int,
                      last_n: int) -> str:
    path = os.path.join(output_dir, f'summary_week_{week}_last{last_n}.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header for detailed power rankings
        writer.writerow([
            'Section', 'Rank', 'Team', 'Power Score',
            'GP', 'W', 'L', 'Win%', 'AvgMargin', 'PtsFor', 'PtsAgainst',
            'R_GP', 'R_AvgMargin', 'R_Win%', 'SOS'
        ])
        season_stats = computation.get('season_stats', {})
        rolling_stats = computation.get('rolling_stats', {})
        sos_scores = computation.get('sos_scores', {})
        for i, (team_id, team_name, score) in enumerate(rankings, 1):
            s = season_stats.get(team_id, {})
            r = rolling_stats.get(team_id, {})
            sos = sos_scores.get(team_id, 0.0)
            writer.writerow([
                'PowerRankings', i, team_name, f"{score:.3f}",
                s.get('games_played', 0), s.get('wins', 0), s.get('losses', 0), f"{s.get('win_pct', 0):.3f}",
                f"{s.get('avg_margin', 0):.2f}", f"{s.get('avg_points_for', 0):.1f}", f"{s.get('avg_points_against', 0):.1f}",
                r.get('games_played', 0), f"{r.get('avg_margin', 0):.2f}", f"{r.get('win_pct', 0):.3f}", f"{sos:.3f}"
            ])
        # Spreads with market
        writer.writerow(['Spreads', 'Header', 'Away @ Home', 'Projected', 'Betting Line', 'Market', 'Edge'])
        for row in spreads:
            writer.writerow([
                'Spreads', 'Game', f"{row['away_team']} @ {row['home_team']}",
                f"{row['projected_spread']:+.1f}", row['betting_line'],
                (row.get('market_line') or ''),
                (f"{row.get('edge'):+.1f}" if isinstance(row.get('edge'), (int, float)) else '')
            ])
    return path


def write_summary_html(output_dir: str,
                       rankings: List[Tuple[str, str, float]],
                       computation: Dict[str, Any],
                       spreads: List[Dict],
                       week: int,
                       last_n: int) -> str:
    path = os.path.join(output_dir, f'summary_week_{week}_last{last_n}.html')
    ts = datetime.now(timezone.utc).isoformat()
    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>NFL Projects Summary</title>",
        "<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:8px} th{background:#f3f3f3}</style>",
        "</head><body>",
        f"<h1>NFL Projects Summary - Week {week} (Last {last_n} games)</h1>",
        f"<p>Generated: {ts} UTC</p>",
        "<h2>Power Rankings (All Teams)</h2>",
        "<table><tr>"
        "<th>Rank</th><th>Team</th><th>Power</th>"
        "<th>GP</th><th>W</th><th>L</th><th>Win%</th>"
        "<th>AvgMargin</th><th>PtsFor</th><th>PtsAg</th>"
        "<th>R_GP</th><th>R_AvgMargin</th><th>R_Win%</th><th>SOS</th>"
        "</tr>",
    ]
    season_stats = computation.get('season_stats', {})
    rolling_stats = computation.get('rolling_stats', {})
    sos_scores = computation.get('sos_scores', {})
    for i, (team_id, team_name, score) in enumerate(rankings, 1):
        s = season_stats.get(team_id, {})
        r = rolling_stats.get(team_id, {})
        sos = sos_scores.get(team_id, 0.0)
        html.append(
            "<tr>"
            f"<td>{i}</td><td>{team_name}</td><td>{score:.3f}</td>"
            f"<td>{s.get('games_played', 0)}</td><td>{s.get('wins', 0)}</td><td>{s.get('losses', 0)}</td><td>{s.get('win_pct', 0):.3f}</td>"
            f"<td>{s.get('avg_margin', 0):.2f}</td><td>{s.get('avg_points_for', 0):.1f}</td><td>{s.get('avg_points_against', 0):.1f}</td>"
            f"<td>{r.get('games_played', 0)}</td><td>{r.get('avg_margin', 0):.2f}</td><td>{r.get('win_pct', 0):.3f}</td><td>{sos:.3f}</td>"
            "</tr>"
        )
    html.append("</table>")

    html.append("<h2>Spread Predictions</h2>")
    html.append("<table><tr><th>Matchup</th><th>Projected</th><th>Betting Line</th><th>Market</th><th>Edge</th><th>Date</th></tr>")
    for row in spreads:
        matchup = f"{row['away_team']} @ {row['home_team']}"
        market_line = row.get('market_line') or ''
        edge = row.get('edge')
        edge_str = f"{edge:+.1f}" if isinstance(edge, (int, float)) else ''
        html.append(
            f"<tr><td>{matchup}</td><td>{row['projected_spread']:+.1f}</td><td>{row['betting_line']}</td><td>{market_line}</td><td>{edge_str}</td><td>{row['game_date']}</td></tr>"
        )
    html.append("</table>")
    html.append("</body></html>")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html))
    return path


def main():
    parser = argparse.ArgumentParser(description="Run Power Rankings then NFL Model and summarize results")
    parser.add_argument('--week', type=int, help='NFL week number (default: autodetect)')
    parser.add_argument('--last-n', type=int, default=17, help='Most recent games per team (default: 17)')
    parser.add_argument('--schedule', type=str, default='auto', help='Schedule CSV path or "auto" to fetch from ESPN')
    parser.add_argument('--season', type=int, default=datetime.now().year, help='Season year for schedule when using auto')
    parser.add_argument('--output', type=str, default='./output', help='Output directory for artifacts')
    args = parser.parse_args()

    output_dir = ensure_abs(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # 1) Power rankings
    pr_csv, rankings, computation = run_power_rankings(args.week, args.last_n, output_dir)
    print("\n=== Power Rankings (All Teams) ===")
    for i, (_, team_name, score) in enumerate(rankings, 1):
        print(f"{i:2d}. {team_name:<25} {score:6.3f}")
    print(f"Saved power rankings CSV: {pr_csv}")

    # 2) Prepare power rankings for nfl_model (abbreviations), then run spreads
    pr_abbrev_csv = make_abbrev_power_csv(pr_csv, output_dir)
    # Determine target week for NFL model schedule use
    target_week = args.week or 1
    # Resolve schedule
    if args.schedule == 'auto':
        try:
            schedule_csv, odds_map = fetch_schedule_from_espn(week=target_week, season=args.season, output_dir=output_dir)
            print(f"Fetched schedule from ESPN: {schedule_csv}")
        except Exception as e:
            raise RuntimeError(f"Failed to auto-fetch schedule: {e}")
    else:
        schedule_csv = ensure_abs(args.schedule)
        if not os.path.exists(schedule_csv):
            raise FileNotFoundError(f"Schedule CSV not found: {schedule_csv}")
        odds_map = None

    spreads_csv, spreads = run_spread_model(pr_abbrev_csv, schedule_csv, target_week, output_dir, odds_map)
    print("\n=== Spread Predictions ===")
    for row in spreads:
        market = row.get('market_spread')
        edge = row.get('edge')
        suffix = ''
        if market is not None:
            ml = row.get('market_line') or ''
            suffix = f" | Market: {ml} | Edge: {edge:+.1f}"
        print(f"{row['away_team']} @ {row['home_team']}: {row['projected_spread']:+.1f} ({row['betting_line']}){suffix}")
    print(f"Saved spreads CSV: {spreads_csv}")

    # 3) Combined summary CSV and HTML
    summary_csv = write_summary_csv(output_dir, rankings, computation, spreads, target_week, args.last_n)
    summary_html = write_summary_html(output_dir, rankings, computation, spreads, target_week, args.last_n)
    print("\n=== Summary Artifacts ===")
    print(f"Summary CSV:  {summary_csv}")
    print(f"Summary HTML: {summary_html}")


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
End-to-end runner for the Projects Suite (NFL + NCAA).

Executes the Power Rankings suite and corresponding spread model for the
requested league, printing results while exporting a consolidated CSV and
an HTML summary page.

Usage examples:
  python run_full_projects.py --week 1 --league nfl --last-n 17 --output ./output
  python run_full_projects.py --week 1 --league ncaa --schedule test_schedule.csv
"""

import argparse
import os
import sys
import csv
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any
import yaml
import hashlib


def ensure_abs(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def build_team_abbr_map(league: str, teams: List[Dict[str, Any]] | None = None) -> Dict[str, str]:
    """Map full team names to standard abbreviations for downstream loaders."""

    lg = (league or 'nfl').lower()
    if lg in {'ncaa', 'college', 'college-football'}:
        mapping: Dict[str, str] = {}
        for entry in teams or []:
            info = entry.get('team') if isinstance(entry, dict) else None
            info = info or entry
            name = (info or {}).get('displayName') or (info or {}).get('name')
            abbr = (info or {}).get('abbreviation') or (info or {}).get('shortDisplayName')
            if name:
                mapping[name] = abbr or name
        return mapping

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


def run_power_rankings(week: int | None,
                       last_n: int,
                       output_dir: str,
                       league: str) -> Tuple[str, List[Tuple[str, str, float]], Dict[str, Any], List[Dict[str, Any]]]:
    """Run power rankings and return (csv_path, rankings, computation_data, teams)."""
    from power_ranking.power_ranking.api.client_factory import get_client
    from power_ranking.power_ranking.models.power_rankings import PowerRankModel
    from power_ranking.power_ranking.export.csv_exporter import CSVExporter

    client = get_client('sync', league=league)
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

    def _extract_team_names(events: List[Dict[str, Any]] | None) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for ev in events or []:
            for comp in ev.get('competitions') or []:
                for competitor in comp.get('competitors') or []:
                    team = competitor.get('team') or {}
                    tid = team.get('id')
                    name = team.get('displayName') or team.get('name')
                    if tid and name:
                        mapping.setdefault(str(tid), str(name))
        return mapping

    # Determine target week and load teams
    if week is None:
        week = client.get_current_week()

    teams = client.get_teams()

    # Collect all completed current-season games through the requested week
    merged_ids: set[str] = set()
    current_events: List[Dict[str, Any]] = []
    for w in range(1, max(1, int(week)) + 1):
        try:
            sb = client.get_scoreboard(week=w)
        except Exception:
            continue
        for ev in (sb.get('events') or []):
            # Keep only completed games
            status_name = (ev.get('status') or {}).get('type', {}).get('name')
            if status_name != 'STATUS_FINAL':
                continue
            eid = str(ev.get('id'))
            if eid in merged_ids:
                continue
            # Ensure week_number present for downstream logic/exports
            wk_info = ev.get('week') or {}
            ev.setdefault('week_number', wk_info.get('number', w))
            current_events.append(ev)
            merged_ids.add(eid)

    # Append previous season to allow last-N to reach back into prior year
    try:
        last_year = client.get_last_completed_season()
        last_season = client.get_season_final_rankings(last_year)
    except Exception:
        last_season = None

    if last_season and last_season.get('events'):
        for ev in (last_season.get('events') or []):
            eid = str(ev.get('id'))
            if eid in merged_ids:
                continue
            # Keep only completed games if status is present; otherwise assume completed for historical
            status_name = (ev.get('status') or {}).get('type', {}).get('name') if isinstance(ev, dict) else None
            if status_name and status_name != 'STATUS_FINAL':
                continue
            current_events.append(ev)
            merged_ids.add(eid)

    merged = {
        'events': current_events,
        'week': {'number': week},
    }

    rankings, computation = model.compute(merged, teams, last_n_games=last_n)
    combined_team_map: Dict[str, str] = {}
    combined_team_map.update(computation.get('teams_map', {}))
    combined_team_map.update(_extract_team_names(current_events))
    if last_season and isinstance(last_season, dict):
        combined_team_map.update(_extract_team_names(last_season.get('events')))
    computation['teams_map'] = combined_team_map
    rankings = [
        (team_id, combined_team_map.get(team_id, team_name), score)
        for team_id, team_name, score in rankings
    ]
    exporter = CSVExporter(output_dir)
    # Use a friendly export name
    export_week = f"{league.lower()}_week_{week}_last{last_n}"
    csv_path = exporter.export_rankings(rankings, export_week)
    return csv_path, rankings, computation, teams


def prepare_power_csv(full_power_csv: str,
                      output_dir: str,
                      league: str,
                      teams: List[Dict[str, Any]] | None = None,
                      teams_map: Dict[str, str] | None = None) -> str:
    """Normalize power ranking CSV for downstream spread models."""
    import pandas as pd

    df = pd.read_csv(full_power_csv)
    lg = (league or 'nfl').lower()
    abbr_map = build_team_abbr_map(lg, teams)

    if lg == 'nfl':
        df_abbr = df.copy()
        df_abbr['team_name'] = df_abbr['team_name'].map(lambda n: abbr_map.get(n, n))
        out_path = os.path.join(output_dir, 'power_rankings_abbr.csv')
        df_abbr[['team_name', 'power_score']].to_csv(out_path, index=False)
        return out_path

    # NCAA: keep descriptive names for schedule alignment, but emit abbreviations as helper column
    df_ncaa = df.copy()
    id_to_name: Dict[str, str] = {}
    if teams_map:
        id_to_name.update({str(k): str(v) for k, v in teams_map.items()})

    id_to_abbr: Dict[str, str] = {}
    for entry in teams or []:
        team = entry.get('team') if isinstance(entry, dict) else None
        team = team or entry
        tid = team.get('id') if isinstance(team, dict) else None
        if not tid:
            continue
        tid_str = str(tid)
        display = str(team.get('displayName') or team.get('name') or id_to_name.get(tid_str, tid_str))
        abbr = str(team.get('abbreviation') or team.get('shortDisplayName') or display)
        id_to_name.setdefault(tid_str, display)
        id_to_abbr[tid_str] = abbr

    if 'team_id' in df_ncaa.columns:
        df_ncaa['team_id'] = df_ncaa['team_id'].astype(str)
        df_ncaa['team_name'] = df_ncaa['team_id'].map(id_to_name).fillna(df_ncaa.get('team_name'))
        df_ncaa['team_abbr'] = df_ncaa['team_id'].map(lambda tid: id_to_abbr.get(tid, abbr_map.get(id_to_name.get(tid, ''), id_to_name.get(tid, tid))))
    else:
        df_ncaa['team_name'] = df_ncaa['team_name'].astype(str).map(lambda n: id_to_name.get(n, n))
        df_ncaa['team_abbr'] = df_ncaa['team_name'].map(lambda n: abbr_map.get(n, n))

    df_ncaa['team_name'] = df_ncaa['team_name'].astype(str)
    if abbr_map and 'team_abbr' not in df_ncaa:
        df_ncaa['team_abbr'] = df_ncaa['team_name'].map(lambda n: abbr_map.get(n, n))
    out_path = os.path.join(output_dir, 'power_rankings_ncaa.csv')
    df_ncaa[['team_name', 'power_score']].to_csv(out_path, index=False)
    return out_path


def run_spread_model(power_csv: str,
                     schedule_csv: str,
                     week: int,
                     output_dir: str,
                     league: str,
                     odds_map: Dict[Tuple[str, str], Dict[str, Any]] | None = None) -> Tuple[str, List[Dict]]:
    """Run league-specific spread model using provided power rankings and schedule.

    Returns (spreads_csv_path, results_list_of_dicts)
    """
    import pandas as pd

    lg = (league or 'nfl').lower()
    if lg == 'ncaa':
        from ncaa_model.spread_model import SpreadCalculator
        from ncaa_model.data_loader import DataLoader
        calib_filename = 'ncaa_params.yaml'
        default_hfa = 3.0
    else:
        from nfl_model.spread_model import SpreadCalculator
        from nfl_model.data_loader import DataLoader
        calib_filename = 'params.yaml'
        default_hfa = 2.0

    loader = DataLoader(power_csv, schedule_csv)
    power = loader.load_power_rankings()
    week_df = loader.load_schedule(week)
    matchups = [(r.home_team, r.away_team, getattr(r, 'game_date', '')) for r in week_df.itertuples(index=False)]

    # Load calibration parameters if available
    calib_path = os.path.join(os.getcwd(), 'calibration', calib_filename)
    a, b = 0.0, 1.0
    hfa = default_hfa
    prob_map = None
    blend = {'low': 3.0, 'high': 7.0}
    enable_calibration = True
    use_calibrated_probability = True
    params_version = None
    # Edge policy config
    edge_enabled = False
    edge_threshold = 0.0
    conf_enabled = False
    conf_margin_threshold = 0.0
    try:
        if os.path.exists(calib_path):
            with open(calib_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
                params_version = cfg.get('version')
                cal_cfg = cfg.get('calibration', {})
                margin_cfg = cal_cfg.get('margin', {})
                a = float(margin_cfg.get('a', 0.0))
                b = float(margin_cfg.get('b', 1.0))
                hfa = float(cfg.get('model', {}).get('hfa', 2.0))
                bl = cal_cfg.get('blend', {})
                blend['low'] = float(bl.get('low', 3.0))
                blend['high'] = float(bl.get('high', 7.0))
                enable_calibration = bool(cal_cfg.get('enable_calibration', True))
                use_calibrated_probability = bool(cal_cfg.get('use_calibrated_probability', True))
                policy = cal_cfg.get('policy', {})
                edge_policy = policy.get('edge', {}) if isinstance(policy, dict) else {}
                conf_policy = policy.get('confidence', {}) if isinstance(policy, dict) else {}
                edge_enabled = bool(edge_policy.get('enabled', False))
                edge_threshold = float(edge_policy.get('threshold', 0.0) or 0.0)
                conf_enabled = bool(conf_policy.get('enabled', False))
                conf_margin_threshold = float(conf_policy.get('margin_threshold', 0.0) or 0.0)
                prob = cal_cfg.get('probability')
                if prob and isinstance(prob.get('x'), list) and isinstance(prob.get('p'), list):
                    prob_map = {'x': [float(v) for v in prob['x']], 'p': [float(v) for v in prob['p']]}
    except Exception:
        pass

    calc = SpreadCalculator(home_field_advantage=hfa)
    results = calc.calculate_week_spreads(matchups, power, week)

    def calibrated_value(raw: float) -> float:
        if not enable_calibration:
            return raw
        cal = a + b * raw
        lo, hi = blend['low'], blend['high']
        mag = abs(raw)
        if mag <= lo:
            return raw
        if mag >= hi:
            return cal
        t = (mag - lo) / (hi - lo)
        return (1 - t) * raw + t * cal

    def prob_home_from_margin(m: float) -> float:
        if use_calibrated_probability and prob_map:
            xs = prob_map['x']
            ps = prob_map['p']
            if m <= xs[0]:
                return ps[0]
            if m >= xs[-1]:
                return ps[-1]
            import bisect
            i = bisect.bisect_left(xs, m)
            x0,x1 = xs[i-1], xs[i]
            p0,p1 = ps[i-1], ps[i]
            t = (m - x0)/(x1 - x0) if x1!=x0 else 0.0
            return p0 + t*(p1 - p0)
        # fallback sigmoid
        import math
        return 1/(1+math.exp(-m/3.5))

    # Write a CSV
    spreads_csv = os.path.join(output_dir, f'spreads_{lg}_week_{week}.csv')
    with open(spreads_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['week', 'home_team', 'away_team', 'projected_spread_raw', 'projected_spread', 'home_win_prob', 'betting_line', 'market_spread', 'market_line', 'edge', 'actionable', 'game_date'])
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
                        edge = calibrated_value(r.projected_spread) - mk
            # Determine if actionable based on policy
            actionable = True
            if odds_map and edge_enabled and edge is not None:
                actionable = abs(edge) >= edge_threshold
            elif not odds_map and conf_enabled:
                actionable = abs(calibrated_value(r.projected_spread)) >= conf_margin_threshold
            writer.writerow([
                r.week, r.home_team, r.away_team, f"{r.projected_spread:.1f}",
                f"{calibrated_value(r.projected_spread):.1f}", f"{prob_home_from_margin(calibrated_value(r.projected_spread)):.3f}",
                calc.format_spread_as_betting_line(calibrated_value(r.projected_spread), r.home_team),
                (f"{mk:.1f}" if isinstance(mk, (int, float)) else ''),
                (ml or ''),
                (f"{edge:+.1f}" if isinstance(edge, (int, float)) else ''),
                ('Y' if actionable else ''),
                r.game_date
            ])

    # Collect results list
    results_list = [
        {
            'week': r.week,
            'home_team': r.home_team,
            'away_team': r.away_team,
            'projected_spread_raw': r.projected_spread,
            'projected_spread': calibrated_value(r.projected_spread),
            'home_win_prob': prob_home_from_margin(calibrated_value(r.projected_spread)),
            'betting_line': calc.format_spread_as_betting_line(calibrated_value(r.projected_spread), r.home_team),
            'market_spread': (odds_map or {}).get((r.home_team, r.away_team), {}).get('market_spread'),
            'market_line': (odds_map or {}).get((r.home_team, r.away_team), {}).get('market_line'),
            'edge': (calibrated_value(r.projected_spread) - (odds_map or {}).get((r.home_team, r.away_team), {}).get('market_spread')
                     if (odds_map or {}).get((r.home_team, r.away_team), {}).get('market_spread') is not None else None),
            'actionable': (abs((calibrated_value(r.projected_spread) - (odds_map or {}).get((r.home_team, r.away_team), {}).get('market_spread'))) >= edge_threshold
                           if (odds_map and edge_enabled and (odds_map or {}).get((r.home_team, r.away_team), {}).get('market_spread') is not None)
                           else (abs(calibrated_value(r.projected_spread)) >= conf_margin_threshold if (not odds_map and conf_enabled) else True)),
            'game_date': r.game_date,
        }
        for r in results
    ]

    return spreads_csv, results_list


def fetch_schedule_from_espn(week: int,
                             season: int,
                             output_dir: str,
                             league: str) -> Tuple[str, Dict[Tuple[str, str], Dict[str, Any]]]:
    """Fetch Week schedule from ESPN and write a normalized schedule CSV.

    Returns path to the CSV with columns: week,home_team,away_team,game_date
    Teams are written as abbreviations.
    """
    from power_ranking.power_ranking.api.client_factory import get_client

    client = get_client('sync', league=league)
    data = client.get_scoreboard(week=week, season=season)
    teams = client.get_teams()

    lg = (league or 'nfl').lower()
    abbr_map = build_team_abbr_map(lg, teams)
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
        if lg == 'ncaa':
            home_label = home_name
            away_label = away_name
        else:
            home_label = home_abbr
            away_label = away_abbr
        game_date = (event.get('date') or '').split('T')[0]
        rows.append((week, home_label, away_label, game_date))

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
                    if lg == 'ncaa':
                        from ncaa_model.spread_model import SpreadCalculator as LeagueSpreadCalculator
                        default_hfa = 3.0
                    else:
                        from nfl_model.spread_model import SpreadCalculator as LeagueSpreadCalculator
                        default_hfa = 2.0
                    market_line = LeagueSpreadCalculator(home_field_advantage=default_hfa).format_spread_as_betting_line(market_spread, home_abbr)
                except Exception:
                    # Manual format
                    if market_spread > 0:
                        market_line = f"{home_abbr} -{abs(market_spread):.1f}"
                    elif market_spread < 0:
                        market_line = f"{home_abbr} +{abs(market_spread):.1f}"
                    else:
                        market_line = f"{home_abbr} PK"
        odds_map[(home_label, away_label)] = {
            'market_spread': market_spread,
            'market_line': market_line,
        }

    if not rows:
        raise RuntimeError(f"No games found from ESPN for season {season} week {week}")

    out = os.path.join(output_dir, f'schedule_{lg}_week_{week}_{season}.csv')
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
                      last_n: int,
                      league: str) -> str:
    path = os.path.join(output_dir, f'summary_{league.lower()}_week_{week}_last{last_n}.csv')
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
                       last_n: int,
                       league: str,
                       params_version: Any | None = None,
                       params_hash: str | None = None) -> str:
    lg = (league or 'nfl').lower()
    league_label = 'NCAA' if lg == 'ncaa' else 'NFL'
    path = os.path.join(output_dir, f'summary_{lg}_week_{week}_last{last_n}.html')
    ts = datetime.now(timezone.utc).isoformat()
    html = [
        "<!DOCTYPE html>",
        f"<html><head><meta charset='utf-8'><title>{league_label} Projects Summary</title>",
        "<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:8px} th{background:#f3f3f3}</style>",
        "</head><body>",
        f"<h1>{league_label} Projects Summary - Week {week} (Last {last_n} games)</h1>",
        f"<p>Generated: {ts} UTC</p>",
        (f"<p>Calibration Params — version: {params_version}, hash: {params_hash}</p>" if (params_version is not None or params_hash is not None) else ""),
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
    html.append("<table><tr><th>Matchup</th><th>Projected</th><th>Betting Line</th><th>Market</th><th>Edge</th><th>Actionable</th><th>Date</th></tr>")
    for row in spreads:
        matchup = f"{row['away_team']} @ {row['home_team']}"
        market_line = row.get('market_line') or ''
        edge = row.get('edge')
        edge_str = f"{edge:+.1f}" if isinstance(edge, (int, float)) else ''
        actionable = 'Y' if row.get('actionable') else ''
        html.append(
            f"<tr><td>{matchup}</td><td>{row['projected_spread']:+.1f}</td><td>{row['betting_line']}</td><td>{market_line}</td><td>{edge_str}</td><td>{actionable}</td><td>{row['game_date']}</td></tr>"
        )
    html.append("</table>")
    html.append("</body></html>")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html))
    return path


def main():
    parser = argparse.ArgumentParser(description="Run Power Rankings then spread model for the chosen league")
    parser.add_argument('--league', choices=['nfl', 'ncaa'], default='nfl', help='League to run (default: nfl)')
    parser.add_argument('--week', type=int, help='Week number to generate outputs for (default: autodetect)')
    parser.add_argument('--last-n', type=int, default=17, help='Most recent games per team (default: 17)')
    parser.add_argument('--schedule', type=str, default='auto', help='Schedule CSV path or "auto" to fetch from ESPN')
    parser.add_argument('--season', type=int, default=datetime.now().year, help='Season year for schedule when using auto')
    parser.add_argument('--output', type=str, default='./output', help='Output directory for artifacts')
    args = parser.parse_args()

    league = args.league.lower()
    league_label = 'NCAA' if league == 'ncaa' else 'NFL'

    output_dir = ensure_abs(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # 1) Power rankings
    pr_csv, rankings, computation, teams = run_power_rankings(args.week, args.last_n, output_dir, league)
    print(f"\n=== {league_label} Power Rankings (All Teams) ===")
    for i, (_, team_name, score) in enumerate(rankings, 1):
        print(f"{i:2d}. {team_name:<25} {score:6.3f}")
    print(f"Saved power rankings CSV: {pr_csv}")

    # 2) Prepare power rankings for spread model (abbreviations for NFL, names for NCAA)
    pr_processed_csv = prepare_power_csv(pr_csv, output_dir, league, teams, computation.get('teams_map'))
    # Determine target week for NFL model schedule use
    target_week = args.week or 1
    # Resolve schedule
    if args.schedule == 'auto':
        try:
            schedule_csv, odds_map = fetch_schedule_from_espn(week=target_week, season=args.season, output_dir=output_dir, league=league)
            print(f"Fetched {league_label} schedule from ESPN: {schedule_csv}")
        except Exception as e:
            raise RuntimeError(f"Failed to auto-fetch schedule: {e}")
    else:
        schedule_csv = ensure_abs(args.schedule)
        if not os.path.exists(schedule_csv):
            raise FileNotFoundError(f"Schedule CSV not found: {schedule_csv}")
        odds_map = None

    spreads_csv, spreads = run_spread_model(pr_processed_csv, schedule_csv, target_week, output_dir, league, odds_map)
    print(f"\n=== {league_label} Spread Predictions ===")
    # Determine filter for printing (read policy locally)
    print_filter = None
    try:
        cal_path = os.path.join(os.getcwd(), 'calibration', 'ncaa_params.yaml' if league == 'ncaa' else 'params.yaml')
        if os.path.exists(cal_path):
            with open(cal_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            pol = (cfg.get('calibration') or {}).get('policy') or {}
            edge_pol = pol.get('edge') or {}
            conf_pol = pol.get('confidence') or {}
            _edge_enabled = bool(edge_pol.get('enabled', False))
            _edge_threshold = float(edge_pol.get('threshold', 0.0) or 0.0)
            _conf_enabled = bool(conf_pol.get('enabled', False))
            _conf_thr = float(conf_pol.get('margin_threshold', 0.0) or 0.0)
            if odds_map and _edge_enabled:
                print_filter = lambda row: (isinstance(row.get('edge'), (int, float)) and abs(row['edge']) >= _edge_threshold)
                print(f"Applied edge filter: abs(edge) >= {_edge_threshold:.1f}")
            elif (not odds_map) and _conf_enabled:
                print_filter = lambda row: abs(row['projected_spread']) >= _conf_thr
                print(f"Applied confidence filter: abs(projected) >= {_conf_thr:.1f}")
    except Exception:
        pass

    for row in spreads:
        if print_filter and not print_filter(row):
            continue
        market = row.get('market_spread')
        edge = row.get('edge')
        suffix = ''
        if market is not None:
            ml = row.get('market_line') or ''
            suffix = f" | Market: {ml} | Edge: {edge:+.1f}"
        tag = " [EDGE]" if row.get('actionable') else ''
        print(f"{row['away_team']} @ {row['home_team']}: {row['projected_spread']:+.1f} ({row['betting_line']}){suffix}{tag}")
    print(f"Saved spreads CSV: {spreads_csv}")

    # 3) Combined summary CSV and HTML
    summary_csv = write_summary_csv(output_dir, rankings, computation, spreads, target_week, args.last_n, league)
    # Compute params info for logging and HTML if available
    calib_path = os.path.join(os.getcwd(), 'calibration', 'ncaa_params.yaml' if league == 'ncaa' else 'params.yaml')
    params_version = None
    params_hash = None
    try:
        if os.path.exists(calib_path):
            with open(calib_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
                params_version = cfg.get('version')
            with open(calib_path, 'rb') as fb:
                params_hash = hashlib.sha256(fb.read()).hexdigest()[:12]
    except Exception:
        pass
    summary_html = write_summary_html(output_dir, rankings, computation, spreads, target_week, args.last_n, league,
                                      params_version=params_version, params_hash=params_hash)
    print("\n=== Summary Artifacts ===")
    print(f"Summary CSV:  {summary_csv}")
    print(f"Summary HTML: {summary_html}")
    if params_version is not None or params_hash is not None:
        print(f"Params: version={params_version if params_version is not None else 'n/a'}, hash={params_hash if params_hash else 'n/a'}")


if __name__ == '__main__':
    sys.exit(main())

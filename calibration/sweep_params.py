#!/usr/bin/env python3
"""
Hyperparameter Tuning: last-N window and model weights sweep.

Evaluates multiple (last_n, weights) configurations over one or more seasons
using the winners-style backtest (no market dependency). Reports accuracy,
coverage vs our predicted spread, and MAE vs actual margin. Can optionally
apply margin calibration (a + b * projected) for MAE evaluation.

Usage examples:
  # Sweep defaults over 2021-2024 with calibrated HFA from params.yaml
  python -m calibration.sweep_params --seasons 2021-2024 --use-calibration --output ./backtests

  # Custom last-N grid and include a heavier recency weight set
  python -m calibration.sweep_params --seasons 2022 2023 2024 \
    --last-n-grid 8 12 14 17 \
    --weights-set recency_heavy \
    --use-calibration --output ./backtests
"""

import argparse
import os
import sys
import yaml
import csv
from typing import Dict, List, Any, Tuple
from datetime import datetime, timezone

from power_ranking.power_ranking.api.espn_client import ESPNClient
from power_ranking.power_ranking.models.power_rankings import PowerRankModel


def ensure_dir(path: str) -> str:
    p = os.path.abspath(os.path.expanduser(path))
    os.makedirs(p, exist_ok=True)
    return p


def get_season_events(client: ESPNClient, season: int) -> List[Dict[str, Any]]:
    return client.get_season_final_rankings(season).get('events', []) or []


def get_prev_season_events(client: ESPNClient, season: int) -> List[Dict[str, Any]]:
    try:
        return client.get_season_final_rankings(season - 1).get('events', []) or []
    except Exception:
        return client.get_last_season_final_rankings().get('events', []) or []


def extract_games(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def build_train_events(prev_events: List[Dict[str, Any]], curr_events: List[Dict[str, Any]], upto_week_exclusive: int) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    events.extend([e for e in prev_events if e.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL'])
    for e in curr_events:
        wk = e.get('week_number') or e.get('week', {}).get('number')
        if not wk or wk >= upto_week_exclusive:
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


def parse_seasons_arg(values: List[str]) -> List[int]:
    acc: List[int] = []
    for v in values:
        if '-' in v:
            a, b = v.split('-', 1)
            acc.extend(range(int(a), int(b) + 1))
        else:
            acc.append(int(v))
    return sorted(set(acc))


def load_calibration() -> Tuple[float, float, float]:
    a, b, hfa = 0.0, 1.0, 2.0
    try:
        with open('calibration/params.yaml', 'r') as f:
            cfg = yaml.safe_load(f) or {}
            a = float(cfg.get('calibration', {}).get('margin', {}).get('a', 0.0))
            b = float(cfg.get('calibration', {}).get('margin', {}).get('b', 1.0))
            hfa = float(cfg.get('model', {}).get('hfa', 2.0))
    except Exception:
        pass
    return a, b, hfa


def get_weight_sets(selected: List[str]) -> Dict[str, Dict[str, float]]:
    # Baseline + a few curated options (sum to ~1.0)
    presets = {
        'baseline': {'season_avg_margin': 0.50, 'rolling_avg_margin': 0.25, 'sos': 0.20, 'recency_factor': 0.05},
        'recency_heavy': {'season_avg_margin': 0.40, 'rolling_avg_margin': 0.35, 'sos': 0.20, 'recency_factor': 0.05},
        'season_heavy': {'season_avg_margin': 0.55, 'rolling_avg_margin': 0.20, 'sos': 0.20, 'recency_factor': 0.05},
        'sos_light': {'season_avg_margin': 0.50, 'rolling_avg_margin': 0.30, 'sos': 0.15, 'recency_factor': 0.05},
        'recency_plus': {'season_avg_margin': 0.50, 'rolling_avg_margin': 0.25, 'sos': 0.15, 'recency_factor': 0.10},
    }
    if not selected:
        keys = ['baseline', 'recency_heavy', 'season_heavy', 'sos_light', 'recency_plus']
        return {k: presets[k] for k in keys}
    out = {}
    for k in selected:
        if k not in presets:
            raise SystemExit(f"Unknown weights-set: {k}. Available: {', '.join(presets.keys())}")
        out[k] = presets[k]
    return out


def eval_config(season: int, last_n: int, weights: Dict[str, float], use_calib: bool, hfa: float, a: float, b: float) -> Dict[str, Any]:
    client = ESPNClient()
    teams = client.get_teams()
    prev_events = get_prev_season_events(client, season)
    curr_events = get_season_events(client, season)
    # Partition
    weeks: Dict[int, List[Dict[str, Any]]] = {}
    for e in curr_events:
        wk = e.get('week_number') or e.get('week', {}).get('number')
        if isinstance(wk, int):
            weeks.setdefault(wk, []).append(e)
    # Metrics
    total = correct = pushes = 0
    covered_y = covered_n = covered_p = 0
    abs_err = []

    model = PowerRankModel(weights=weights)

    for week in range(1, 19):
        games = extract_games(weeks.get(week, []))
        if not games:
            continue
        train_events = build_train_events(prev_events, curr_events, week)
        sb_like = {'events': train_events, 'week': {'number': week}, 'season': {'year': season}}
        rankings, comp = model.compute(sb_like, teams, last_n_games=last_n)
        powers: Dict[str, float] = comp.get('power_scores', {})
        for g in games:
            if g['status'] != 'STATUS_FINAL':
                continue
            home_id, away_id = g['home_id'], g['away_id']
            if home_id not in powers or away_id not in powers:
                continue
            proj_raw = (powers[home_id] - powers[away_id]) + hfa
            proj = (a + b * proj_raw) if use_calib else proj_raw
            actual = g['home_score'] - g['away_score']
            # Winner accuracy
            pred_team = g['home_name'] if proj > 0 else g['away_name'] if proj < 0 else 'TIE'
            actual_team = g['home_name'] if actual > 0 else g['away_name'] if actual < 0 else 'TIE'
            if pred_team == 'TIE' or actual_team == 'TIE':
                pushes += 1
            else:
                if pred_team == actual_team:
                    correct += 1
            total += 1
            # Coverage vs our projected margin
            if proj > 0:
                cov = actual >= proj
            elif proj < 0:
                cov = actual <= proj
            else:
                cov = None
            if cov is None:
                covered_p += 1
            elif cov:
                covered_y += 1
            else:
                covered_n += 1
            abs_err.append(abs(proj - actual))

    wins = correct
    losses = max(total - pushes - wins, 0)
    acc = (wins / max(total - pushes, 1)) if total else 0.0
    cov_total = covered_y + covered_n
    cov_rate = (covered_y / max(cov_total, 1)) if cov_total else 0.0
    mae = (sum(abs_err) / len(abs_err)) if abs_err else None
    return {
        'season': season,
        'last_n': last_n,
        'weights': weights,
        'games': total,
        'pushes': pushes,
        'wins': wins,
        'losses': losses,
        'accuracy': acc,
        'covered_yes': covered_y,
        'covered_no': covered_n,
        'covered_push': covered_p,
        'cover_rate': cov_rate,
        'mae': mae,
    }


def main():
    ap = argparse.ArgumentParser(description='Sweep last-N and weights over seasons')
    ap.add_argument('--seasons', nargs='+', required=True, help='Seasons list or ranges (e.g., 2021-2024)')
    ap.add_argument('--last-n-grid', nargs='*', type=int, default=[8, 12, 14, 17], help='Grid of last-N values')
    ap.add_argument('--weights-set', nargs='*', default=[], help='Which weight presets to include (default: all presets)')
    ap.add_argument('--use-calibration', action='store_true', help='Apply calibration a,b and hfa from calibration/params.yaml')
    ap.add_argument('--hfa', type=float, default=None, help='Override HFA (if not using calibration)')
    ap.add_argument('--output', type=str, default='./backtests', help='Output directory')
    args = ap.parse_args()

    out_dir = ensure_dir(args.output)
    seasons = []
    for token in args.seasons:
        seasons.extend(parse_seasons_arg([token]))
    seasons = sorted(set(seasons))

    a, b, hfa_cal = load_calibration()
    use_calib = args.use_calibration
    hfa = hfa_cal if use_calib else (args.hfa if args.hfa is not None else 2.0)
    weight_sets = get_weight_sets(args.weights_set)

    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(out_dir, f'sweep_params_{ts}.csv')
    out_html = os.path.join(out_dir, f'sweep_params_{ts}.html')

    rows = []
    for season in seasons:
        for last_n in args.last_n_grid:
            for name, weights in weight_sets.items():
                res = eval_config(season, last_n, weights, use_calib, hfa, a, b)
                res['weights_name'] = name
                rows.append(res)

    # Write CSV
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['season','last_n','weights_name','accuracy','cover_rate','mae','games','pushes','wins','losses','weights'])
        for r in rows:
            w.writerow([
                r['season'], r['last_n'], r['weights_name'], f"{r['accuracy']:.3f}", f"{r['cover_rate']:.3f}", f"{r['mae']:.2f}" if r['mae'] is not None else '',
                r['games'], r['pushes'], r['wins'], r['losses'], r['weights']
            ])

    # Write HTML (simple sortable-like table with basic filters)
    with open(out_html, 'w', encoding='utf-8') as f:
        seasons_opts = "".join(f"<option value='{s}'>{s}</option>" for s in seasons)
        lastn_opts = "".join(f"<option value='{n}'>{n}</option>" for n in args.last_n_grid)
        weight_opts = "".join(f"<option value='{n}'>{n}</option>" for n in weight_sets.keys())
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Hyperparameter Sweep</title>"
                "<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3}</style>"
                "</head><body>")
        f.write("<h1>Hyperparameter Sweep Results</h1>")
        f.write(f"<p>Use calibration: {use_calib} | HFA: {hfa:.2f} | a,b: ({a:.3f},{b:.3f})</p>")
        f.write("<div style='margin:8px 0;'>"
                f"<label>Season: <select id='fSeason'><option value=''>All</option>{seasons_opts}</select></label> "
                f"<label>last_n: <select id='fLastN'><option value=''>All</option>{lastn_opts}</select></label> "
                f"<label>Weights: <select id='fWeights'><option value=''>All</option>{weight_opts}</select></label> "
                " <button onclick='filterRows()'>Filter</button> <button onclick='resetFilters()'>Reset</button>"
                "</div>")
        f.write("<script>function filterRows(){var s=document.getElementById('fSeason').value;var n=document.getElementById('fLastN').value;var w=document.getElementById('fWeights').value;var rows=document.querySelectorAll('#grid tr');rows.forEach(function(r){var show=true; if(s && r.dataset.season!==s) show=false; if(n && r.dataset.lastn!==n) show=false; if(w && r.dataset.weights!==w) show=false; r.style.display=show?'':'none';});}\n"
                "function resetFilters(){document.getElementById('fSeason').value='';document.getElementById('fLastN').value='';document.getElementById('fWeights').value='';filterRows();}</script>")
        f.write("<table><tr><th>Season</th><th>last_n</th><th>Weights</th><th>Accuracy</th><th>Cover Rate</th><th>MAE</th><th>Games</th><th>Pushes</th><th>Wins</th><th>Losses</th></tr><tbody id='grid'>")
        for r in rows:
            f.write(f"<tr data-season='{r['season']}' data-lastn='{r['last_n']}' data-weights='{r['weights_name']}'>" +
                    f"<td>{r['season']}</td>" +
                    f"<td>{r['last_n']}</td>" +
                    f"<td>{r['weights_name']}</td>" +
                    f"<td>{r['accuracy']:.3f}</td>" +
                    f"<td>{r['cover_rate']:.3f}</td>" +
                    f"<td>{r['mae']:.2f}</td>" +
                    f"<td>{r['games']}</td>" +
                    f"<td>{r['pushes']}</td>" +
                    f"<td>{r['wins']}</td>" +
                    f"<td>{r['losses']}</td>" +
                    "</tr>")
        f.write("</tbody></table></body></html>")

    print("Sweep complete")
    print(f"Summary CSV:  {out_csv}")
    print(f"Summary HTML: {out_html}")


if __name__ == '__main__':
    sys.exit(main())


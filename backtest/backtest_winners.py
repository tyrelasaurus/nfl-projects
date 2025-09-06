#!/usr/bin/env python3
"""
Backtesting winners (no market odds): Evaluates the model's ability to predict
game winners using power rankings-derived projected margins.

Methodology:
- For season S (default: last completed):
  - For each week w=1..18:
    - Train on: season S-1 full + season S weeks < w (STATUS_FINAL only).
    - Compute power scores with last-N games per team across seasons.
    - For each game in week w (STATUS_FINAL):
      - Project margin = (home_power - away_power) + HFA.
      - Predict winner (home if margin>0, away if margin<0, tie if 0).
      - Compare to actual winner (home_score - away_score).
- Outputs:
  - Per-game CSV with predicted vs actual and correctness.
  - Summary CSV and HTML with accuracy.

Usage:
  python -m backtest.backtest_winners --season 2024 --last-n 17 --hfa 2.0 --output ./backtests
"""

import argparse
import os
import sys
import csv
from typing import Dict, List, Tuple, Any
from datetime import datetime, timezone

from power_ranking.power_ranking.api.client_factory import get_client
from power_ranking.power_ranking.models.power_rankings import PowerRankModel
import yaml


def ensure_dir(path: str) -> str:
    p = os.path.abspath(os.path.expanduser(path))
    os.makedirs(p, exist_ok=True)
    return p


def get_season_events(client, season: int) -> List[Dict[str, Any]]:
    data = client.get_season_final_rankings(season)
    return data.get('events', []) or []


def get_prev_season_events(client, season: int) -> List[Dict[str, Any]]:
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


def build_training_events(prev_events: List[Dict[str, Any]],
                          curr_events: List[Dict[str, Any]],
                          up_to_week_exclusive: int) -> List[Dict[str, Any]]:
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


def main():
    parser = argparse.ArgumentParser(description='Backtest model winner predictions (no market odds)')
    parser.add_argument('--season', type=int, help='Season year (default: last completed)')
    parser.add_argument('--last-n', type=int, default=17, help='Last N games per team for power (default: 17)')
    parser.add_argument('--hfa', type=float, default=2.0, help='Home field advantage (default: 2.0)')
    parser.add_argument('--output', type=str, default='./backtests', help='Output directory')
    args = parser.parse_args()

    client = get_client('sync')
    if not args.season:
        args.season = client.get_last_completed_season()

    out_dir = ensure_dir(args.output)
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    per_game_csv = os.path.join(out_dir, f'backtest_winners_{args.season}_{ts}.csv')
    summary_csv = os.path.join(out_dir, f'backtest_winners_summary_{args.season}_{ts}.csv')
    summary_html = os.path.join(out_dir, f'backtest_winners_summary_{args.season}_{ts}.html')

    teams = client.get_teams()
    curr_events = get_season_events(client, args.season)
    prev_events = get_prev_season_events(client, args.season)

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
    covered_yes = 0
    covered_no = 0
    covered_push = 0

    # Collect all game rows for HTML
    all_rows: List[Dict[str, Any]] = []

    # Load margin calibration if present
    a, b = 0.0, 1.0
    blend = {'low': 3.0, 'high': 7.0}
    try:
        with open('calibration/params.yaml', 'r') as _pf:
            cfg = yaml.safe_load(_pf) or {}
            a = float(cfg.get('calibration', {}).get('margin', {}).get('a', 0.0))
            b = float(cfg.get('calibration', {}).get('margin', {}).get('b', 1.0))
            bl = cfg.get('calibration', {}).get('blend', {})
            blend['low'] = float(bl.get('low', 3.0))
            blend['high'] = float(bl.get('high', 7.0))
    except Exception:
        pass

    with open(per_game_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'season', 'week', 'date', 'home', 'away', 'projected_margin_raw', 'projected_margin_cal', 'actual_margin', 'abs_error_raw', 'abs_error_cal', 'away_score', 'home_score', 'final_score', 'covered_predicted',
            'predicted_winner_side', 'predicted_winner_team', 'actual_winner_side', 'actual_winner_team', 'correct'
        ])

        for week in range(1, 19):
            wevents = weeks.get(week, [])
            games = extract_games_from_events(wevents)
            if not games:
                continue

            train_events = build_training_events(prev_events, curr_events, week)
            sb_like = {'events': train_events, 'week': {'number': week}, 'season': {'year': args.season}}
            rankings, comp = model.compute(sb_like, teams, last_n_games=args.last_n)
            powers: Dict[str, float] = comp.get('power_scores', {})

            for g in games:
                if g['status'] != 'STATUS_FINAL':
                    continue
                home_id, away_id = g['home_id'], g['away_id']
                if home_id not in powers or away_id not in powers:
                    continue
                projected_raw = (powers[home_id] - powers[away_id]) + args.hfa
                lo, hi = blend['low'], blend['high']
                cal_lin = a + b * projected_raw
                mag = abs(projected_raw)
                if mag <= lo:
                    projected_cal = projected_raw
                elif mag >= hi:
                    projected_cal = cal_lin
                else:
                    t = (mag - lo) / (hi - lo)
                    projected_cal = (1 - t) * projected_raw + t * cal_lin
                actual_margin = g['home_score'] - g['away_score']
                predicted_winner = 'home' if projected_raw > 0 else 'away' if projected_raw < 0 else 'push'
                actual_winner = 'home' if actual_margin > 0 else 'away' if actual_margin < 0 else 'push'
                predicted_team = g['home_name'] if predicted_winner == 'home' else g['away_name'] if predicted_winner == 'away' else 'TIE'
                actual_team = g['home_name'] if actual_winner == 'home' else g['away_name'] if actual_winner == 'away' else 'TIE'

                # Coverage of our predicted spread (use calibrated for fairness)
                if projected_cal > 0:
                    covered = 'Yes' if actual_margin >= projected_cal else 'No'
                elif projected_cal < 0:
                    covered = 'Yes' if actual_margin <= projected_cal else 'No'
                else:
                    covered = 'Push'
                if covered == 'Yes':
                    covered_yes += 1
                elif covered == 'No':
                    covered_no += 1
                else:
                    covered_push += 1

                if predicted_winner == 'push' or actual_winner == 'push':
                    pushes += 1
                    is_correct = ''
                else:
                    is_correct = '1' if predicted_winner == actual_winner else '0'
                    if is_correct == '1':
                        correct += 1
                total += 1
                w.writerow([
                    args.season, week, g['date'], g['home_abbr'], g['away_abbr'], f"{projected_raw:+.1f}", f"{projected_cal:+.1f}", f"{actual_margin:+.1f}", f"{abs(projected_raw-actual_margin):.1f}", f"{abs(projected_cal-actual_margin):.1f}", g['away_score'], g['home_score'], f"{g['away_score']}-{g['home_score']}", covered,
                    predicted_winner, predicted_team, actual_winner, actual_team, is_correct
                ])
                all_rows.append({
                    'season': args.season,
                    'week': week,
                    'date': g['date'],
                    'home': g['home_abbr'],
                    'away': g['away_abbr'],
                    'home_name': g['home_name'],
                    'away_name': g['away_name'],
                    'home_score': g['home_score'],
                    'away_score': g['away_score'],
                    'projected_margin': projected_raw,
                    'projected_margin_cal': projected_cal,
                    'actual_margin': actual_margin,
                    'abs_error_raw': abs(projected_raw-actual_margin),
                    'abs_error_cal': abs(projected_cal-actual_margin),
                    'covered': covered,
                    'predicted_team': predicted_team,
                    'actual_team': actual_team,
                    'correct': is_correct == '1'
                })

    # Summary CSV
    with open(summary_csv, 'w', newline='') as f:
        w = csv.writer(f)
        acc = (correct / max(total - pushes, 1)) if total else 0.0
        cov_total = covered_yes + covered_no
        cov_rate = (covered_yes / max(cov_total, 1)) if cov_total else 0.0
        w.writerow(['season', 'games', 'pushes', 'wins', 'losses', 'accuracy', 'covered_yes', 'covered_no', 'covered_push', 'cover_rate'])
        w.writerow([args.season, total, pushes, correct, max(total - pushes - correct, 0), f"{acc:.3f}", covered_yes, covered_no, covered_push, f"{cov_rate:.3f}"])

    # Summary HTML (with full per-game table)
    with open(summary_html, 'w', encoding='utf-8') as f:
        acc = (correct / max(total - pushes, 1)) if total else 0.0
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Backtest Winners Summary</title>"
                "<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3}</style>"
                "</head><body>")
        f.write(f"<h1>Backtest Winners Summary - Season {args.season}</h1>")
        f.write("<table><tr><th>Games</th><th>Pushes</th><th>Wins</th><th>Losses</th><th>Accuracy</th><th>Covered Yes</th><th>Covered No</th><th>Covered Push</th><th>Cover Rate</th></tr>")
        cov_total = covered_yes + covered_no
        cov_rate = (covered_yes / max(cov_total, 1)) if cov_total else 0.0
        f.write(f"<tr><td>{total}</td><td>{pushes}</td><td>{correct}</td><td>{max(total - pushes - correct, 0)}</td><td>{acc:.3f}</td><td>{covered_yes}</td><td>{covered_no}</td><td>{covered_push}</td><td>{cov_rate:.3f}</td></tr></table>")
        f.write(f"<p>Per-game CSV: {per_game_csv}<br>Summary CSV: {summary_csv}</p>")
        # Filters
        f.write("<h2>Per-Game Results</h2>")
        f.write("<div style='margin:8px 0;'>"
                "<label>Week: <select id='fWeek'><option value=''>All</option>" +
                "".join(f"<option value='{w}'>{w}</option>" for w in range(1,19)) + "</select></label>\n"
                " <label>Team: <input id='fTeam' placeholder='ABB or Name' /></label>\n"
                " <label>Team Position: <select id='fPos'><option value=''>Any</option><option value='home'>Home</option><option value='away'>Away</option></select></label>\n"
                " <label>Predicted Side: <select id='fPred'><option value=''>Any</option><option value='home'>Home</option><option value='away'>Away</option><option value='push'>Push</option></select></label>\n"
                " <label>Coverage: <select id='fCov'><option value=''>Any</option><option value='covered'>Covered</option><option value='not'>Not Covered</option><option value='push'>Push</option></select></label>\n"
                " <button onclick='filterRows()'>Filter</button> <button onclick='resetFilters()'>Reset</button>"
                "</div>")
        f.write("<script>function norm(x){return (x||'').toLowerCase()}\n"
                "function matchesTeam(home,away,hname,aname,t,pos){if(!t)return true;var hm=home.toLowerCase().includes(t)||hname.includes(t);var am=away.toLowerCase().includes(t)||aname.includes(t);if(pos==='home')return hm; if(pos==='away')return am; return hm||am;}\n"
                "function filterRows(){var w=document.getElementById('fWeek').value;var t=norm(document.getElementById('fTeam').value);var p=document.getElementById('fPred').value;var pos=document.getElementById('fPos').value;var cov=document.getElementById('fCov').value;var rows=document.querySelectorAll('#results tr');rows.forEach(function(r){var show=true; if(w && r.dataset.week!==w){show=false;} var home=r.dataset.home||''; var away=r.dataset.away||''; var pside=r.dataset.pside||''; var hname=(r.dataset.hname||'').toLowerCase(); var aname=(r.dataset.aname||'').toLowerCase(); var covered=(r.dataset.covered||''); if(t && !matchesTeam(home,away,hname,aname,t,pos)){show=false;} if(p && pside!==p){show=false;} if(cov){ if(cov==='covered' && covered!=='yes') show=false; else if(cov==='not' && covered!=='no') show=false; else if(cov==='push' && covered!=='push') show=false;} r.style.display=show?'':'none';});}\n"
                "function resetFilters(){document.getElementById('fWeek').value='';document.getElementById('fTeam').value='';document.getElementById('fPred').value='';document.getElementById('fPos').value='';document.getElementById('fCov').value='';filterRows();}\n"
                "</script>")
        # Weekly MAE (raw vs calibrated)
        from collections import defaultdict
        mae_week = defaultdict(lambda: {'raw': [], 'cal': []})
        for r in all_rows:
            mae_week[r['week']]['raw'].append(r['abs_error_raw'])
            mae_week[r['week']]['cal'].append(r['abs_error_cal'])
        f.write("<h3>Weekly MAE (Raw vs Calibrated)</h3>")
        f.write("<table><tr><th>Week</th><th>MAE Raw</th><th>MAE Cal</th></tr>")
        for wk in sorted(mae_week.keys()):
            raw = mae_week[wk]['raw']
            cal = mae_week[wk]['cal']
            raw_mae = sum(raw)/len(raw) if raw else 0.0
            cal_mae = sum(cal)/len(cal) if cal else 0.0
            f.write(f"<tr><td>{wk}</td><td>{raw_mae:.2f}</td><td>{cal_mae:.2f}</td></tr>")
        f.write("</table>")

        f.write("<table><tr>" \
                "<th>Week</th><th>Date</th><th>Away</th><th>Home</th>" \
                "<th>Projected (Raw)</th><th>Projected (Cal)</th><th>Actual Margin</th><th>AbsErr Raw</th><th>AbsErr Cal</th><th>Final Score</th><th>Covered (Predicted)</th>" \
                "<th>Predicted Winner</th><th>Actual Winner</th><th>Correct</th>" \
                "</tr><tbody id='results'>")
        for row in sorted(all_rows, key=lambda r: (r['week'], r['date'])):
            f.write(f"<tr data-week='{row['week']}' data-home='{row['home']}' data-away='{row['away']}' data-pside='" + ("home" if row['projected_margin']>0 else ("away" if row['projected_margin']<0 else "push")) + f"' data-hname='{row['home_name']}' data-aname='{row['away_name']}' data-covered='{row['covered'].lower()}'>" +
                    f"<td>{row['week']}</td>" +
                    f"<td>{row['date']}</td>" +
                    f"<td>{row['away']}</td>" +
                    f"<td>{row['home']}</td>" +
                    f"<td>{row['projected_margin']:+.1f}</td>" +
                    f"<td>{row['projected_margin_cal']:+.1f}</td>" +
                    f"<td>{row['actual_margin']:+.1f}</td>" +
                    f"<td>{row['abs_error_raw']:.1f}</td>" +
                    f"<td>{row['abs_error_cal']:.1f}</td>" +
                    f"<td>{row['away_score']}-{row['home_score']}</td>" +
                    f"<td>{row['covered']}</td>" +
                    f"<td>{row['predicted_team']}</td>" +
                    f"<td>{row['actual_team']}</td>" +
                    f"<td>{'✅' if row['correct'] else '❌'}</td>" +
                    "</tr>")
        f.write("</tbody></table>")
        f.write("</body></html>")

    print(f"Backtest winners complete for {args.season}")
    print(f"Per-game CSV: {per_game_csv}")
    print(f"Summary CSV:  {summary_csv}")
    print(f"Summary HTML: {summary_html}")


if __name__ == '__main__':
    sys.exit(main())

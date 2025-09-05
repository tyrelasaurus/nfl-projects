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


def run_winners_for_season(season: int, last_n: int, hfa: float, out_dir: str) -> Tuple[Dict[str, Any], str, str]:
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
    covered_yes = 0
    covered_no = 0
    covered_push = 0
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    rows: List[Dict[str, Any]] = []
    per_game_csv = os.path.join(out_dir, f'backtest_winners_{season}_{ts}.csv')
    with open(per_game_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['season','week','date','home','away','projected_margin','actual_margin','away_score','home_score','final_score','covered_predicted','predicted_team','actual_team','correct'])
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
                # Coverage vs our predicted spread
                if projected > 0:
                    covered = 'Yes' if actual_margin >= projected else 'No'
                elif projected < 0:
                    covered = 'Yes' if actual_margin <= projected else 'No'
                else:
                    covered = 'Push'
                if covered == 'Yes':
                    covered_yes += 1
                elif covered == 'No':
                    covered_no += 1
                else:
                    covered_push += 1
                if predicted_team == 'TIE' or actual_team == 'TIE':
                    pushes += 1
                    is_correct = ''
                else:
                    is_correct = '1' if predicted_team == actual_team else '0'
                    if is_correct == '1':
                        correct += 1
                total += 1
                w.writerow([season, week, g['date'], g['home_abbr'], g['away_abbr'], f"{projected:+.1f}", f"{actual_margin:+.1f}", g['away_score'], g['home_score'], f"{g['away_score']}-{g['home_score']}", covered, predicted_team, actual_team, is_correct])
                rows.append({
                    'week': week,
                    'date': g['date'],
                    'home': g['home_abbr'],
                    'away': g['away_abbr'],
                    'home_name': g['home_name'],
                    'away_name': g['away_name'],
                    'projected_margin': projected,
                    'actual_margin': actual_margin,
                    'home_score': g['home_score'],
                    'away_score': g['away_score'],
                    'covered': covered,
                    'predicted_team': predicted_team,
                    'actual_team': actual_team,
                    'correct': (is_correct == '1'),
                })
    # Summary record
    wins = correct
    losses = max(total - pushes - wins, 0)
    acc = (wins / max(total - pushes, 1)) if total else 0.0
    # Build per-season detailed HTML with filters (like single-season)
    per_season_html = os.path.join(out_dir, f'backtest_winners_summary_{season}_{ts}.html')
    acc = (correct / max(total - pushes, 1)) if total else 0.0
    cov_total = covered_yes + covered_no
    cov_rate = (covered_yes / max(cov_total, 1)) if cov_total else 0.0
    with open(per_season_html, 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Backtest Winners Summary</title>"
                "<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3}</style>"
                "</head><body>")
        f.write(f"<h1>Backtest Winners Summary - Season {season}</h1>")
        f.write("<table><tr><th>Games</th><th>Pushes</th><th>Wins</th><th>Losses</th><th>Accuracy</th><th>Covered Yes</th><th>Covered No</th><th>Covered Push</th><th>Cover Rate</th></tr>")
        f.write(f"<tr><td>{total}</td><td>{pushes}</td><td>{correct}</td><td>{max(total - pushes - correct, 0)}</td><td>{acc:.3f}</td><td>{covered_yes}</td><td>{covered_no}</td><td>{covered_push}</td><td>{cov_rate:.3f}</td></tr></table>")
        f.write(f"<p>Per-game CSV: {per_game_csv}</p>")
        # Filters
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
        f.write("<table><tr><th>Week</th><th>Date</th><th>Away</th><th>Home</th><th>Projected Margin (Home)</th><th>Actual Margin</th><th>Final Score</th><th>Covered (Predicted)</th><th>Predicted Winner</th><th>Actual Winner</th><th>Correct</th></tr><tbody id='results'>")
        for r in sorted(rows, key=lambda r: (r['week'], r['date'])):
            f.write(f"<tr data-week='{r['week']}' data-home='{r['home']}' data-away='{r['away']}' data-pside='" + ("home" if r['projected_margin']>0 else ("away" if r['projected_margin']<0 else "push")) + f"' data-hname='{r['home_name']}' data-aname='{r['away_name']}' data-covered='{r['covered'].lower()}'>" +
                    f"<td>{r['week']}</td>" +
                    f"<td>{r['date']}</td>" +
                    f"<td>{r['away']}</td>" +
                    f"<td>{r['home']}</td>" +
                    f"<td>{r['projected_margin']:+.1f}</td>" +
                    f"<td>{r['actual_margin']:+.1f}</td>" +
                    f"<td>{r['away_score']}-{r['home_score']}</td>" +
                    f"<td>{r['covered']}</td>" +
                    f"<td>{r['predicted_team']}</td>" +
                    f"<td>{r['actual_team']}</td>" +
                    f"<td>{'✅' if r['correct'] else '❌'}</td>" +
                    "</tr>")
        f.write("</tbody></table></body></html>")

    return {
        'season': season,
        'games': total,
        'pushes': pushes,
        'wins': wins,
        'losses': losses,
        'accuracy': acc,
        'per_game_csv': per_game_csv,
    }, per_game_csv, per_season_html


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
        rec, _, per_html = run_winners_for_season(season, args.last_n, args.hfa, out_dir)
        rec['per_game_html'] = per_html
        records.append(rec)

    with open(agg_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['season', 'games', 'pushes', 'wins', 'losses', 'accuracy', 'per_game_csv'])
        for r in records:
            w.writerow([r['season'], r['games'], r['pushes'], r['wins'], r['losses'], f"{r['accuracy']:.3f}", r['per_game_csv']])

    # HTML index
    with open(index_html, 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Winners Backtests Index</title>"
                "<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3} .section{margin-top:24px}</style>"
                "</head><body>")
        f.write("<h1>Winners Backtests (Aggregate)</h1>")
        # Per-season summary + links
        f.write("<div class='section'><h2>Per-Season Summary</h2>")
        f.write("<table><tr><th>Season</th><th>Games</th><th>Pushes</th><th>Wins</th><th>Losses</th><th>Accuracy</th><th>CSV</th><th>HTML</th></tr>")
        for r in records:
            f.write("<tr>" +
                    f"<td>{r['season']}</td>" +
                    f"<td>{r['games']}</td>" +
                    f"<td>{r['pushes']}</td>" +
                    f"<td>{r['wins']}</td>" +
                    f"<td>{r['losses']}</td>" +
                    f"<td>{r['accuracy']:.3f}</td>" +
                    f"<td>{r['per_game_csv']}</td>" +
                    f"<td>{r.get('per_game_html','')}</td>" +
                    "</tr>")
        f.write("</table></div>")

        # Aggregate per-game table across all seasons with filters
        f.write("<div class='section'><h2>All Seasons - Per-Game Results</h2>")
        # Filters
        season_opts = "".join(f"<option value='{r['season']}'>{r['season']}</option>" for r in records)
        f.write("<div style='margin:8px 0;'>"
                f"<label>Season: <select id='fSeason'><option value=''>All</option>{season_opts}</select></label>\n"
                " <label>Week: <select id='fWeek'><option value=''>All</option>" + "".join(f"<option value='{w}'>{w}</option>" for w in range(1,19)) + "</select></label>\n"
                " <label>Team: <input id='fTeam' placeholder='ABB or Name' /></label>\n"
                " <label>Team Position: <select id='fPos'><option value=''>Any</option><option value='home'>Home</option><option value='away'>Away</option></select></label>\n"
                " <label>Predicted Side: <select id='fPred'><option value=''>Any</option><option value='home'>Home</option><option value='away'>Away</option><option value='push'>Push</option></select></label>\n"
                " <label>Coverage: <select id='fCov'><option value=''>Any</option><option value='covered'>Covered</option><option value='not'>Not Covered</option><option value='push'>Push</option></select></label>\n"
                " <button onclick='filterRows()'>Filter</button> <button onclick='resetFilters()'>Reset</button>"
                "</div>")
        # JS helpers
        f.write("<script>function norm(x){return (x||'').toLowerCase()}\n"
                "function matchesTeam(home,away,hname,aname,t,pos){if(!t)return true;var hm=home.toLowerCase().includes(t)||hname.includes(t);var am=away.toLowerCase().includes(t)||aname.includes(t);if(pos==='home')return hm; if(pos==='away')return am; return hm||am;}\n"
                "function filterRows(){var s=document.getElementById('fSeason').value;var w=document.getElementById('fWeek').value;var t=norm(document.getElementById('fTeam').value);var p=document.getElementById('fPred').value;var pos=document.getElementById('fPos').value;var cov=document.getElementById('fCov').value;var rows=document.querySelectorAll('#aggResults tr');rows.forEach(function(r){var show=true; if(s && r.dataset.season!==s){show=false;} if(w && r.dataset.week!==w){show=false;} var home=r.dataset.home||''; var away=r.dataset.away||''; var pside=r.dataset.pside||''; var hname=(r.dataset.hname||'').toLowerCase(); var aname=(r.dataset.aname||'').toLowerCase(); var covered=(r.dataset.covered||''); if(t && !matchesTeam(home,away,hname,aname,t,pos)){show=false;} if(p && pside!==p){show=false;} if(cov){ if(cov==='covered' && covered!=='yes') show=false; else if(cov==='not' && covered!=='no') show=false; else if(cov==='push' && covered!=='push') show=false;} r.style.display=show?'':'none';});}\n"
                "function resetFilters(){document.getElementById('fSeason').value='';document.getElementById('fWeek').value='';document.getElementById('fTeam').value='';document.getElementById('fPred').value='';document.getElementById('fPos').value='';document.getElementById('fCov').value='';filterRows();}\n"
                "</script>")
        # Build combined rows
        import csv as _csv
        all_rows = []
        for r in records:
            try:
                with open(r['per_game_csv'], 'r') as _pf:
                    rdr = _csv.DictReader(_pf)
                    for row in rdr:
                        all_rows.append({
                            'season': str(r['season']),
                            'week': row.get('week',''),
                            'date': row.get('date',''),
                            'away': row.get('away',''),
                            'home': row.get('home',''),
                            'projected': row.get('projected_margin') or row.get('projected',''),
                            'actual_margin': row.get('actual_margin',''),
                            'final_score': row.get('final_score',''),
                            'covered': (row.get('covered_predicted','') or '').lower(),
                            'predicted_team': row.get('predicted_team',''),
                            'actual_team': row.get('actual_team',''),
                            # We don’t have full names in CSV; fallback to ABBs
                            'home_name': row.get('home',''),
                            'away_name': row.get('away',''),
                        })
            except Exception:
                continue
        # Table
        f.write("<table><tr><th>Season</th><th>Week</th><th>Date</th><th>Away</th><th>Home</th><th>Projected Margin (Home)</th><th>Actual Margin</th><th>Final Score</th><th>Covered (Predicted)</th><th>Predicted Winner</th><th>Actual Winner</th></tr><tbody id='aggResults'>")
        for r in sorted(all_rows, key=lambda x: (x['season'], x['week'], x['date'])):
            pside = 'home' if (str(r['projected']).startswith('+') or (isinstance(r['projected'], str) and r['projected'] and not r['projected'].startswith('-'))) else 'away' if (isinstance(r['projected'], str) and r['projected'].startswith('-')) else 'push'
            f.write(f"<tr data-season='{r['season']}' data-week='{r['week']}' data-home='{r['home']}' data-away='{r['away']}' data-pside='{pside}' data-hname='{r['home_name']}' data-aname='{r['away_name']}' data-covered='{r['covered']}'>" +
                    f"<td>{r['season']}</td>" +
                    f"<td>{r['week']}</td>" +
                    f"<td>{r['date']}</td>" +
                    f"<td>{r['away']}</td>" +
                    f"<td>{r['home']}</td>" +
                    f"<td>{r['projected']}</td>" +
                    f"<td>{r['actual_margin']}</td>" +
                    f"<td>{r['final_score']}</td>" +
                    f"<td>{r['covered'].capitalize()}</td>" +
                    f"<td>{r['predicted_team']}</td>" +
                    f"<td>{r['actual_team']}</td>" +
                    "</tr>")
        f.write("</tbody></table></div>")
        f.write("</body></html>")

    print("Aggregate winners backtests complete")
    print(f"Seasons: {', '.join(map(str,seasons))}")
    print(f"Aggregate CSV: {agg_csv}")
    print(f"Index HTML:    {index_html}")


if __name__ == '__main__':
    sys.exit(main())

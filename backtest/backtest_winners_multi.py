#!/usr/bin/env python3
"""
Run winners backtests over multiple seasons for the chosen league.

Usage examples:
  python -m backtest.backtest_winners_multi --league nfl --seasons 2021-2024 --last-n 17 --output ./backtests
  python -m backtest.backtest_winners_multi --league ncaa --seasons 2022 2023 --last-n 12 --hfa 3.0
"""

import argparse
import csv
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from power_ranking.power_ranking.api.client_factory import get_client
from power_ranking.power_ranking.models.power_rankings import PowerRankModel

try:  # Allow package or standalone execution
    from .backtest_winners import (  # type: ignore[import-not-found]
        DEFAULT_HFA,
        VALID_LEAGUES,
        build_training_events,
        ensure_dir,
        extract_games_from_events,
        get_prev_season_events,
        get_season_events,
        parse_league,
    )
except ImportError:  # pragma: no cover - standalone
    from backtest_winners import (  # type: ignore[import-not-found]
        DEFAULT_HFA,
        VALID_LEAGUES,
        build_training_events,
        ensure_dir,
        extract_games_from_events,
        get_prev_season_events,
        get_season_events,
        parse_league,
    )


def iter_seasons(arg: Sequence[str]) -> Iterable[int]:
    for token in arg:
        if '-' in token:
            start, end = token.split('-', 1)
            try:
                s, e = int(start), int(end)
            except ValueError:
                continue
            for year in range(s, e + 1):
                yield year
        else:
            try:
                yield int(token)
            except ValueError:
                continue


def run_winners_for_season(league: str, season: int, last_n: int, hfa: float, out_dir: str) -> Tuple[Dict[str, Any], str, str]:
    league = parse_league(league)
    client = get_client('sync', league=league)
    teams = client.get_teams()
    curr_events = get_season_events(client, season)
    prev_events = get_prev_season_events(client, season)

    weeks: Dict[int, List[Dict[str, Any]]] = {}
    for event in curr_events:
        wk = event.get('week_number') or event.get('week', {}).get('number')
        if isinstance(wk, int):
            weeks.setdefault(wk, []).append(event)
    week_numbers = sorted(weeks.keys())

    model = PowerRankModel()
    total = 0
    pushes = 0
    wins = 0
    cover_yes = 0
    cover_no = 0
    cover_push = 0

    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    per_game_csv = os.path.join(out_dir, f'backtest_winners_{league}_{season}_{ts}.csv')
    summary_csv = os.path.join(out_dir, f'backtest_winners_summary_{league}_{season}_{ts}.csv')
    summary_html = os.path.join(out_dir, f'backtest_winners_summary_{league}_{season}_{ts}.html')

    with open(per_game_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['league','season','week','date','home','away','projected_margin','actual_margin','away_score','home_score','final_score','covered','predicted_team','actual_team','correct'])

        for week in week_numbers:
            wevents = weeks.get(week, [])
            games = extract_games_from_events(wevents)
            if not games:
                continue

            train_events = build_training_events(prev_events, curr_events, week)
            sb_like = {'events': train_events, 'week': {'number': week}, 'season': {'year': season}}
            _, comp = model.compute(sb_like, teams, last_n_games=last_n)
            powers: Dict[str, float] = comp.get('power_scores', {})

            for g in games:
                if g['status'] != 'STATUS_FINAL':
                    continue
                home_id = g['home_id']
                away_id = g['away_id']
                if home_id not in powers or away_id not in powers:
                    continue

                projected = (powers[home_id] - powers[away_id]) + hfa
                actual_margin = g['home_score'] - g['away_score']
                predicted_team = g['home_name'] if projected > 0 else g['away_name'] if projected < 0 else 'TIE'
                actual_team = g['home_name'] if actual_margin > 0 else g['away_name'] if actual_margin < 0 else 'TIE'
                is_push = abs(actual_margin) < 1e-9
                is_correct = (predicted_team == actual_team) and not is_push

                if projected > 0:
                    covered = 'Yes' if actual_margin >= projected else 'No'
                elif projected < 0:
                    covered = 'Yes' if actual_margin <= projected else 'No'
                else:
                    covered = 'Push'

                if covered == 'Yes':
                    cover_yes += 1
                elif covered == 'No':
                    cover_no += 1
                else:
                    cover_push += 1

                if is_push:
                    pushes += 1
                else:
                    total += 1
                    if is_correct:
                        wins += 1

                writer.writerow([
                    league, season, week, g['date'], g['home_abbr'], g['away_abbr'],
                    projected, actual_margin, g['away_score'], g['home_score'],
                    f"{g['away_score']} @ {g['home_score']}", covered, predicted_team, actual_team,
                    1 if is_correct else 0
                ])

    losses = max(total - wins, 0)
    accuracy = wins / max(total, 1) if total else 0.0
    cover_rate = cover_yes / max(cover_yes + cover_no, 1) if (cover_yes + cover_no) else 0.0

    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['league','season','games','pushes','wins','losses','accuracy','cover_yes','cover_no','cover_push','cover_rate'])
        writer.writerow([league, season, total + pushes, pushes, wins, losses, accuracy, cover_yes, cover_no, cover_push, cover_rate])

    summary_record = {
        'league': league,
        'season': season,
        'games': total + pushes,
        'pushes': pushes,
        'wins': wins,
        'losses': losses,
        'accuracy': accuracy,
        'cover_yes': cover_yes,
        'cover_no': cover_no,
        'cover_push': cover_push,
        'cover_rate': cover_rate,
        'per_game_csv': per_game_csv,
        'summary_csv': summary_csv,
        'summary_html': summary_html,
        'week_numbers': week_numbers,
    }
    render_html_summary(summary_record, per_game_csv, summary_csv, summary_html)
    return summary_record, per_game_csv, summary_csv, summary_html


def render_html_summary(summary: Dict[str, Any], per_game_csv: str, summary_csv: str, html_path: str) -> None:
    import csv

    league = summary['league'].upper()
    season = summary['season']
    week_numbers = summary.get('week_numbers', [])
    with open(summary_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        summary_row = next(iter(reader), {})

    rows = []
    with open(per_game_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    week_options = ''.join(f"<option value='{w}'>{w}</option>" for w in week_numbers)
    cover_rate = summary_row.get('cover_rate')
    cover_rate_display = f"{float(cover_rate):.3f}" if cover_rate not in ('', None) else ''

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Backtest Winners Summary</title>",
        "<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3}</style>",
        "</head><body>",
        f"<h1>{league} Backtest Winners Summary - Season {season}</h1>",
        "<table><tr><th>Games</th><th>Pushes</th><th>Wins</th><th>Losses</th><th>Accuracy</th><th>Covered Yes</th><th>Covered No</th><th>Covered Push</th><th>Cover Rate</th></tr>",
        f"<tr><td>{summary_row.get('games', '')}</td><td>{summary_row.get('pushes', '')}</td><td>{summary_row.get('wins', '')}</td><td>{summary_row.get('losses', '')}</td><td>{float(summary_row.get('accuracy', 0.0)):.3f}</td><td>{summary_row.get('cover_yes', '')}</td><td>{summary_row.get('cover_no', '')}</td><td>{summary_row.get('cover_push', '')}</td><td>{cover_rate_display}</td></tr></table>",
        f"<p>Per-game CSV: {os.path.basename(per_game_csv)}</p>",
        "<div style='margin:8px 0;'>"
        "<label>Week: <select id='fWeek'><option value=''>All</option>" + week_options + "</select></label>\n"
        " <label>Team: <input id='fTeam' placeholder='ABB or Name' /></label>\n"
        " <label>Team Position: <select id='fPos'><option value=''>Any</option><option value='home'>Home</option><option value='away'>Away</option></select></label>\n"
        " <label>Predicted Side: <select id='fPred'><option value=''>Any</option><option value='home'>Home</option><option value='away'>Away</option><option value='tie'>Tie</option></select></label>\n"
        " <label>Coverage: <select id='fCov'><option value=''>Any</option><option value='covered'>Covered</option><option value='not'>Not Covered</option><option value='push'>Push</option></select></label>\n"
        " <button onclick='filterRows()'>Filter</button> <button onclick='resetFilters()'>Reset</button>"
        "</div>",
        "<script>function norm(x){return (x||'').toLowerCase()}\n"
        "function matchesTeam(home,away,hname,aname,t,pos){if(!t)return true;var hm=home.toLowerCase().includes(t)||hname.includes(t);var am=away.toLowerCase().includes(t)||aname.includes(t);if(pos==='home')return hm; if(pos==='away')return am; return hm||am;}\n"
        "function filterRows(){var w=document.getElementById('fWeek').value;var t=norm(document.getElementById('fTeam').value);var p=document.getElementById('fPred').value;var pos=document.getElementById('fPos').value;var cov=document.getElementById('fCov').value;var rows=document.querySelectorAll('#results tr');rows.forEach(function(r){var show=true; if(w && r.dataset.week!==w){show=false;} var home=r.dataset.home||''; var away=r.dataset.away||''; var pside=r.dataset.pside||''; var hname=(r.dataset.hname||'').toLowerCase(); var aname=(r.dataset.aname||'').toLowerCase(); var covered=(r.dataset.covered||''); if(t && !matchesTeam(home,away,hname,aname,t,pos)){show=false;} if(p && pside!==p){show=false;} if(cov){ if(cov==='covered' && covered!=='yes') show=false; else if(cov==='not' && covered!=='no') show=false; else if(cov==='push' && covered!=='push') show=false;} r.style.display=show?'':'none';});}\n"
        "function resetFilters(){document.getElementById('fWeek').value='';document.getElementById('fTeam').value='';document.getElementById('fPred').value='';document.getElementById('fPos').value='';document.getElementById('fCov').value='';filterRows();}\n"
        "</script>",
        "<table><tr><th>Week</th><th>Date</th><th>Away</th><th>Home</th><th>Projected Margin (Home)</th><th>Actual Margin</th><th>Final Score</th><th>Covered</th><th>Predicted Winner</th><th>Actual Winner</th><th>Correct</th></tr><tbody id='results'>"
    ]

    for row in rows:
        html.append(
            f"<tr data-week='{row['week']}' data-home='{row['home']}' data-away='{row['away']}' data-pside='{row['predicted_team'].lower() if row['predicted_team'] != 'TIE' else 'tie'}' "
            f"data-hname='{row['home'].lower()}' data-aname='{row['away'].lower()}' data-covered='{str(row['covered']).strip().lower()}'>"
            f"<td>{row['week']}</td><td>{row['date']}</td><td>{row['away']}</td><td>{row['home']}</td>"
            f"<td>{float(row['projected_margin']):+.1f}</td><td>{float(row['actual_margin']):+.1f}</td>"
            f"<td>{row['final_score']}</td><td>{row['covered']}</td>"
            f"<td>{row['predicted_team']}</td><td>{row['actual_team']}</td><td>{'Y' if row['correct'] == '1' else ''}</td></tr>"
        )

    html.extend(["</tbody></table>", "</body></html>"])

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))


def main():
    parser = argparse.ArgumentParser(description='Run winners backtests over multiple seasons')
    parser.add_argument('--league', choices=VALID_LEAGUES, default='nfl', help='League to run (default: nfl)')
    parser.add_argument('--seasons', nargs='+', required=True, help='Season list, e.g. 2021-2024 or 2022 2023')
    parser.add_argument('--last-n', type=int, default=17, help='Last N games per team for power (default: 17)')
    parser.add_argument('--hfa', type=float, default=None, help='Home field advantage (default: league average)')
    parser.add_argument('--output', type=str, default='./backtests', help='Output directory')
    args = parser.parse_args()

    league = parse_league(args.league)
    hfa = args.hfa if args.hfa is not None else DEFAULT_HFA[league]
    out_dir = ensure_dir(args.output)

    seasons = sorted(set(iter_seasons(args.seasons)))
    if not seasons:
        raise SystemExit('No valid seasons provided.')

    aggregate_rows: List[Dict[str, Any]] = []
    for season in seasons:
        summary, per_game_csv, summary_csv, summary_html = run_winners_for_season(league, season, args.last_n, hfa, out_dir)
        aggregate_rows.append(summary)
        print(f"Completed {league.upper()} season {season}: per-game -> {per_game_csv}, summary -> {summary_csv}")

    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    aggregate_csv = os.path.join(out_dir, f'backtest_winners_multi_{league}_{ts}.csv')
    with open(aggregate_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['league','season','games','pushes','wins','losses','accuracy','cover_yes','cover_no','cover_push','cover_rate'])
        for row in aggregate_rows:
            writer.writerow([
                row['league'], row['season'], row['games'], row['pushes'], row['wins'],
                row['losses'], row['accuracy'], row['cover_yes'], row['cover_no'],
                row['cover_push'], row['cover_rate']
            ])

    print(f"Wrote aggregate summary CSV: {aggregate_csv}")


if __name__ == '__main__':
    raise SystemExit(main())

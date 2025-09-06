#!/usr/bin/env python3
"""
Residual Analytics & Dashboard for Winners Backtests.

Loads one or more winners backtest per-game CSVs and builds an HTML dashboard
with segment summaries:
 - Overall metrics (MAE raw/calibrated, bias)
 - By Home/Away (based on game perspective)
 - By spread magnitude buckets (|projected_cal|: 0-3, 3-7, 7+)
 - By week (MAE raw/cal)
 - Top teams by MAE (participation-based)
 - Full per-game table with filters (season, week, team, position, predicted side, coverage)

Usage:
  python -m calibration.residual_analytics --inputs backtests/backtest_winners_*.csv --output ./backtests
"""

import argparse
import glob
import os
import sys
from typing import List, Dict, Any
from datetime import datetime, timezone
import pandas as pd
import numpy as np


def ensure_dir(path: str) -> str:
    p = os.path.abspath(os.path.expanduser(path))
    os.makedirs(p, exist_ok=True)
    return p


def load_games(patterns: List[str]) -> pd.DataFrame:
    frames = []
    for pat in patterns:
        for fp in glob.glob(pat):
            try:
                df = pd.read_csv(fp)
                # Required: season/week/home/away and margins
                need_any = {'projected_margin_cal', 'projected_margin', 'actual_margin'}
                if not {'season', 'week', 'home', 'away', 'actual_margin'}.issubset(df.columns):
                    continue
                # Harmonize
                if 'projected_margin_cal' in df.columns:
                    df['proj_cal'] = pd.to_numeric(df['projected_margin_cal'], errors='coerce')
                elif 'projected_margin' in df.columns:
                    df['proj_cal'] = pd.to_numeric(df['projected_margin'], errors='coerce')
                else:
                    continue
                df['proj_raw'] = pd.to_numeric(df.get('projected_margin_raw', df.get('projected_margin', np.nan)), errors='coerce')
                df['actual'] = pd.to_numeric(df['actual_margin'], errors='coerce')
                df['season'] = pd.to_numeric(df['season'], errors='coerce')
                df['week'] = pd.to_numeric(df['week'], errors='coerce')
                df['home'] = df['home'].astype(str)
                df['away'] = df['away'].astype(str)
                df['covered_predicted'] = df.get('covered_predicted', '')
                frames.append(df[['season','week','home','away','proj_raw','proj_cal','actual','covered_predicted']].copy())
            except Exception:
                continue
    if not frames:
        raise SystemExit('No matching per-game winners CSVs found')
    df = pd.concat(frames, ignore_index=True).dropna(subset=['proj_cal','actual'])
    # Residuals: actual - projected
    df['resid_cal'] = df['actual'] - df['proj_cal']
    df['resid_raw'] = df['actual'] - df['proj_raw']
    df['abs_err_cal'] = df['resid_cal'].abs()
    df['abs_err_raw'] = df['resid_raw'].abs()
    # Predicted side from calibrated
    df['pred_side'] = np.where(df['proj_cal']>0,'home', np.where(df['proj_cal']<0,'away','push'))
    return df


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out['overall'] = {
        'mae_raw': df['abs_err_raw'].mean(),
        'mae_cal': df['abs_err_cal'].mean(),
        'bias_raw': df['resid_raw'].mean(),
        'bias_cal': df['resid_cal'].mean(),
        'n': len(df)
    }
    # Home/Away (based on pred_side)
    out['by_side'] = df.groupby('pred_side').agg(
        mae_raw=('abs_err_raw','mean'), mae_cal=('abs_err_cal','mean'), n=('pred_side','size')
    ).reset_index()
    # Spread buckets
    bins = [0,3,7,100]
    labels = ['0-3','3-7','7+']
    df['bucket'] = pd.cut(df['proj_cal'].abs(), bins=bins, labels=labels, right=False)
    out['by_bucket'] = df.groupby('bucket').agg(
        mae_raw=('abs_err_raw','mean'), mae_cal=('abs_err_cal','mean'), n=('bucket','size')
    ).reset_index()
    # Week
    out['by_week'] = df.groupby('week').agg(
        mae_raw=('abs_err_raw','mean'), mae_cal=('abs_err_cal','mean'), n=('week','size')
    ).reset_index().sort_values('week')
    # Teams (participation-based, average of abs error per game)
    team_rows = []
    for side in ['home','away']:
        grp = df.groupby(side).agg(mae_cal=('abs_err_cal','mean'), n=(side,'size')).reset_index().rename(columns={side:'team'})
        team_rows.append(grp)
    teams = pd.concat(team_rows, ignore_index=True)
    teams = teams.groupby('team').agg(mae_cal=('mae_cal','mean'), n=('n','sum')).reset_index().sort_values(['mae_cal','n'], ascending=[False, False])
    out['by_team'] = teams
    return out


def render_html(df: pd.DataFrame, summ: Dict[str, Any], out_path: str):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Residual Analytics</title>"
                "<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3} .section{margin-top:24px}</style>"
                "</head><body>")
        f.write("<h1>Residual Analytics</h1>")
        o = summ['overall']
        f.write(f"<p>Overall: MAE Raw={o['mae_raw']:.2f}, MAE Cal={o['mae_cal']:.2f}, Bias Raw={o['bias_raw']:.2f}, Bias Cal={o['bias_cal']:.2f}, N={o['n']}</p>")
        # By side
        f.write("<div class='section'><h2>By Predicted Side</h2><table><tr><th>Side</th><th>MAE Raw</th><th>MAE Cal</th><th>N</th></tr>")
        for _, r in summ['by_side'].iterrows():
            f.write(f"<tr><td>{r['pred_side']}</td><td>{r['mae_raw']:.2f}</td><td>{r['mae_cal']:.2f}</td><td>{int(r['n'])}</td></tr>")
        f.write("</table></div>")
        # By bucket
        f.write("<div class='section'><h2>By Spread Bucket (|proj_cal|)</h2><table><tr><th>Bucket</th><th>MAE Raw</th><th>MAE Cal</th><th>N</th></tr>")
        for _, r in summ['by_bucket'].iterrows():
            f.write(f"<tr><td>{r['bucket']}</td><td>{r['mae_raw']:.2f}</td><td>{r['mae_cal']:.2f}</td><td>{int(r['n'])}</td></tr>")
        f.write("</table></div>")
        # By week
        f.write("<div class='section'><h2>By Week</h2><table><tr><th>Week</th><th>MAE Raw</th><th>MAE Cal</th><th>N</th></tr>")
        for _, r in summ['by_week'].iterrows():
            f.write(f"<tr><td>{int(r['week'])}</td><td>{r['mae_raw']:.2f}</td><td>{r['mae_cal']:.2f}</td><td>{int(r['n'])}</td></tr>")
        f.write("</table></div>")
        # By team (top 20 worst MAE)
        teams = summ['by_team'].head(20)
        f.write("<div class='section'><h2>Teams with Highest MAE (Top 20)</h2><table><tr><th>Team</th><th>MAE Cal</th><th>N</th></tr>")
        for _, r in teams.iterrows():
            f.write(f"<tr><td>{r['team']}</td><td>{r['mae_cal']:.2f}</td><td>{int(r['n'])}</td></tr>")
        f.write("</table></div>")
        # Full per-game table with filters
        f.write("<div class='section'><h2>All Games</h2>")
        seasons = sorted(df['season'].dropna().unique().tolist())
        f.write("<div style='margin:8px 0;'>" +
                f"<label>Season: <select id='fSeason'><option value=''>All</option>{''.join(f'<option value="{int(s)}">{int(s)}</option>' for s in seasons)}</select></label> " +
                f"<label>Week: <select id='fWeek'><option value=''>All</option>{''.join(f'<option value="{w}">{w}</option>' for w in range(1,19))}</select></label> " +
                "<label>Team: <input id='fTeam' placeholder='ABB'/></label> " +
                "<label>Team Position: <select id='fPos'><option value=''>Any</option><option value='home'>Home</option><option value='away'>Away</option></select></label> " +
                "<label>Predicted Side: <select id='fPred'><option value=''>Any</option><option value='home'>Home</option><option value='away'>Away</option><option value='push'>Push</option></select></label> " +
                "<label>Coverage: <select id='fCov'><option value=''>Any</option><option value='covered'>Covered</option><option value='not'>Not Covered</option><option value='push'>Push</option></select></label> " +
                "<button onclick='filterRows()'>Filter</button> <button onclick='resetFilters()'>Reset</button>" +
                "</div>")
        f.write("<script>function norm(x){return (x||'').toLowerCase()}\n"
                "function matchesTeam(home,away,t,pos){if(!t) return true; var hm=home.toLowerCase().includes(t); var am=away.toLowerCase().includes(t); if(pos==='home') return hm; if(pos==='away') return am; return hm||am;}\n"
                "function filterRows(){var s=document.getElementById('fSeason').value; var w=document.getElementById('fWeek').value; var t=norm(document.getElementById('fTeam').value); var pos=document.getElementById('fPos').value; var pred=document.getElementById('fPred').value; var cov=document.getElementById('fCov').value; var rows=document.querySelectorAll('#games tr'); rows.forEach(function(r){ var show=true; if(s && r.dataset.season!==s) show=false; if(w && r.dataset.week!==w) show=false; var home=r.dataset.home||''; var away=r.dataset.away||''; var pside=r.dataset.pside||''; var covered=r.dataset.covered||''; if(t && !matchesTeam(home,away,t,pos)) show=false; if(pred && pside!==pred) show=false; if(cov){ if(cov==='covered' && covered!=='yes') show=false; else if(cov==='not' && covered!=='no') show=false; else if(cov==='push' && covered!=='push') show=false;} r.style.display=show?'':'none'; }); }\n"
                "function resetFilters(){document.getElementById('fSeason').value=''; document.getElementById('fWeek').value=''; document.getElementById('fTeam').value=''; document.getElementById('fPos').value=''; document.getElementById('fPred').value=''; document.getElementById('fCov').value=''; filterRows(); }\n"
                "</script>")
        f.write("<table><tr><th>Season</th><th>Week</th><th>Away</th><th>Home</th><th>Proj Raw</th><th>Proj Cal</th><th>Actual</th><th>AbsErr Raw</th><th>AbsErr Cal</th><th>Covered</th></tr><tbody id='games'>")
        for _, r in df.sort_values(['season','week']).iterrows():
            pside = 'home' if r['proj_cal']>0 else ('away' if r['proj_cal']<0 else 'push')
            f.write(f"<tr data-season='{int(r['season'])}' data-week='{int(r['week'])}' data-home='{r['home']}' data-away='{r['away']}' data-pside='{pside}' data-covered='{str(r['covered_predicted']).lower()}'>" +
                    f"<td>{int(r['season'])}</td>" +
                    f"<td>{int(r['week'])}</td>" +
                    f"<td>{r['away']}</td>" +
                    f"<td>{r['home']}</td>" +
                    f"<td>{r['proj_raw']:+.1f}</td>" +
                    f"<td>{r['proj_cal']:+.1f}</td>" +
                    f"<td>{r['actual']:+.1f}</td>" +
                    f"<td>{r['abs_err_raw']:.1f}</td>" +
                    f"<td>{r['abs_err_cal']:.1f}</td>" +
                    f"<td>{r['covered_predicted']}</td>" +
                    "</tr>")
        f.write("</tbody></table></div>")
        f.write("</body></html>")


def main():
    ap = argparse.ArgumentParser(description='Residual analytics dashboard')
    ap.add_argument('--inputs', nargs='+', required=True, help='Glob patterns of winners per-game CSVs')
    ap.add_argument('--output', default='./backtests', help='Output directory')
    args = ap.parse_args()

    out_dir = ensure_dir(args.output)
    df = load_games(args.inputs)
    summ = summarize(df)
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out_html = os.path.join(out_dir, f'residual_analytics_{ts}.html')
    render_html(df, summ, out_html)
    print('Residual analytics written:', out_html)


if __name__ == '__main__':
    sys.exit(main())


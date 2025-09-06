import os
import types
import tempfile
import csv
import importlib
import sys


def test_runner_with_mocked_client(monkeypatch):
    # Build a minimal mocked ESPN client
    class MockClient:
        def get_teams(self):
            # Minimal structure used by power rankings
            return []

        def get_scoreboard(self, week=None, season=None):
            return {'events': [], 'week': {'number': week or 1}}

        def has_current_season_games(self, week=None):
            return False

        def get_last_completed_season(self):
            return 2024

        def get_season_final_rankings(self, season):
            return {'events': []}

    # Monkeypatch client factory to return our mock
    # Create a fake client_factory module to satisfy imports without installation
    cf_mod_name = 'power_ranking.power_ranking.api.client_factory'
    pkg_names = ['power_ranking', 'power_ranking.power_ranking', 'power_ranking.power_ranking.api']
    for name in pkg_names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    cf = types.ModuleType(cf_mod_name)
    cf.get_client = lambda strategy='sync': MockClient()
    sys.modules[cf_mod_name] = cf

    # Prepare synthetic power rankings CSV with two teams
    with tempfile.TemporaryDirectory() as td:
        pr = os.path.join(td, 'power.csv')
        with open(pr, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['team_name', 'power_score'])
            w.writerow(['KC', 10.0])
            w.writerow(['BUF', 7.0])

        sched = os.path.join(td, 'schedule.csv')
        with open(sched, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['week', 'home_team', 'away_team', 'game_date'])
            w.writerow([1, 'KC', 'BUF', '2025-09-01'])

        # Prepare a minimal power rankings CSV and stubbed runner hook
        pr_full = os.path.join(td, 'power_full.csv')
        with open(pr_full, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['team_name', 'power_score'])
            w.writerow(['Kansas City Chiefs', 10.0])
            w.writerow(['Buffalo Bills', 7.0])

        # Run spread calculation through DataLoader and SpreadCalculator via runner helpers
        import run_full_projects as rfp
        monkeypatch.setattr(rfp, 'run_power_rankings', lambda week, last_n, output_dir: (
            pr_full,
            [("KC", "Kansas City Chiefs", 10.0), ("BUF", "Buffalo Bills", 7.0)],
            {'season_stats': {}, 'rolling_stats': {}, 'sos_scores': {}}
        ))

        # Call internal helpers directly
        csv_path, rankings, comp = rfp.run_power_rankings(week=1, last_n=17, output_dir=td)
        assert os.path.exists(csv_path)

        abbr_path = rfp.make_abbrev_power_csv(csv_path, td)
        assert os.path.exists(abbr_path)

        # Use offline schedule file
        spreads_csv, spreads = rfp.run_spread_model(abbr_path, sched, 1, td, odds_map=None)
        assert os.path.exists(spreads_csv)
        assert len(spreads) == 1

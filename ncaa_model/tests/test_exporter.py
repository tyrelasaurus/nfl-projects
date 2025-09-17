import os
import tempfile
from ncaa_model.exporter import CSVExporter
from ncaa_model.spread_model import MatchupResult


def test_exporter_writes_files():
    with tempfile.TemporaryDirectory() as td:
        exporter = CSVExporter(output_directory=td)
        rows = [
            MatchupResult(
                week=1,
                home_team='KC',
                away_team='BUF',
                home_power=10.0,
                away_power=7.0,
                neutral_diff=3.0,
                home_field_adj=2.0,
                projected_spread=5.0,
                game_date='2025-09-01',
            )
        ]
        spreads = exporter.export_week_spreads(rows, week=1)
        summary = exporter.export_summary_stats(rows, week=1)
        assert os.path.exists(spreads)
        assert os.path.exists(summary)


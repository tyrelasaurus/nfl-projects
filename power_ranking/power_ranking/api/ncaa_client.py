"""ESPN API client tailored for NCAA Division I FBS football."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import re

from .espn_client import ESPNClient


class CollegeFootballESPNClient(ESPNClient):
    """Thin wrapper around :class:`ESPNClient` for NCAA FBS resources."""

    def __init__(self, base_url: str | None = None, division_group: str = "80"):
        super().__init__(base_url=base_url or "https://site.api.espn.com/apis/site/v2")
        self._league_path = "sports/football/college-football"
        # ESPN group 80 corresponds to the FBS subdivision
        self.division_group = division_group

    # ------------------------------------------------------------------
    # Core fetch helpers
    # ------------------------------------------------------------------
    def get_teams(self) -> List[Dict[str, Any]]:
        params = {"group": self.division_group}
        data = self._make_request(f"{self._league_path}/teams", params=params)
        return data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])

    def get_scoreboard(
        self,
        week: Optional[int] = None,
        season: Optional[int] = None,
        group: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "group": group or self.division_group,
        }
        if week:
            params["week"] = week
        if season:
            params["seasontype"] = "2"  # Regular season
            params["year"] = str(season)
        return self._make_request(f"{self._league_path}/scoreboard", params=params)

    # ------------------------------------------------------------------
    # Season utilities
    # ------------------------------------------------------------------
    def get_current_week(self) -> int:
        try:
            data = self.get_scoreboard()
        except Exception:
            return 1
        return data.get("week", {}).get("number", 1)

    def get_last_completed_season(self) -> int:
        # Treat the last fully completed regular season as previous calendar year
        return datetime.utcnow().year - 1

    def get_last_season_final_rankings(self) -> Dict[str, Any]:
        last_year = self.get_last_completed_season()
        return self.get_season_final_rankings(last_year)

    def get_season_final_rankings(self, season: int) -> Dict[str, Any]:
        events = self._collect_season_events(season)
        return {
            "events": events,
            "week": {"number": max((ev.get("week_number") or 0) for ev in events) if events else 0},
            "season": {"year": season},
            "total_games": len(events),
        }

    def has_current_season_games(self, week: int | None = None) -> bool:
        try:
            data = self.get_scoreboard(week=week)
        except Exception:
            return False
        events = data.get("events") or []
        return any(ev.get("status", {}).get("type", {}).get("state") == "post" for ev in events)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _collect_season_events(self, season: int) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        windows = self._get_calendar_windows(season)
        if not windows:
            # Fallback to legacy week iteration if calendar unavailable
            windows = [
                {
                    'params': {'week': week},
                    'week_hint': week,
                }
                for week in range(1, 18)
            ]

        for window in windows:
            params = {
                'group': self.division_group,
                'seasontype': '2',
                'limit': '1000',
                'year': str(season),
            }
            params.update(window.get('params', {}))

            try:
                data = self._make_request(f"{self._league_path}/scoreboard", params=params)
            except Exception:
                continue

            week_number = data.get('week', {}).get('number') or window.get('week_hint')
            for event in data.get('events') or []:
                status = (event.get('status') or {}).get('type', {})
                if status.get('name') != 'STATUS_FINAL':
                    continue

                event_id = str(event.get('id'))
                if event_id in seen:
                    continue

                if week_number is not None:
                    event.setdefault('week_number', week_number)
                events.append(event)
                seen.add(event_id)

        return events

    # ------------------------------------------------------------------
    # Calendar helpers
    # ------------------------------------------------------------------
    def _get_calendar_windows(self, season: int) -> List[Dict[str, Any]]:
        """Return list of request parameter dictionaries derived from calendar."""

        try:
            meta = self._make_request(
                f"{self._league_path}/scoreboard",
                {
                    'group': self.division_group,
                    'seasontype': '2',
                    'year': str(season),
                    'dates': str(season),
                    'limit': '1',
                },
            )
        except Exception:
            return []

        leagues = meta.get('leagues') or []
        calendar_windows: List[Dict[str, Any]] = []

        for league in leagues:
            calendar = league.get('calendar') or []
            for cal_item in calendar:
                entries = []
                if isinstance(cal_item, dict) and cal_item.get('entries'):
                    entries = cal_item['entries']
                else:
                    entries = [cal_item]

                for entry in entries:
                    start = self._parse_calendar_date(entry, 'startDate')
                    end = self._parse_calendar_date(entry, 'endDate')
                    if not start or not end:
                        continue

                    # Keep windows anchored to the requested season (allow spillover into bowls early next year)
                    if start.year not in (season, season - 1, season + 1):
                        continue
                    if end.year not in (season, season - 1, season + 1):
                        continue
                    date_range = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
                    week_hint = self._extract_week_hint(entry)
                    calendar_windows.append({
                        'params': {'dates': date_range},
                        'week_hint': week_hint,
                    })

        return calendar_windows

    @staticmethod
    def _parse_calendar_date(entry: Any, key: str) -> Optional[datetime]:
        value = None
        if isinstance(entry, dict):
            value = entry.get(key)
        elif isinstance(entry, str) and key == 'startDate':
            # Some calendars may be simple ISO strings
            value = entry
        if not value:
            return None
        try:
            value_str = str(value)
            value_str = value_str.replace('Z', '+00:00')
            return datetime.fromisoformat(value_str)
        except Exception:
            return None

    @staticmethod
    def _extract_week_hint(entry: Any) -> Optional[int]:
        if isinstance(entry, dict):
            value = entry.get('value') or entry.get('week')
            if isinstance(value, int):
                return value
            label = entry.get('label') or entry.get('text')
            if isinstance(label, str):
                match = re.search(r'(\d+)', label)
                if match:
                    try:
                        return int(match.group(1))
                    except ValueError:
                        return None
        return None


__all__ = ["CollegeFootballESPNClient"]

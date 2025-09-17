from typing import Protocol, Any


class ESPNClientLike(Protocol):
    def get_teams(self) -> Any: ...
    def get_scoreboard(self, week: int | None = None, season: int | None = None, *args, **kwargs) -> Any: ...
    def get_current_week(self) -> int: ...
    # Optional extended methods used in backtests/runners
    def get_last_completed_season(self) -> int: ...
    def get_season_final_rankings(self, season: int) -> Any: ...
    def get_last_season_final_rankings(self) -> Any: ...
    def has_current_season_games(self, week: int | None = None) -> bool: ...


def get_client(strategy: str = "sync", league: str = "nfl") -> ESPNClientLike:
    """Return an ESPN client instance according to strategy and league.

    Parameters
    ----------
    strategy:
        - "sync": default synchronous client
        - "performance": high-performance client (NFL-only)
        - "async": asyncio client (NFL-only)
    league:
        - "nfl" (default)
        - "ncaa" / "college" / "college-football" for NCAA FBS data
    """

    s = (strategy or "sync").lower()
    lg = (league or "nfl").lower()

    if lg in {"ncaa", "college", "college-football"}:
        from .ncaa_client import CollegeFootballESPNClient

        if s not in {"sync", "default"}:
            raise ValueError(f"Strategy '{strategy}' not supported for NCAA client")
        return CollegeFootballESPNClient()

    if s == "performance":
        from .performance_client import PerformanceESPNClient
        return PerformanceESPNClient()

    if s == "async":
        from .async_espn_client import AsyncESPNClient
        return AsyncESPNClient()

    from .espn_client import ESPNClient
    return ESPNClient()


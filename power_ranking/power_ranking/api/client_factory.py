from typing import Protocol, Any


class ESPNClientLike(Protocol):
    def get_teams(self) -> Any: ...
    def get_scoreboard(self, week: int | None = None, season: int | None = None) -> Any: ...
    def get_current_week(self) -> int: ...
    # Optional extended methods used in backtests/runners
    def get_last_completed_season(self) -> int: ...
    def get_season_final_rankings(self, season: int) -> Any: ...
    def get_last_season_final_rankings(self) -> Any: ...
    def has_current_season_games(self, week: int | None = None) -> bool: ...


def get_client(strategy: str = "sync") -> ESPNClientLike:
    """Return an ESPN client instance according to strategy.

    - "sync": default synchronous client
    - "performance": high-performance client (threaded/hybrid)
    - "async": returns the async client instance (callers must use `await` appropriately)
    """
    s = (strategy or "sync").lower()
    if s == "performance":
        from .performance_client import PerformanceESPNClient
        return PerformanceESPNClient()
    elif s == "async":
        from .async_espn_client import AsyncESPNClient
        return AsyncESPNClient()
    else:
        from .espn_client import ESPNClient
        return ESPNClient()


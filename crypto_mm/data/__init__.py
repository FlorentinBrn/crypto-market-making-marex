"""Data layer : feed WebSocket, persistance CSV, replay, analytics.

Dépend de `core` pour les types mais encapsule toutes les I/O et la
gestion d'état longue durée (queues, threads writers, buffers rolling).

Les imports de `feed` (qui dépend de `websocket-client`) et `replay`
(qui dépend de `feed`) sont lazy pour que le package reste importable
même quand l'environnement n'a pas encore installé toutes les deps.
"""
from .analytics import SpreadSummary, SpreadTracker
from .storage import CsvAppendWriter, write_dataframe_like

__all__ = [
    "CoinbaseMarketDataApp",
    "CsvAppendWriter",
    "SpreadSummary",
    "SpreadTracker",
    "load_events",
    "replay",
    "write_dataframe_like",
]


def __getattr__(name: str):
    # Lazy imports : on ne charge feed/replay qu'au premier accès.
    # Évite les erreurs d'import quand websocket-client n'est pas installé
    # pour des usages qui n'en ont pas besoin (tests unitaires de core,
    # analytics-only, etc.).
    if name == "CoinbaseMarketDataApp":
        from .feed import CoinbaseMarketDataApp as _c
        return _c
    if name in ("load_events", "replay"):
        from . import replay as _r
        return getattr(_r, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

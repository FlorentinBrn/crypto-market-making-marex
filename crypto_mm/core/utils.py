from __future__ import annotations

from datetime import datetime, timezone


def parse_timestamp(value: str | None) -> datetime:
    """Convertit un timestamp ISO8601 en datetime timezone-aware."""
    if not value:
        return datetime.now(tz=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def utc_now() -> datetime:
    """Horloge UTC utilisée partout dans la stratégie."""
    return datetime.now(tz=timezone.utc)


def local_now() -> datetime:
    """Heure locale de la machine qui exécute l'application (pour l'affichage)."""
    return datetime.now().astimezone()


def to_local_time(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone()


def format_local_time(value: datetime | None, include_ms: bool = True) -> str:
    """Formate une heure pour un affichage console lisible."""
    local_dt = to_local_time(value)
    if local_dt is None:
        return "-"
    fmt = "%H:%M:%S.%f" if include_ms else "%H:%M:%S"
    text = local_dt.strftime(fmt)
    return text[:-3] if include_ms else text

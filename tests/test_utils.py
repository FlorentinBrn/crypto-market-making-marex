from datetime import timezone

from crypto_mm.core.utils import format_local_time, parse_timestamp, utc_now


def test_parse_timestamp_none_returns_utc_now() -> None:
    ts = parse_timestamp(None)
    assert ts.tzinfo is not None


def test_parse_timestamp_handles_z_suffix() -> None:
    ts = parse_timestamp("2024-01-01T00:00:00.123456Z")
    assert ts.tzinfo == timezone.utc
    assert ts.year == 2024


def test_utc_now_is_timezone_aware() -> None:
    assert utc_now().tzinfo is not None


def test_format_local_time_handles_none() -> None:
    assert format_local_time(None) == "-"


def test_format_local_time_returns_hms_with_ms() -> None:
    ts = parse_timestamp("2024-01-01T12:34:56.789000Z")
    out = format_local_time(ts)
    # Format "HH:MM:SS.mmm"
    assert len(out) == 12
    assert out.count(":") == 2
    assert "." in out

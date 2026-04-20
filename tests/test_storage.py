"""Tests du writer CSV asynchrone.

IMPORTANT : CsvAppendWriter démarre un thread writer ; chaque test doit
appeler w.close() pour le stopper proprement, sinon les threads fuitent
et les runs suivants peuvent hang.
"""
from pathlib import Path

import pandas as pd

from crypto_mm.data.storage import CsvAppendWriter, write_dataframe_like


def test_csv_writer_writes_header_and_rows(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    w = CsvAppendWriter(out, flush_size=2)
    try:
        w.append({"a": 1, "b": 2})
        w.append({"a": 3, "b": 4})
        w.flush()
        df = pd.read_csv(out)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2
    finally:
        w.close()


def test_csv_writer_persists_across_flushes(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    w = CsvAppendWriter(out, flush_size=100)
    try:
        # On dépose 2 lignes, on ferme, puis on relit : les 2 doivent être là.
        w.append({"a": 1, "b": 2})
        w.append({"a": 5, "b": 6})
        w.flush()
    finally:
        w.close()
    df = pd.read_csv(out)
    assert len(df) == 2


def test_csv_writer_high_volume(tmp_path: Path) -> None:
    """Scénario réaliste : beaucoup d'appends, un close final."""
    out = tmp_path / "out.csv"
    w = CsvAppendWriter(out, flush_size=50)
    try:
        for i in range(300):
            w.append({"x": i})
        w.flush()
    finally:
        w.close()
    df = pd.read_csv(out)
    assert len(df) == 300


def test_write_dataframe_like_empty_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "empty.csv"
    write_dataframe_like(out, [])
    assert out.exists()
    assert out.read_text() == ""


def test_write_dataframe_like_with_rows(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    write_dataframe_like(out, [{"x": 1, "y": 2}, {"x": 3, "y": 4}])
    df = pd.read_csv(out)
    assert len(df) == 2
    assert list(df.columns) == ["x", "y"]

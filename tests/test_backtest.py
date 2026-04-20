"""Tests du framework de backtest avec reconstruction du carnet."""
from pathlib import Path

import pandas as pd

from crypto_mm.backtest import (
    _read_csv,
    compute_risk_metrics,
    iter_replay_events,
    replay_with_book,
    walkforward_backtest,
)
from crypto_mm.config import Settings


def _write_book_and_trades(tmp_path: Path) -> None:
    """Produit un mini dataset réaliste : 20 snapshots de carnet + 20 trades."""
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)

    book_rows = []
    trade_rows = []
    base_ts = pd.Timestamp("2024-01-01", tz="UTC")
    for i in range(20):
        ts = base_ts + pd.Timedelta(milliseconds=i * 500)
        iso = ts.isoformat()
        price_mid = 100_000.0 + i
        # 3 niveaux bid + 3 niveaux ask par snapshot
        for rank, offset in enumerate([1, 2, 3], start=1):
            book_rows.append(
                {
                    "timestamp": iso,
                    "side": "bid",
                    "rank": rank,
                    "price": price_mid - offset,
                    "quantity": 0.5 * rank,
                }
            )
            book_rows.append(
                {
                    "timestamp": iso,
                    "side": "ask",
                    "rank": rank,
                    "price": price_mid + offset,
                    "quantity": 0.5 * rank,
                }
            )
        # Un trade par tick, alternant BUY/SELL au touch
        trade_rows.append(
            {
                "time": iso,
                "trade_id": f"t{i}",
                "product_id": "BTC-USD",
                "price": price_mid - 1 if i % 2 == 0 else price_mid + 1,
                "size": 0.05,
                "side": "BUY" if i % 2 == 0 else "SELL",
            }
        )
    pd.DataFrame(book_rows).to_csv(raw / "book.csv", index=False)
    pd.DataFrame(trade_rows).to_csv(raw / "trades.csv", index=False)


def test_read_csv_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.csv"
    f.write_text("")
    assert _read_csv(f).empty


def test_read_csv_with_data(tmp_path: Path) -> None:
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2\n3,4\n")
    df = _read_csv(f)
    assert len(df) == 2


def test_iter_replay_events_interleaves_book_and_trades(tmp_path: Path) -> None:
    _write_book_and_trades(tmp_path)
    book_df = pd.read_csv(tmp_path / "raw" / "book.csv")
    trades_df = pd.read_csv(tmp_path / "raw" / "trades.csv")
    events = list(iter_replay_events(book_df, trades_df))
    # Chaque event est un tuple (ts, kind, payload)
    assert all(len(e) == 3 for e in events)
    kinds = [e[1] for e in events]
    assert "book" in kinds
    assert "trade" in kinds


def test_replay_with_book_produces_curve_and_metrics(tmp_path: Path) -> None:
    _write_book_and_trades(tmp_path)
    book_df = pd.read_csv(tmp_path / "raw" / "book.csv")
    trades_df = pd.read_csv(tmp_path / "raw" / "trades.csv")
    settings = Settings(output_dir=tmp_path, quote_refresh_interval_ms=0)
    events = iter_replay_events(book_df, trades_df)
    curve, metrics = replay_with_book(events, settings)
    # La courbe doit avoir au moins quelques lignes.
    assert len(curve) >= 1
    # Toutes les métriques attendues doivent être présentes.
    expected_keys = {
        "final_pnl",
        "total_return_pct",
        "max_drawdown_usd",
        "max_drawdown_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "fills",
        "fill_rate_pct",
        "hit_ratio_pct",
        "avg_pnl_per_fill",
        "inventory_std_btc",
        "time_in_loss_pct",
        "reduce_only_pct",
    }
    assert expected_keys.issubset(set(metrics.keys()))


def test_compute_risk_metrics_empty_curve_returns_zeros() -> None:
    settings = Settings()
    m = compute_risk_metrics(
        pd.DataFrame(),
        settings,
        fills=[],
        trades_encountered=0,
        trades_eligible_for_fill=0,
        reduce_only_ticks=0,
        total_ticks=0,
    )
    assert m["final_pnl"] == 0.0
    assert m["fills"] == 0
    assert m["fill_rate_pct"] == 0.0


def test_walkforward_minimum_folds(tmp_path: Path) -> None:
    """Le walkforward doit lever ValueError si pas assez de données."""
    _write_book_and_trades(tmp_path)
    settings = Settings(output_dir=tmp_path, quote_refresh_interval_ms=0)
    # n_folds=10 sur 20 trades → fold_size=2 < 50 → doit raise.
    raised = False
    try:
        walkforward_backtest(tmp_path, settings, n_folds=10)
    except ValueError:
        raised = True
    assert raised, "walkforward_backtest aurait dû lever ValueError"

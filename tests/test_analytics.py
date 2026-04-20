from crypto_mm.data.analytics import SpreadTracker
from crypto_mm.core.orderbook import OrderBook


def _make_book() -> OrderBook:
    book = OrderBook()
    book.apply_coinbase_event(
        {
            "type": "snapshot",
            "updates": [
                {"side": "bid", "price_level": "99999", "new_quantity": "0.5"},
                {"side": "bid", "price_level": "99998", "new_quantity": "1.0"},
                {"side": "ask", "price_level": "100001", "new_quantity": "0.5"},
                {"side": "ask", "price_level": "100002", "new_quantity": "1.0"},
            ],
        }
    )
    return book


def test_update_fills_timeline_and_microstructure_rows() -> None:
    book = _make_book()
    st = SpreadTracker(sizes=[0.1, 1.0])
    st.update("2024-01-01T00:00:00Z", book)
    assert st.timeline_rows
    assert st.microstructure_rows
    micro = st.latest_microstructure()
    assert "mid_price" in micro
    assert "micro_signal_bps" in micro


def test_on_trade_updates_trade_flow_and_vpin() -> None:
    st = SpreadTracker(sizes=[0.1], vpin_bucket_size=0.3, vpin_history_size=5)
    # Convention Coinbase : side = maker. BUY → agresseur SELL.
    st.on_trade(ts_ms=1_000.0, side="SELL", size=0.2)  # agresseur = BUY
    st.on_trade(ts_ms=1_100.0, side="SELL", size=0.2)  # agresseur = BUY
    # Sur deux trades 100% "achat agresseur", trade_flow doit être positif.
    flow = st._trade_flow_signed_ratio()
    assert flow > 0.5


def test_on_trade_builds_vpin_buckets() -> None:
    st = SpreadTracker(sizes=[0.1], vpin_bucket_size=0.5, vpin_history_size=10)
    # On pousse au-delà de 0.5 BTC pour fermer un bucket.
    st.on_trade(ts_ms=1_000.0, side="SELL", size=0.3)
    st.on_trade(ts_ms=1_010.0, side="SELL", size=0.3)
    # Un bucket doit avoir été fermé.
    assert len(st._vpin_buckets) >= 1
    # VPIN doit être >= 0.
    assert st.last_vpin >= 0.0


def test_trade_flow_sliding_window_trims_old_trades() -> None:
    st = SpreadTracker(sizes=[0.1], trade_flow_window_ms=1_000)
    st.on_trade(ts_ms=0.0, side="SELL", size=1.0)  # agresseur BUY
    # Un trade récent côté opposé → devrait basculer le ratio.
    st.on_trade(ts_ms=5_000.0, side="BUY", size=1.0)  # agresseur SELL
    flow = st._trade_flow_signed_ratio()
    # Le premier trade est hors fenêtre, on ne garde que le récent (SELL agressif).
    assert flow < 0


def test_combine_signals_clips_at_bounds() -> None:
    combined = SpreadTracker._combine_signals(
        microprice_edge_bps=100.0,
        imbalance_l3=1.0,
        ofi_ewma=1000.0,
        trade_flow=1.0,
    )
    assert combined == 6.0

    combined = SpreadTracker._combine_signals(
        microprice_edge_bps=-100.0,
        imbalance_l3=-1.0,
        ofi_ewma=-1000.0,
        trade_flow=-1.0,
    )
    assert combined == -6.0


def test_combine_signals_zero_when_all_zero() -> None:
    assert SpreadTracker._combine_signals(0.0, 0.0, 0.0, 0.0) == 0.0


def test_summaries_have_stats_after_updates() -> None:
    book = _make_book()
    st = SpreadTracker(sizes=[0.1])
    for i in range(3):
        st.update(f"2024-01-01T00:00:0{i}Z", book)
    summaries = st.summarize()
    assert summaries[0.1].observations == 3
    assert summaries[0.1].average is not None

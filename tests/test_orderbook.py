from crypto_mm.core.orderbook import OrderBook


def _build_book() -> OrderBook:
    book = OrderBook()
    book.apply_coinbase_event(
        {
            "type": "snapshot",
            "updates": [
                {"side": "bid", "price_level": "99999", "new_quantity": "0.5"},
                {"side": "bid", "price_level": "99998", "new_quantity": "1.0"},
                {"side": "bid", "price_level": "99997", "new_quantity": "2.0"},
                {"side": "ask", "price_level": "100001", "new_quantity": "0.5"},
                {"side": "ask", "price_level": "100002", "new_quantity": "1.0"},
                {"side": "ask", "price_level": "100003", "new_quantity": "2.0"},
            ],
        }
    )
    return book


def test_best_bid_ask_and_mid() -> None:
    book = _build_book()
    assert book.best_bid() == 99999.0
    assert book.best_ask() == 100001.0
    assert book.mid_price() == 100000.0


def test_microprice_is_between_bid_and_ask() -> None:
    book = _build_book()
    micro = book.microprice()
    assert micro is not None
    assert 99999.0 <= micro <= 100001.0


def test_imbalance_is_zero_when_symmetric() -> None:
    book = _build_book()
    # Bids et asks symétriques → imbalance doit être très proche de 0.
    assert abs(book.imbalance_by_levels(3)) < 1e-9
    assert abs(book.imbalance_by_volume(1.0)) < 1e-9


def test_imbalance_positive_when_bids_heavier() -> None:
    book = OrderBook()
    book.apply_coinbase_event(
        {
            "type": "snapshot",
            "updates": [
                {"side": "bid", "price_level": "100", "new_quantity": "10"},
                {"side": "ask", "price_level": "101", "new_quantity": "1"},
            ],
        }
    )
    assert book.imbalance_by_levels(1) > 0.5


def test_touch_depth_returns_best_size() -> None:
    book = _build_book()
    assert book.touch_depth("bid") == 0.5
    assert book.touch_depth("ask") == 0.5


def test_depth_metrics_compute_spread() -> None:
    book = _build_book()
    metrics = book.depth_metrics(1.0)
    assert metrics.spread_abs is not None
    assert metrics.spread_abs > 0


def test_compute_ofi_delta_returns_zero_at_first_call() -> None:
    book = _build_book()
    assert book.compute_ofi_delta() == 0.0


def test_compute_ofi_delta_positive_when_bid_rises() -> None:
    book = _build_book()
    book.compute_ofi_delta()  # initialise la mémoire

    # Bid monte de 99999 à 99999.5
    book.apply_coinbase_event(
        {
            "type": "update",
            "updates": [
                {"side": "bid", "price_level": "99999.5", "new_quantity": "0.7"},
            ],
        }
    )
    ofi = book.compute_ofi_delta()
    assert ofi > 0  # nouveau best bid → pression acheteuse


def test_compute_ofi_delta_negative_when_ask_falls() -> None:
    book = _build_book()
    book.compute_ofi_delta()
    # Nouveau meilleur ask plus bas → pression vendeuse, OFI < 0
    book.apply_coinbase_event(
        {
            "type": "update",
            "updates": [
                {"side": "ask", "price_level": "100000.5", "new_quantity": "0.8"},
            ],
        }
    )
    ofi = book.compute_ofi_delta()
    assert ofi < 0

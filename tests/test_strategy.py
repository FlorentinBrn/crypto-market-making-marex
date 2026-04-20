from datetime import datetime, timedelta, timezone

from crypto_mm.core.models import Trade
from crypto_mm.core.risk import RiskManager
from crypto_mm.core.strategy import MarketMaker


def make_strategy() -> MarketMaker:
    risk = RiskManager(max_notional_usd=1_000_000, max_loss_usd=100_000, initial_equity=1_000_000)
    return MarketMaker(
        quote_size_btc=0.1,
        min_quote_size_btc=0.02,
        base_half_spread_bps=1.0,
        min_half_spread_bps=0.3,
        max_half_spread_bps=8.0,
        inventory_skew_bps_per_btc=6.0,
        risk_manager=risk,
        volatility_window=20,
        volatility_spread_multiplier=0.0,
        initial_cash_usd=1_000_000,
        touch_join_threshold_bps=2.0,
        queue_ahead_factor=1.0,
        fill_intensity=2.0,
        cooldown_ms_after_fill=0,
        quote_refresh_interval_ms=0,
        microprice_weight=1.0,
        imbalance_shift_bps=3.0,
        imbalance_widening_bps=1.5,
    )


def test_quotes_are_skewed_when_position_is_long() -> None:
    mm = make_strategy()
    mm.update_quotes(100_000.0, best_bid=99_999.0, best_ask=100_001.0)
    mm.position_btc = 1.0
    bid, ask = mm.update_quotes(100_000.0, best_bid=99_999.0, best_ask=100_001.0, force=True)
    assert bid is not None and ask is not None
    # Avec long, on abaisse l'ask (sortir l'inventaire) et on recule le bid.
    assert ask.price < 100_001.0 + 1e-9
    assert bid.price <= 99_999.0


def test_buy_then_sell_generates_realized_pnl() -> None:
    mm = make_strategy()
    mm.update_quotes(100_000.0, best_bid=100_000.0, best_ask=100_001.0)
    mm.current_bid.price = 100_000.0
    mm.current_ask.price = 100_100.0

    base_time = datetime.now(timezone.utc)
    buy_trade = Trade(
        trade_id="1",
        product_id="BTC-USD",
        price=100_000.0,
        size=1.0,
        side="BUY",
        time=base_time,
    )
    fills = mm.on_trade(
        buy_trade,
        best_bid=100_000.0,
        best_ask=100_001.0,
        bid_touch_depth=0.01,
        ask_touch_depth=0.01,
        mid_price=100_000.5,
    )
    assert len(fills) == 1
    assert mm.position_btc > 0
    assert mm.avg_entry_price == 100_000.0

    sell_trade = Trade(
        trade_id="2",
        product_id="BTC-USD",
        price=100_100.0,
        size=1.0,
        side="SELL",
        time=base_time + timedelta(milliseconds=1),
    )
    fills = mm.on_trade(
        sell_trade,
        best_bid=100_000.0,
        best_ask=100_100.0,
        bid_touch_depth=0.01,
        ask_touch_depth=0.01,
        mid_price=100_050.0,
    )
    assert len(fills) == 1
    assert mm.position_btc == 0.0
    assert mm.realized_pnl > 0.0


def test_touch_fill_occurs_when_quote_is_joining_best_bid() -> None:
    mm = make_strategy()
    bid, ask = mm.update_quotes(100_000.0, best_bid=100_000.0, best_ask=100_001.0)
    assert bid is not None
    assert bid.price == 100_000.0

    trade = Trade(
        trade_id="touch-1",
        product_id="BTC-USD",
        price=100_000.0,
        size=0.25,
        side="BUY",
        time=datetime.now(timezone.utc),
    )
    fills = mm.on_trade(
        trade,
        best_bid=100_000.0,
        best_ask=100_001.0,
        bid_touch_depth=0.10,
        ask_touch_depth=0.10,
        mid_price=100_000.5,
    )
    assert len(fills) == 1
    assert 0.0 < fills[0].size <= mm.quote_size_btc


def test_reduce_only_disables_bid_when_inventory_is_long_and_risk_is_stressed() -> None:
    risk = RiskManager(
        max_notional_usd=100_000,
        max_loss_usd=100_000,
        initial_equity=1_000_000,
        reduce_only_loss_utilization=0.75,
        reduce_only_exposure_utilization=0.50,
    )
    mm = MarketMaker(
        quote_size_btc=0.1,
        min_quote_size_btc=0.02,
        base_half_spread_bps=1.0,
        min_half_spread_bps=0.3,
        max_half_spread_bps=8.0,
        inventory_skew_bps_per_btc=6.0,
        risk_manager=risk,
        volatility_window=20,
        volatility_spread_multiplier=0.0,
        initial_cash_usd=1_000_000,
        touch_join_threshold_bps=2.0,
        queue_ahead_factor=1.0,
        fill_intensity=1.5,
        cooldown_ms_after_fill=0,
        quote_refresh_interval_ms=0,
    )
    mm.position_btc = 0.9
    bid, ask = mm.update_quotes(100_000.0, best_bid=100_000.0, best_ask=100_001.0)
    assert bid is None
    assert ask is not None


def test_risk_blocks_quote_if_position_would_exceed_notional() -> None:
    risk = RiskManager(max_notional_usd=5_000, max_loss_usd=100_000, initial_equity=1_000_000)
    mm = MarketMaker(
        quote_size_btc=0.1,
        min_quote_size_btc=0.02,
        base_half_spread_bps=1.0,
        min_half_spread_bps=0.3,
        max_half_spread_bps=8.0,
        inventory_skew_bps_per_btc=0.0,
        risk_manager=risk,
        volatility_window=20,
        volatility_spread_multiplier=0.0,
        initial_cash_usd=1_000_000,
        touch_join_threshold_bps=2.0,
        queue_ahead_factor=1.0,
        fill_intensity=1.5,
        cooldown_ms_after_fill=0,
        quote_refresh_interval_ms=0,
    )
    bid, ask = mm.update_quotes(100_000.0, best_bid=100_000.0, best_ask=100_001.0)
    assert bid is None
    assert ask is None


def test_microprice_signal_pushes_quotes_up_when_edge_is_positive() -> None:
    mm = make_strategy()
    bid, ask = mm.update_quotes(
        100_000.0,
        best_bid=99_999.0,
        best_ask=100_001.0,
        microprice=100_001.0,
        imbalance_l1=0.6,
        imbalance_l3=0.5,
        force=True,
    )
    assert bid is not None and ask is not None
    assert bid.price >= 99_999.0
    assert ask.price >= 100_001.0


def test_combined_signal_bps_is_used_directly() -> None:
    """Si on passe un signal combiné, il prend le dessus sur la reconstruction simple.

    On utilise un marché volontairement LARGE (spread 20 bps) pour que le
    touch-join ne clampe pas nos quotes au best_bid/best_ask (ce qui
    masquerait l'effet du signal).
    """
    mm = make_strategy()
    # Signal fort positif → quotes doivent remonter.
    bid_pos, ask_pos = mm.update_quotes(
        100_000.0,
        best_bid=99_900.0,
        best_ask=100_100.0,
        combined_signal_bps=5.0,
        force=True,
    )
    assert bid_pos is not None and ask_pos is not None

    mm2 = make_strategy()
    bid_neg, ask_neg = mm2.update_quotes(
        100_000.0,
        best_bid=99_900.0,
        best_ask=100_100.0,
        combined_signal_bps=-5.0,
        force=True,
    )
    assert bid_neg is not None and ask_neg is not None
    # Signal positif → prix plus hauts que signal négatif.
    assert bid_pos.price > bid_neg.price
    assert ask_pos.price > ask_neg.price


def test_high_vpin_widens_spread() -> None:
    """VPIN au-dessus du seuil de référence doit élargir le half_spread."""
    mm_low_vpin = make_strategy()
    mm_low_vpin.update_quotes(
        100_000.0, best_bid=99_990.0, best_ask=100_010.0, vpin=0.0, force=True
    )
    half_low = mm_low_vpin.last_quote_context["half_spread_bps"]

    mm_high_vpin = make_strategy()
    mm_high_vpin.update_quotes(
        100_000.0, best_bid=99_990.0, best_ask=100_010.0, vpin=0.9, force=True
    )
    half_high = mm_high_vpin.last_quote_context["half_spread_bps"]

    assert half_high > half_low


def test_bandit_off_by_default() -> None:
    mm = make_strategy()
    assert mm.bandit is None
    assert mm.use_contextual_bandit is False


def test_bandit_on_when_enabled() -> None:
    risk = RiskManager(max_notional_usd=1_000_000, max_loss_usd=100_000, initial_equity=1_000_000)
    mm = MarketMaker(
        quote_size_btc=0.1,
        base_half_spread_bps=1.0,
        inventory_skew_bps_per_btc=6.0,
        risk_manager=risk,
        quote_refresh_interval_ms=0,
        use_contextual_bandit=True,
    )
    assert mm.bandit is not None

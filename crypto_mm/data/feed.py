from __future__ import annotations

import json
import threading
import time
from collections import deque

import websocket
from rich.live import Live

from .analytics import SpreadTracker
from ..tools.bench import (
    STAGE_APPLY_BOOK,
    STAGE_COMPUTE_SIGNALS,
    STAGE_ON_TRADE,
    STAGE_PARSE,
    STAGE_RENDER,
    STAGE_TOTAL_L2,
    STAGE_TOTAL_TRADE,
    STAGE_UPDATE_QUOTES,
    LatencyRecorder,
)
from ..ui.config import Settings
from ..ui.console import build_dashboard, build_placeholder
from ..core.models import Trade
from ..core.orderbook import OrderBook
from ..core.risk import RiskManager
from .storage import CsvAppendWriter
from ..core.strategy import MarketMaker
from ..core.utils import parse_timestamp


class CoinbaseMarketDataApp:
    """Application principale de collecte + simulation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # Composants cœur
        self.book = OrderBook()
        self.spread_tracker = SpreadTracker(
            sizes=[0.1, 1, 5, 10],
            ofi_ewma_alpha=settings.ofi_ewma_alpha,
            trade_flow_window_ms=settings.trade_flow_window_ms,
            vpin_bucket_size=settings.vpin_bucket_size_btc,
            vpin_history_size=settings.vpin_history_size,
            fast_vol_alpha=settings.fast_vol_alpha,
        )
        self.recent_trades: deque[dict] = deque(
            maxlen=settings.recent_trades_to_display
        )
        self.risk_manager = RiskManager(
            max_notional_usd=settings.max_notional_usd,
            max_loss_usd=settings.max_loss_usd,
            initial_equity=settings.initial_cash_usd,
            reduce_only_loss_utilization=settings.reduce_only_loss_utilization,
            reduce_only_exposure_utilization=settings.reduce_only_exposure_utilization,
        )
        self.strategy = MarketMaker(
            quote_size_btc=settings.quote_size_btc,
            min_quote_size_btc=settings.min_quote_size_btc,
            base_half_spread_bps=settings.base_half_spread_bps,
            min_half_spread_bps=settings.min_half_spread_bps,
            max_half_spread_bps=settings.max_half_spread_bps,
            inventory_skew_bps_per_btc=settings.inventory_skew_bps_per_btc,
            risk_manager=self.risk_manager,
            volatility_window=settings.volatility_window,
            volatility_spread_multiplier=settings.volatility_spread_multiplier,
            initial_cash_usd=settings.initial_cash_usd,
            touch_join_threshold_bps=settings.touch_join_threshold_bps,
            queue_ahead_factor=settings.queue_ahead_factor,
            fill_intensity=settings.fill_intensity,
            cooldown_ms_after_fill=settings.cooldown_ms_after_fill,
            quote_refresh_interval_ms=settings.quote_refresh_interval_ms,
            microprice_weight=settings.microprice_weight,
            imbalance_shift_bps=settings.imbalance_shift_bps,
            imbalance_widening_bps=settings.imbalance_widening_bps,
            use_contextual_bandit=settings.use_contextual_bandit,
            ewma_spread_alpha=settings.ewma_spread_alpha,
            rolling_spread_window=settings.rolling_spread_window,
            combined_signal_weight=settings.combined_signal_weight,
            vpin_reference=settings.vpin_reference,
            vpin_widening_bps_per_unit=settings.vpin_widening_bps_per_unit,
            fast_vol_widening_coef=settings.fast_vol_widening_coef,
            ofi_aggressive_join_threshold=settings.ofi_aggressive_join_threshold,
        )

        # Writers CSV async
        raw_dir = settings.output_dir / "raw"
        sim_dir = settings.output_dir / "simulation"
        analytics_dir = settings.output_dir / "analytics"
        self.book_writer = CsvAppendWriter(
            raw_dir / "book.csv", settings.csv_flush_size
        )
        self.trade_writer = CsvAppendWriter(
            raw_dir / "trades.csv", settings.csv_flush_size
        )
        self.fill_writer = CsvAppendWriter(
            sim_dir / "fills.csv", settings.csv_flush_size
        )
        self.pnl_writer = CsvAppendWriter(sim_dir / "pnl.csv", settings.csv_flush_size)
        self.state_writer = CsvAppendWriter(
            analytics_dir / "state.csv", settings.csv_flush_size
        )

        # WebSocket
        self.ws: websocket.WebSocketApp | None = None
        self.should_stop = False
        self.last_render = 0.0
        self.live: Live | None = None
        self.state_lock = threading.RLock()

        # Diagnostics
        self.connection_status = "initialized"
        self.last_error: str | None = None
        self.messages_received = 0
        self.trades_processed = 0
        self.last_message_time: float | None = None

        # Throttle de l'écriture d'état (state.csv + pnl.csv) — écrire à
        # chaque message L2 explose le volume disque alors que ces lignes
        # ne sont utiles que périodiquement.
        self._last_state_write_ms: float = 0.0
        self._state_write_min_interval_ms: float = 100.0

        # Bench de latence : activé via Settings.bench_latency.
        self.latency_recorder = LatencyRecorder(enabled=settings.bench_latency)
        self._bench_flush_every_n: int = 5_000
        self._bench_msg_since_flush: int = 0

    # ------------------------------------------------------------------
    # Callbacks websocket
    # ------------------------------------------------------------------
    def on_open(self, ws: websocket.WebSocketApp) -> None:
        self.connection_status = "connected"
        for channel in ["level2", "market_trades"]:
            payload = {
                "type": "subscribe",
                "product_ids": [self.settings.product_id],
                "channel": channel,
            }
            ws.send(json.dumps(payload))
        ws.send(json.dumps({"type": "subscribe", "channel": "heartbeats"}))

    def on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        rec = self.latency_recorder
        t_parse = rec.start()
        data = json.loads(message)
        channel = data.get("channel")
        timestamp = data.get("timestamp")
        events = data.get("events", [])
        rec.record(STAGE_PARSE, t_parse)

        self.messages_received += 1
        self.last_message_time = time.time()

        if channel == "l2_data" or channel == "level2":
            t_l2 = rec.start()
            self._handle_level2(timestamp, events)
            rec.record(STAGE_TOTAL_L2, t_l2)
        elif channel == "market_trades":
            t_tr = rec.start()
            self._handle_market_trades(events)
            rec.record(STAGE_TOTAL_TRADE, t_tr)

        t_render = rec.start()
        self._maybe_render(timestamp)
        rec.record(STAGE_RENDER, t_render)

        # Flush périodique du CSV de latence — hors chemin critique car
        # amortis sur 5000 messages.
        if rec.enabled:
            self._bench_msg_since_flush += 1
            if self._bench_msg_since_flush >= self._bench_flush_every_n:
                self._bench_msg_since_flush = 0
                rec.flush(self.settings.output_dir / "bench" / "latency.csv")

    def on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        self.last_error = str(error)
        self.connection_status = "error"
        print(f"Erreur websocket: {error}")

    def on_close(
        self, ws: websocket.WebSocketApp, close_status_code, close_msg
    ) -> None:
        self.connection_status = "closed"
        print(f"Connexion fermée. code={close_status_code} message={close_msg}")

    # ------------------------------------------------------------------
    # Handler level2
    # ------------------------------------------------------------------
    def _handle_level2(self, timestamp: str | None, events: list[dict]) -> None:
        rec = self.latency_recorder
        with self.state_lock:
            # 1) Applique les updates.
            t_apply = rec.start()
            book = self.book
            for event in events:
                book.apply_coinbase_event(event)
            rec.record(STAGE_APPLY_BOOK, t_apply)
            if not timestamp:
                return

            # 2) Snapshot rows pour book.csv (top N cachés après 1ère lecture).
            top_n = self.settings.top_levels_to_display
            rows = book.snapshot_rows(top_n)
            for row in rows:
                row["timestamp"] = timestamp
            self.book_writer.extend(rows)

            # 3) Met à jour les métriques microstructure. Les getters sont
            # cachés par l'OrderBook, donc les appels suivants seront gratuits.
            t_sig = rec.start()
            self.spread_tracker.update(timestamp, book)

            # 4) Lecture unique des valeurs carnet.
            mid = book.mid_price()
            best_bid = book.best_bid()
            best_ask = book.best_ask()
            best_bid_size = book.best_bid_size()
            best_ask_size = book.best_ask_size()
            microprice = book.microprice()
            imbalance_l1 = book.imbalance_by_levels(1)
            imbalance_l3 = book.imbalance_by_levels(3)
            micro = self.spread_tracker.latest_microstructure()
            rec.record(STAGE_COMPUTE_SIGNALS, t_sig)

            # 5) Update quotes.
            t_quote = rec.start()
            self.strategy.update_quotes(
                mid_price=mid,
                best_bid=best_bid,
                best_ask=best_ask,
                best_bid_size=best_bid_size,
                best_ask_size=best_ask_size,
                microprice=microprice,
                imbalance_l1=imbalance_l1,
                imbalance_l3=imbalance_l3,
                ofi_ewma=micro.get("ofi_ewma") or 0.0,
                trade_flow_signed=micro.get("trade_flow_signed") or 0.0,
                vpin=micro.get("vpin") or 0.0,
                fast_vol_bps=micro.get("fast_vol_bps") or 0.0,
                combined_signal_bps=micro.get("micro_signal_bps"),
            )
            rec.record(STAGE_UPDATE_QUOTES, t_quote)

            # 6) Écriture d'état throttlée.
            now_ms = time.time() * 1000.0
            if now_ms - self._last_state_write_ms >= self._state_write_min_interval_ms:
                self._last_state_write_ms = now_ms
                mtm = self.strategy.mark_to_market(mid)
                self.pnl_writer.append({"timestamp": timestamp, **mtm})
                self.state_writer.append(
                    self._build_state_row(
                        timestamp,
                        mid,
                        best_bid,
                        best_ask,
                        best_bid_size,
                        best_ask_size,
                        micro,
                        mtm,
                    )
                )

    def _build_state_row(
        self,
        timestamp: str | None,
        mid: float | None,
        best_bid: float | None,
        best_ask: float | None,
        best_bid_size: float | None,
        best_ask_size: float | None,
        micro: dict,
        mtm: dict,
    ) -> dict:
        """Construit une ligne d'état — appelée seulement quand on écrit."""
        ctx = self.strategy.last_quote_context
        risk_status = self.risk_manager.status(
            mtm["equity"], self.strategy.position_btc, mid
        )
        return {
            "timestamp": timestamp,
            "mid_price": mid,
            "best_bid": best_bid,
            "best_bid_size": best_bid_size,
            "best_ask": best_ask,
            "best_ask_size": best_ask_size,
            "microprice": micro.get("microprice"),
            "microprice_edge_bps": micro.get("microprice_edge_bps"),
            "imbalance_l1": micro.get("imbalance_l1"),
            "imbalance_l3": micro.get("imbalance_l3"),
            "imbalance_1btc": micro.get("imbalance_1btc"),
            "ofi_event": micro.get("ofi_event"),
            "ofi_ewma": micro.get("ofi_ewma"),
            "trade_flow_signed": micro.get("trade_flow_signed"),
            "vpin": micro.get("vpin"),
            "fast_vol_bps": micro.get("fast_vol_bps"),
            "micro_signal_bps": micro.get("micro_signal_bps"),
            "position_btc": mtm["position_btc"],
            "avg_entry_price": mtm["avg_entry_price"],
            "exposure_usd": mtm["exposure_usd"],
            "realized_pnl": mtm["realized_pnl"],
            "unrealized_pnl": mtm["unrealized_pnl"],
            "equity": mtm["equity"],
            "quote_bid": (
                None
                if self.strategy.current_bid is None
                else self.strategy.current_bid.price
            ),
            "quote_ask": (
                None
                if self.strategy.current_ask is None
                else self.strategy.current_ask.price
            ),
            "quote_size_btc": ctx.get("quote_size_btc"),
            "risk_level": risk_status.get("risk_level"),
            "health_score": risk_status.get("health_score"),
            "reduce_only": risk_status.get("reduce_only"),
            "loss_utilization": risk_status.get("loss_utilization"),
            "exposure_utilization": risk_status.get("exposure_utilization"),
            "half_spread_bps": ctx.get("half_spread_bps"),
            "volatility_bps": ctx.get("volatility_bps"),
            "signal_shift_bps": ctx.get("signal_shift_bps"),
            "inventory_skew_bps": ctx.get("inventory_skew_bps"),
            "rolling_inside_spread_bps": ctx.get("rolling_inside_spread_bps"),
            "ewma_inside_spread_bps": ctx.get("ewma_inside_spread_bps"),
            "toxicity_widening_bps": ctx.get("toxicity_widening_bps"),
            "fast_vol_widening_bps": ctx.get("fast_vol_widening_bps"),
            "bandit_arm": ctx.get("bandit_arm"),
            "bandit_state": ctx.get("bandit_state"),
            "bandit_reward": ctx.get("bandit_reward"),
        }

    # ------------------------------------------------------------------
    # Handler market_trades — chemin fill
    # ------------------------------------------------------------------
    def _handle_market_trades(self, events: list[dict]) -> None:
        with self.state_lock:
            book = self.book
            best_bid = book.best_bid()
            best_ask = book.best_ask()
            mid = book.mid_price()
            bid_touch_depth = book.touch_depth("bid") if best_bid is not None else 0.0
            ask_touch_depth = book.touch_depth("ask") if best_ask is not None else 0.0

            for event in events:
                for trade_msg in event.get("trades", []):
                    trade = Trade(
                        trade_id=str(trade_msg["trade_id"]),
                        product_id=trade_msg["product_id"],
                        price=float(trade_msg["price"]),
                        size=float(trade_msg["size"]),
                        side=trade_msg["side"].upper(),
                        time=parse_timestamp(trade_msg.get("time")),
                    )
                    row = {
                        "time": trade.time.isoformat(),
                        "trade_id": trade.trade_id,
                        "product_id": trade.product_id,
                        "price": trade.price,
                        "size": trade.size,
                        "side": trade.side,
                    }
                    self.trade_writer.append(row)
                    self.recent_trades.appendleft(row)
                    self.trades_processed += 1

                    # Met à jour VPIN / trade-flow côté analytics.
                    self.spread_tracker.on_trade(
                        ts_ms=trade.time.timestamp() * 1000.0,
                        side=trade.side,
                        size=trade.size,
                    )

                    t_on_trade = self.latency_recorder.start()
                    fills = self.strategy.on_trade(
                        trade,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        bid_touch_depth=bid_touch_depth,
                        ask_touch_depth=ask_touch_depth,
                        mid_price=mid,
                    )
                    self.latency_recorder.record(STAGE_ON_TRADE, t_on_trade)
                    if fills:
                        for fill in fills:
                            self.fill_writer.append(
                                {
                                    "time": fill.time.isoformat(),
                                    "side": fill.side,
                                    "price": fill.price,
                                    "size": fill.size,
                                    "reason": fill.reason,
                                }
                            )
                        # Après un fill, recalcul immédiat des quotes pour
                        # refléter le nouveau risque/inventaire.
                        micro = self.spread_tracker.latest_microstructure()
                        self.strategy.update_quotes(
                            mid_price=mid,
                            best_bid=best_bid,
                            best_ask=best_ask,
                            best_bid_size=book.best_bid_size(),
                            best_ask_size=book.best_ask_size(),
                            microprice=book.microprice(),
                            imbalance_l1=book.imbalance_by_levels(1),
                            imbalance_l3=book.imbalance_by_levels(3),
                            ofi_ewma=micro.get("ofi_ewma") or 0.0,
                            trade_flow_signed=micro.get("trade_flow_signed") or 0.0,
                            vpin=micro.get("vpin") or 0.0,
                            fast_vol_bps=micro.get("fast_vol_bps") or 0.0,
                            combined_signal_bps=micro.get("micro_signal_bps"),
                            force=True,
                        )

    # ------------------------------------------------------------------
    # Rendu dashboard
    # ------------------------------------------------------------------
    def _maybe_render(self, timestamp: str | None) -> None:
        if not self.settings.render_console or self.live is None:
            return
        now = time.time()
        if now - self.last_render < self.settings.snapshot_interval_sec:
            return
        self.last_render = now
        snapshot = self.get_dashboard_snapshot(
            levels=self.settings.top_levels_to_display,
            trades=self.settings.recent_trades_to_display,
            spread_points=self.settings.dashboard_history_points,
        )
        self.live.update(build_dashboard(snapshot), refresh=True)

    def get_dashboard_snapshot(
        self, levels: int = 10, trades: int = 30, spread_points: int = 1500
    ) -> dict:
        """Snapshot complet pour les interfaces locales."""
        with self.state_lock:
            book = self.book
            mid = book.mid_price()
            mtm = self.strategy.mark_to_market(mid)
            spread_summaries = self.spread_tracker.summarize()
            top = book.top_levels(levels)
            spread_rows = self.spread_tracker.timeline_rows[-spread_points:]
            micro_rows = self.spread_tracker.microstructure_rows[-spread_points:]
            executions = self.strategy.executions[-trades:]
            risk_status = self.risk_manager.status(
                mtm["equity"], self.strategy.position_btc, mid
            )
            return {
                "product_id": self.settings.product_id,
                "connection_status": self.connection_status,
                "last_error": self.last_error,
                "bandit_enabled": self.settings.use_contextual_bandit,
                "best_bid": book.best_bid(),
                "best_ask": book.best_ask(),
                "mid_price": mid,
                "bids": [
                    {"price": price, "quantity": qty} for price, qty in top["bids"]
                ],
                "asks": [
                    {"price": price, "quantity": qty} for price, qty in top["asks"]
                ],
                "spread_summaries": {
                    size: {
                        "average": summary.average,
                        "median": summary.median,
                        "minimum": summary.minimum,
                        "maximum": summary.maximum,
                        "observations": summary.observations,
                    }
                    for size, summary in spread_summaries.items()
                },
                "recent_trades": list(self.recent_trades)[:trades],
                "executions": executions[::-1],
                "current_quotes": {
                    "bid": (
                        self.strategy.current_bid.price
                        if self.strategy.current_bid
                        else None
                    ),
                    "ask": (
                        self.strategy.current_ask.price
                        if self.strategy.current_ask
                        else None
                    ),
                    "bid_remaining": (
                        self.strategy.current_bid.size
                        if self.strategy.current_bid
                        else None
                    ),
                    "ask_remaining": (
                        self.strategy.current_ask.size
                        if self.strategy.current_ask
                        else None
                    ),
                    "size": self.strategy.last_quote_context.get("quote_size_btc"),
                },
                "portfolio": mtm,
                "spread_history": list(spread_rows),
                "microstructure_history": list(micro_rows),
                "microstructure": self.spread_tracker.latest_microstructure(),
                "risk": risk_status,
                "quote_context": dict(self.strategy.last_quote_context),
                "stats": {
                    "messages_received": self.messages_received,
                    "trades_processed": self.trades_processed,
                    "fills_count": len(self.strategy.executions),
                    "last_message_age_ms": (
                        None
                        if self.last_message_time is None
                        else max(0.0, (time.time() - self.last_message_time) * 1000)
                    ),
                },
            }

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.connection_status = "connecting"
        if self.settings.render_console:
            self.live = Live(
                build_placeholder("Connexion au flux Coinbase..."),
                screen=True,
                auto_refresh=False,
                transient=False,
            )
            self.live.start(refresh=True)
        self.ws = websocket.WebSocketApp(
            self.settings.ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        try:
            self.ws.run_forever(ping_interval=20, ping_timeout=10)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Flush et exports finaux en CSV + graphes."""
        # Flush final du recorder de latence si actif.
        if self.latency_recorder.enabled:
            try:
                self.latency_recorder.flush(
                    self.settings.output_dir / "bench" / "latency.csv"
                )
            except Exception as e:
                print(f"Flush bench latency a échoué : {e}")

        # On flush puis on ferme les writers async.
        for w in (
            self.book_writer,
            self.trade_writer,
            self.fill_writer,
            self.pnl_writer,
            self.state_writer,
        ):
            try:
                w.flush()
                w.close()
            except Exception as e:
                print(f"Writer flush/close a échoué : {e}")
        self.spread_tracker.export_csv(
            self.settings.output_dir / "analytics" / "spread_history.csv"
        )
        self.spread_tracker.export_microstructure_csv(
            self.settings.output_dir / "analytics" / "microstructure_history.csv"
        )
        try:
            self.spread_tracker.plot(
                self.settings.output_dir / "plots" / "spread_history.png"
            )
            self.spread_tracker.plot_microstructure(
                self.settings.output_dir / "plots" / "microstructure.png"
            )
        except Exception as e:
            print(f"Graphes non générés : {e}")
        if self.live is not None:
            self.live.stop()
            self.live = None

"""Backtest framework avec reconstruction du carnet d'ordres.

Ce module :

1. Reconstruit dynamiquement un OrderBook à partir des snapshots de
   `raw/book.csv` (top N niveaux par timestamp).
2. Rejoue les trades de `raw/trades.csv` dans l'ordre chronologique en
   les interleavant avec les mises à jour du carnet.
3. Calcule les signaux de microstructure (OFI, VPIN, trade-flow) en
   direct.
4. Simule les fills avec un modèle de file d'attente réaliste
   (cf. `fill_model.QueuePositionTracker`).
5. Produit des métriques de performance et de market making :
   P&L final, drawdown maximum, Calmar, fill rate réel vs naïf,
   queue drag, edge moyen et pondéré, spread capturé, temps de
   détention moyen, P&L par round-trip, turnover.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.analytics import SpreadTracker
from ..ui.config import Settings
from ..core.models import Trade
from ..core.orderbook import OrderBook
from ..core.risk import RiskManager
from ..core.strategy import MarketMaker
from ..core.utils import parse_timestamp
from .fill_model import QueuePositionTracker


# =====================================================================
# Lecture CSV
# =====================================================================
def _read_csv(path: Path) -> pd.DataFrame:
    if path.is_file() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


# =====================================================================
# Reconstruction d'un OrderBook depuis book.csv
# =====================================================================
def _book_snapshot_to_event(rows: pd.DataFrame) -> dict:
    """Convertit un groupe de lignes (un même timestamp) en event 'snapshot'."""
    updates = []
    for _, r in rows.iterrows():
        updates.append(
            {
                "side": str(r["side"]),
                "price_level": str(r["price"]),
                "new_quantity": str(r["quantity"]),
            }
        )
    return {"type": "snapshot", "updates": updates}


def iter_replay_events(book_df: pd.DataFrame, trades_df: pd.DataFrame):
    """Itère sur les événements mergés (book snapshots + trades) dans l'ordre.

    Yield tuples (ts: pd.Timestamp, kind: str, payload: dict|Trade).
    kind ∈ {'book', 'trade'}.
    """
    book_df = book_df.copy()
    trades_df = trades_df.copy()
    if not book_df.empty:
        book_df["ts"] = pd.to_datetime(
            book_df["timestamp"], utc=True, errors="coerce", format="ISO8601"
        )
        book_df = book_df.dropna(subset=["ts"]).sort_values(["ts", "side", "rank"])
    if not trades_df.empty:
        trades_df["ts"] = pd.to_datetime(
            trades_df["time"], utc=True, errors="coerce", format="ISO8601"
        )
        trades_df = trades_df.dropna(subset=["ts"]).sort_values("ts")

    # Grouper les snapshots book par timestamp.
    book_snapshots = []
    if not book_df.empty:
        for ts, grp in book_df.groupby("ts", sort=True):
            book_snapshots.append((ts, grp))

    trade_rows = list(trades_df.itertuples(index=False)) if not trades_df.empty else []

    # Merge : on itère sur les deux en parallèle par timestamp.
    i_b = 0
    i_t = 0
    while i_b < len(book_snapshots) or i_t < len(trade_rows):
        next_b = book_snapshots[i_b][0] if i_b < len(book_snapshots) else None
        next_t = trade_rows[i_t].ts if i_t < len(trade_rows) else None
        if next_t is None or (next_b is not None and next_b <= next_t):
            ts, grp = book_snapshots[i_b]
            yield ts, "book", _book_snapshot_to_event(grp)
            i_b += 1
        else:
            r = trade_rows[i_t]
            trade = Trade(
                trade_id=str(r.trade_id),
                product_id=str(r.product_id),
                price=float(r.price),
                size=float(r.size),
                side=str(r.side).upper(),
                time=parse_timestamp(str(r.time)),
            )
            yield r.ts, "trade", trade
            i_t += 1


# =====================================================================
# Construction d'une stratégie avec overrides
# =====================================================================
def _build_strategy(settings: Settings, overrides: dict | None = None) -> MarketMaker:
    params = asdict(settings)
    if overrides:
        params.update(overrides)
    risk = RiskManager(
        max_notional_usd=params["max_notional_usd"],
        max_loss_usd=params["max_loss_usd"],
        initial_equity=params["initial_cash_usd"],
        reduce_only_loss_utilization=params["reduce_only_loss_utilization"],
        reduce_only_exposure_utilization=params["reduce_only_exposure_utilization"],
    )
    return MarketMaker(
        quote_size_btc=params["quote_size_btc"],
        min_quote_size_btc=params["min_quote_size_btc"],
        base_half_spread_bps=params["base_half_spread_bps"],
        min_half_spread_bps=params["min_half_spread_bps"],
        max_half_spread_bps=params["max_half_spread_bps"],
        inventory_skew_bps_per_btc=params["inventory_skew_bps_per_btc"],
        risk_manager=risk,
        volatility_window=params["volatility_window"],
        volatility_spread_multiplier=params["volatility_spread_multiplier"],
        initial_cash_usd=params["initial_cash_usd"],
        touch_join_threshold_bps=params["touch_join_threshold_bps"],
        queue_ahead_factor=params["queue_ahead_factor"],
        fill_intensity=params["fill_intensity"],
        cooldown_ms_after_fill=params["cooldown_ms_after_fill"],
        quote_refresh_interval_ms=0,  # pas de throttle en backtest
        microprice_weight=params["microprice_weight"],
        imbalance_shift_bps=params["imbalance_shift_bps"],
        imbalance_widening_bps=params["imbalance_widening_bps"],
        use_contextual_bandit=params["use_contextual_bandit"],
        ewma_spread_alpha=params["ewma_spread_alpha"],
        rolling_spread_window=params["rolling_spread_window"],
        combined_signal_weight=params["combined_signal_weight"],
        vpin_reference=params["vpin_reference"],
        vpin_widening_bps_per_unit=params["vpin_widening_bps_per_unit"],
        fast_vol_widening_coef=params["fast_vol_widening_coef"],
        ofi_aggressive_join_threshold=params["ofi_aggressive_join_threshold"],
    )


# =====================================================================
# Replay avec reconstruction du carnet
# =====================================================================
def replay_with_book(
    events,
    settings: Settings,
    overrides: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Rejoue les événements (book + trades) en reconstruisant le carnet.

    Le modèle de fill utilise un ``QueuePositionTracker`` pour être plus
    réaliste que la simulation live : un trade consomme d'abord la file
    d'attente devant nous, le reliquat seulement peut nous toucher. Quand
    notre quote change de prix, on reset à la position de fin de queue.

    `events` est un itérateur de tuples (ts, kind, payload).
    Retourne (equity_curve_df, metrics_dict).
    """
    book = OrderBook()
    tracker = SpreadTracker(
        sizes=[0.1, 1.0, 5.0, 10.0],
        ofi_ewma_alpha=settings.ofi_ewma_alpha,
        trade_flow_window_ms=settings.trade_flow_window_ms,
        vpin_bucket_size=settings.vpin_bucket_size_btc,
        vpin_history_size=settings.vpin_history_size,
        fast_vol_alpha=settings.fast_vol_alpha,
    )
    mm = _build_strategy(settings, overrides)
    queue = QueuePositionTracker()

    equity_rows: list[dict] = []
    fill_events: list[dict] = []  # infos riches par fill pour les métriques MM
    trades_encountered = 0
    trades_touching_quote = (
        0  # le prix nous aurait touché (sans tenir compte de la queue)
    )
    trades_filling_us = 0  # nous avons vraiment été rempli (au moins en partie)
    reduce_only_ticks = 0
    total_ticks = 0

    for ts, kind, payload in events:
        if kind == "book":
            book.apply_coinbase_event(payload)
            mid = book.mid_price()
            if mid is None:
                continue
            tracker.update(
                ts.isoformat() if hasattr(ts, "isoformat") else str(ts), book
            )
            micro = tracker.latest_microstructure()
            mm.update_quotes(
                mid_price=mid,
                best_bid=book.best_bid(),
                best_ask=book.best_ask(),
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
            # Sync queue tracker avec la nouvelle quote. Règles :
            # - on quote STRICTEMENT MIEUX que le best (price > best_bid
            #   côté bid, ou < best_ask côté ask) → on crée un nouveau
            #   niveau ou on est devant toute la liquidité observée →
            #   queue_ahead = 0 ;
            # - on quote AU MÊME PRIX que le best → on arrive en queue
            #   derrière la liquidité déjà affichée → queue_ahead = touch_depth ;
            # - on quote STRICTEMENT MOINS BON que le best → peu probable
            #   en MM (ordre non-compétitif) mais on met une estimation
            #   très pessimiste (queue_ahead = touch_depth + une taille
            #   supposée au niveau) — dans la pratique on ne se fait
            #   presque jamais remplir dans ce cas.
            best_bid = book.best_bid()
            best_ask = book.best_ask()
            if mm.current_bid is not None and best_bid is not None:
                if mm.current_bid.price > best_bid + 1e-9:
                    queue_ahead_bid = 0.0
                elif abs(mm.current_bid.price - best_bid) <= 1e-9:
                    queue_ahead_bid = book.touch_depth("bid") or 0.0
                else:
                    # Quote plus basse que best — on est derrière toute
                    # la liquidité déjà présente au best ET du niveau
                    # intermédiaire, si existant. Approximation raisonnable.
                    queue_ahead_bid = (book.touch_depth("bid") or 0.0) * 2.0
                queue.sync_quote(
                    "bid", mm.current_bid.price, mm.current_bid.size, queue_ahead_bid
                )
            else:
                queue.sync_quote("bid", None, 0.0, 0.0)

            if mm.current_ask is not None and best_ask is not None:
                if mm.current_ask.price < best_ask - 1e-9:
                    queue_ahead_ask = 0.0
                elif abs(mm.current_ask.price - best_ask) <= 1e-9:
                    queue_ahead_ask = book.touch_depth("ask") or 0.0
                else:
                    queue_ahead_ask = (book.touch_depth("ask") or 0.0) * 2.0
                queue.sync_quote(
                    "ask", mm.current_ask.price, mm.current_ask.size, queue_ahead_ask
                )
            else:
                queue.sync_quote("ask", None, 0.0, 0.0)

            mtm = mm.mark_to_market(mid)
            total_ticks += 1
            if bool(mm.last_quote_context.get("reduce_only", False)):
                reduce_only_ticks += 1
            equity_rows.append(
                {
                    "timestamp": ts,
                    "mid_price": mid,
                    "equity": mtm["equity"],
                    "position_btc": mtm["position_btc"],
                    "realized_pnl": mtm["realized_pnl"],
                    "unrealized_pnl": mtm["unrealized_pnl"],
                    "half_spread_bps": mm.last_quote_context.get("half_spread_bps"),
                    "bandit_arm": mm.last_quote_context.get("bandit_arm"),
                    "reduce_only": bool(
                        mm.last_quote_context.get("reduce_only", False)
                    ),
                }
            )
        else:  # trade
            trade: Trade = payload
            trades_encountered += 1
            mid = book.mid_price()

            # Convention Coinbase : `side` = côté du MAKER. L'agresseur = l'inverse.
            # - maker BUY  → agresseur SELL → consomme bids → touche NOTRE bid
            # - maker SELL → agresseur BUY  → consomme asks → touche NOTRE ask
            aggressor = "SELL" if trade.side == "BUY" else "BUY"

            # Check éligibilité "prix" (aurait touché nos quotes sans tenir
            # compte de la queue — statistique brute pour diag).
            if aggressor == "SELL" and mm.current_bid is not None:
                if mm.current_bid.price >= trade.price - 1e-9:
                    trades_touching_quote += 1
            elif aggressor == "BUY" and mm.current_ask is not None:
                if mm.current_ask.price <= trade.price + 1e-9:
                    trades_touching_quote += 1

            # Mise à jour des signaux VPIN / trade-flow (côté maker Coinbase).
            tracker.on_trade(
                ts_ms=trade.time.timestamp() * 1000.0,
                side=trade.side,
                size=trade.size,
            )

            # Modèle de queue : le trade consomme d'abord la file devant
            # nous, on est rempli sur le reliquat.
            filled_qty = queue.on_trade(aggressor, trade.price, trade.size)
            if filled_qty > 1e-12:
                # Injecte le fill dans la stratégie (P&L, position, cash).
                # On respecte le cooldown post-fill comme le code live.
                now_ms = trade.time.timestamp() * 1000.0
                if now_ms - mm.last_fill_ts_ms >= mm.cooldown_ms_after_fill:
                    # Vérifie le risque avant d'exécuter.
                    if aggressor == "SELL" and mm.current_bid is not None:
                        if mm.risk_manager.can_execute_fill(
                            mm.position_btc,
                            filled_qty,
                            mm.current_bid.price,
                            mm.last_equity,
                        ):
                            fill = mm._execute_buy(trade, filled_qty)
                            mm.last_fill_ts_ms = now_ms
                            fill_events.append(_fill_event(fill, mid, mm, trade))
                            trades_filling_us += 1
                            # Notre taille a changé → sync queue tracker.
                            queue.bid.my_size_btc = (
                                mm.current_bid.size if mm.current_bid else 0.0
                            )
                    elif aggressor == "BUY" and mm.current_ask is not None:
                        if mm.risk_manager.can_execute_fill(
                            mm.position_btc,
                            -filled_qty,
                            mm.current_ask.price,
                            mm.last_equity,
                        ):
                            fill = mm._execute_sell(trade, filled_qty)
                            mm.last_fill_ts_ms = now_ms
                            fill_events.append(_fill_event(fill, mid, mm, trade))
                            trades_filling_us += 1
                            queue.ask.my_size_btc = (
                                mm.current_ask.size if mm.current_ask else 0.0
                            )

    curve = pd.DataFrame(equity_rows)
    metrics = compute_risk_metrics(
        curve,
        settings,
        fills=mm.executions,
        fill_events=fill_events,
        trades_encountered=trades_encountered,
        trades_touching_quote=trades_touching_quote,
        trades_filling_us=trades_filling_us,
        reduce_only_ticks=reduce_only_ticks,
        total_ticks=total_ticks,
    )
    return curve, metrics


def _fill_event(fill, mid: float | None, mm, trade) -> dict:
    """Crée une ligne riche pour un fill, utilisée par les métriques MM.

    On capture le mid au moment du fill pour pouvoir mesurer le spread
    capturé, ainsi que l'inventaire résultant.
    """
    edge_bps = 0.0
    if mid and mid > 0:
        if fill.side == "BUY":
            edge_bps = (mid - fill.price) / mid * 10_000.0
        else:
            edge_bps = (fill.price - mid) / mid * 10_000.0
    return {
        "time": fill.time.isoformat(),
        "side": fill.side,
        "price": fill.price,
        "size": fill.size,
        "mid_at_fill": mid,
        "edge_bps": edge_bps,
        "position_after": mm.position_btc,
        "realized_after": mm.realized_pnl,
    }


# =====================================================================
# Helpers métriques
# =====================================================================
def _observation_seconds(curve: pd.DataFrame) -> float:
    """Durée totale observée en secondes."""
    if curve.empty or "timestamp" not in curve.columns:
        return 0.0
    ts = pd.to_datetime(
        curve["timestamp"], utc=True, errors="coerce", format="ISO8601"
    ).dropna()
    if len(ts) < 2:
        return 0.0
    return max((ts.iloc[-1] - ts.iloc[0]).total_seconds(), 0.0)


# =====================================================================
# Métriques market-making spécifiques
# =====================================================================
def _mm_metrics_from_fill_events(
    fill_events: list[dict], observation_seconds: float
) -> dict:
    """Calcule les métriques propres au market making à partir des fills.

    - ``avg_edge_bps`` : distance moyenne du fill au mid au moment du fill.
      Pour un MM, on veut être rempli à un prix _meilleur_ que le mid
      (edge > 0). Un edge moyen négatif est un signal d'alerte.
    - ``spread_captured_usd`` : somme de ``edge_bps × notional_fill / 10_000``.
      C'est l'avantage tarifaire cumulé capturé par le MM au moment des fills
      (à distinguer du P&L, qui dépend aussi du chemin du mid après coup).
    - ``inventory_turnover_btc`` : somme des tailles de fills absolues.
      Représente le volume tradé total.
    - ``fills_per_hour`` : cadence des fills, indépendante de la taille.
    - ``avg_holding_time_sec`` : temps moyen entre l'ouverture et la
      clôture FIFO d'un lot de position.
    - ``round_trips`` : nombre de paires buy/sell (ou sell/buy) appariées.
    """
    out = {
        "avg_edge_bps": 0.0,
        "edge_weighted_bps": 0.0,
        "spread_captured_usd": 0.0,
        "inventory_turnover_btc": 0.0,
        "fills_per_hour": 0.0,
        "avg_holding_time_sec": 0.0,
        "round_trips": 0,
        "realized_per_round_trip_usd": 0.0,
    }
    if not fill_events:
        return out

    edges = [float(f.get("edge_bps", 0.0)) for f in fill_events]
    sizes = [float(f.get("size", 0.0)) for f in fill_events]
    prices = [float(f.get("price", 0.0)) for f in fill_events]

    out["avg_edge_bps"] = float(np.mean(edges)) if edges else 0.0
    # Edge pondéré par le notional : plus représentatif parce qu'un gros fill
    # avec edge faible pèse plus dans le P&L qu'un petit fill avec edge fort.
    total_notional = sum(p * s for p, s in zip(prices, sizes))
    if total_notional > 0:
        weighted_edge = (
            sum(e * p * s for e, p, s in zip(edges, prices, sizes)) / total_notional
        )
        out["edge_weighted_bps"] = float(weighted_edge)
    # Edge × notional → USD capturés (en théorie, à la marge du fill).
    captured = [e * p * s / 10_000.0 for e, p, s in zip(edges, prices, sizes)]
    out["spread_captured_usd"] = float(sum(captured))
    out["inventory_turnover_btc"] = float(sum(sizes))

    if observation_seconds > 0:
        out["fills_per_hour"] = len(fill_events) * 3600.0 / observation_seconds

    # Apparie les fills opposés en FIFO pour mesurer, sur chaque paire
    # (ouverture, clôture), le temps écoulé et le P&L réalisé.
    long_queue: list[tuple[pd.Timestamp, float, float]] = []  # (t, size, price)
    short_queue: list[tuple[pd.Timestamp, float, float]] = []
    holding_times_sec: list[float] = []
    round_trip_pnls: list[float] = []

    for f in fill_events:
        side = str(f.get("side", "")).upper()
        size = float(f.get("size", 0.0))
        price = float(f.get("price", 0.0))
        t = pd.to_datetime(f.get("time"), utc=True, errors="coerce")
        if pd.isna(t) or size <= 0:
            continue
        if side == "BUY":
            # Clôture des shorts puis ouverture de long.
            while size > 1e-12 and short_queue:
                t_open, s_open, p_open = short_queue[0]
                close = min(size, s_open)
                holding_times_sec.append((t - t_open).total_seconds())
                # Round-trip : short ouvert à ``p_open`` refermé à
                # ``price`` → gain réalisé = (p_open - price) * close.
                round_trip_pnls.append((p_open - price) * close)
                size -= close
                if close >= s_open - 1e-12:
                    short_queue.pop(0)
                else:
                    short_queue[0] = (t_open, s_open - close, p_open)
            if size > 1e-12:
                long_queue.append((t, size, price))
        else:  # SELL
            while size > 1e-12 and long_queue:
                t_open, s_open, p_open = long_queue[0]
                close = min(size, s_open)
                holding_times_sec.append((t - t_open).total_seconds())
                # Round-trip P&L : long ouvert à p_open, refermé à price
                # → gagne (price - p_open) × close.
                round_trip_pnls.append((price - p_open) * close)
                size -= close
                if close >= s_open - 1e-12:
                    long_queue.pop(0)
                else:
                    long_queue[0] = (t_open, s_open - close, p_open)
            if size > 1e-12:
                short_queue.append((t, size, price))

    out["round_trips"] = len(round_trip_pnls)
    if holding_times_sec:
        out["avg_holding_time_sec"] = float(np.mean(holding_times_sec))
    if round_trip_pnls:
        out["realized_per_round_trip_usd"] = float(np.mean(round_trip_pnls))
    return out


# =====================================================================
# Métriques de risque
# =====================================================================
def compute_risk_metrics(
    curve: pd.DataFrame,
    settings: Settings,
    fills: list[dict],
    fill_events: list[dict] | None = None,
    trades_encountered: int = 0,
    trades_touching_quote: int = 0,
    trades_filling_us: int = 0,
    reduce_only_ticks: int = 0,
    total_ticks: int = 0,
    trades_eligible_for_fill: int | None = None,
) -> dict:
    """Panel de métriques de performance et de risque, orienté market making."""
    if trades_eligible_for_fill is not None and trades_touching_quote == 0:
        trades_touching_quote = trades_eligible_for_fill

    base = {
        "final_pnl": 0.0,
        "total_return_pct": 0.0,
        "max_drawdown_usd": 0.0,
        "max_drawdown_pct": 0.0,
        "calmar_ratio": 0.0,
        "fills": len(fills),
        "trades_encountered": trades_encountered,
        "trades_touching_quote": trades_touching_quote,
        "trades_filling_us": trades_filling_us,
        "fill_rate_pct": 0.0,
        "naive_fill_rate_pct": 0.0,
        "queue_drag_pct": 0.0,
        "avg_pnl_per_fill": 0.0,
        "inventory_std_btc": 0.0,
        "time_in_loss_pct": 0.0,
        "reduce_only_pct": 0.0,
        "observation_seconds": 0.0,
        "observation_minutes": 0.0,
        "avg_edge_bps": 0.0,
        "edge_weighted_bps": 0.0,
        "spread_captured_usd": 0.0,
        "inventory_turnover_btc": 0.0,
        "fills_per_hour": 0.0,
        "avg_holding_time_sec": 0.0,
        "round_trips": 0,
        "realized_per_round_trip_usd": 0.0,
    }
    if curve.empty:
        return base

    equity = curve["equity"].to_numpy(dtype=float)
    initial = float(settings.initial_cash_usd)
    final_pnl = float(equity[-1] - initial)
    total_return_pct = 100.0 * final_pnl / initial if initial > 0 else 0.0

    running_max = pd.Series(equity).cummax()
    drawdown = pd.Series(equity) - running_max
    max_dd_usd = float(drawdown.min())
    max_dd_pct = 100.0 * max_dd_usd / initial if initial > 0 else 0.0

    observation_seconds = _observation_seconds(curve)

    calmar = 0.0
    if max_dd_pct < -1e-12:
        calmar = total_return_pct / abs(max_dd_pct)

    fill_rate_pct = 0.0
    if trades_touching_quote > 0:
        fill_rate_pct = 100.0 * trades_filling_us / trades_touching_quote
    # Sans modèle de file d'attente, tout trade qui touche en prix est
    # supposé remplir. On compare au fill rate réel pour quantifier le
    # "queue drag" : fraction des touches au prix perdues à cause de la
    # liquidité présente devant nous.
    naive_fill_rate_pct = 100.0 if trades_touching_quote > 0 else 0.0
    queue_drag_pct = max(0.0, naive_fill_rate_pct - fill_rate_pct)

    avg_pnl_per_fill = final_pnl / len(fills) if fills else 0.0

    inventory_std = float(pd.Series(curve["position_btc"]).std(ddof=0))
    time_in_loss = float((pd.Series(equity) < initial).mean() * 100.0)
    reduce_only_pct = (
        100.0 * reduce_only_ticks / total_ticks if total_ticks > 0 else 0.0
    )

    mm_extra = _mm_metrics_from_fill_events(
        fill_events or [], observation_seconds=observation_seconds
    )

    return {
        "final_pnl": final_pnl,
        "total_return_pct": total_return_pct,
        "max_drawdown_usd": max_dd_usd,
        "max_drawdown_pct": max_dd_pct,
        "calmar_ratio": calmar,
        "fills": len(fills),
        "trades_encountered": trades_encountered,
        "trades_touching_quote": trades_touching_quote,
        "trades_filling_us": trades_filling_us,
        "fill_rate_pct": fill_rate_pct,
        "naive_fill_rate_pct": naive_fill_rate_pct,
        "queue_drag_pct": queue_drag_pct,
        "avg_pnl_per_fill": avg_pnl_per_fill,
        "inventory_std_btc": inventory_std,
        "time_in_loss_pct": time_in_loss,
        "reduce_only_pct": reduce_only_pct,
        "observation_seconds": observation_seconds,
        "observation_minutes": observation_seconds / 60.0,
        **mm_extra,
    }


# =====================================================================
# Walk-forward
# =====================================================================
def walkforward_backtest(
    data_dir: Path,
    settings: Settings,
    n_folds: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward simple avec grid search sur 3 paramètres clés."""
    book_df = _read_csv(data_dir / "raw" / "book.csv")
    trades_df = _read_csv(data_dir / "raw" / "trades.csv")
    if book_df.empty or trades_df.empty:
        raise FileNotFoundError(
            f"Données manquantes : {data_dir}/raw/book.csv et/ou trades.csv."
        )

    # On découpe trades_df en n_folds ; book est filtré par fenêtre temporelle.
    trades_df = trades_df.copy()
    trades_df["ts"] = pd.to_datetime(
        trades_df["time"], utc=True, errors="coerce", format="ISO8601"
    )
    trades_df = trades_df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    book_df = book_df.copy()
    book_df["ts"] = pd.to_datetime(
        book_df["timestamp"], utc=True, errors="coerce", format="ISO8601"
    )
    book_df = book_df.dropna(subset=["ts"]).sort_values("ts")

    fold_size = len(trades_df) // n_folds
    if fold_size < 50:
        raise ValueError(
            f"Pas assez de données pour {n_folds} folds (taille fold={fold_size})."
        )

    grid = list(
        product(
            [0.8, 1.0, 1.3],  # volatility_spread_multiplier
            [4.0, 6.0, 8.0],  # inventory_skew_bps_per_btc
            [1.2, 1.5, 1.8],  # fill_intensity
        )
    )

    results: list[dict] = []
    curves: list[pd.DataFrame] = []

    for fold in range(n_folds):
        t_start = fold * fold_size
        t_end = t_start + fold_size
        fold_trades = trades_df.iloc[t_start:t_end]
        if fold_trades.empty:
            continue
        fold_start_ts = fold_trades["ts"].iloc[0]
        fold_end_ts = fold_trades["ts"].iloc[-1]
        fold_book = book_df[
            (book_df["ts"] >= fold_start_ts) & (book_df["ts"] <= fold_end_ts)
        ]
        if fold_book.empty:
            continue

        # Split train/test 60/40 à l'intérieur du fold.
        split_idx = int(len(fold_trades) * 0.6)
        train_trades = fold_trades.iloc[:split_idx]
        test_trades = fold_trades.iloc[split_idx:]
        if train_trades.empty or test_trades.empty:
            continue
        train_end_ts = train_trades["ts"].iloc[-1]
        train_book = fold_book[fold_book["ts"] <= train_end_ts]
        test_book = fold_book[fold_book["ts"] > train_end_ts]

        # Grid search sur train
        best_params = None
        best_score = -math.inf
        for vol_mult, inv_skew, fi in grid:
            overrides = {
                "volatility_spread_multiplier": vol_mult,
                "inventory_skew_bps_per_btc": inv_skew,
                "fill_intensity": fi,
            }
            events = iter_replay_events(train_book, train_trades)
            _, m = replay_with_book(events, settings, overrides)

            # Score de sélection orienté market making. On privilégie les
            # configs qui :
            # - génèrent du P&L réel (final_pnl)
            # - avec des round-trips profitables (realized_per_round_trip)
            # - sans sacrifier trop de drawdown
            # - en gardant un fill rate non-trivial (qu'on trade bien)
            rt_pnl = m.get("realized_per_round_trip_usd", 0.0)
            fill_rate = m.get("fill_rate_pct", 0.0)
            score = (
                1.0 * m["final_pnl"]
                + 50.0 * rt_pnl
                + 0.1 * fill_rate
                - 5.0 * abs(m["max_drawdown_usd"])
            )
            if score > best_score:
                best_score = score
                best_params = overrides

        # Test avec les meilleurs params
        events = iter_replay_events(test_book, test_trades)
        curve, metrics = replay_with_book(events, settings, best_params)
        if not curve.empty:
            curve = curve.copy()
            curve["fold"] = fold + 1
            curves.append(curve)
        results.append(
            {
                "fold": fold + 1,
                "train_start": train_trades["ts"].iloc[0],
                "train_end": train_trades["ts"].iloc[-1],
                "test_start": test_trades["ts"].iloc[0],
                "test_end": test_trades["ts"].iloc[-1],
                "best_volatility_spread_multiplier": best_params[
                    "volatility_spread_multiplier"
                ],
                "best_inventory_skew_bps_per_btc": best_params[
                    "inventory_skew_bps_per_btc"
                ],
                "best_fill_intensity": best_params["fill_intensity"],
                **metrics,
            }
        )

    results_df = pd.DataFrame(results)
    curve_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return results_df, curve_df


def plot_walkforward(
    results_df: pd.DataFrame, curve_df: pd.DataFrame, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not curve_df.empty:
        plt.figure(figsize=(11, 4))
        for fold, sub in curve_df.groupby("fold"):
            plt.plot(sub["timestamp"], sub["equity"], label=f"Fold {fold}")
        plt.legend()
        plt.title("Equity curve walk-forward")
        plt.xlabel("Time")
        plt.ylabel("Equity (USD)")
        plt.tight_layout()
        plt.savefig(output_dir / "walkforward_equity.png")
        plt.close()

    if not results_df.empty:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        axes[0].bar(results_df["fold"], results_df["final_pnl"])
        axes[0].set_title("Final P&L par fold (USD)")
        axes[0].set_xlabel("Fold")
        axes[1].bar(
            results_df["fold"],
            results_df.get("edge_weighted_bps", pd.Series([0] * len(results_df))),
        )
        axes[1].set_title("Edge pondéré (bps)")
        axes[1].set_xlabel("Fold")
        axes[2].bar(
            results_df["fold"],
            results_df.get(
                "realized_per_round_trip_usd", pd.Series([0] * len(results_df))
            ),
        )
        axes[2].set_title("P&L par round-trip (USD)")
        axes[2].set_xlabel("Fold")
        axes[3].bar(results_df["fold"], results_df["fill_rate_pct"])
        axes[3].set_title("Fill rate réel (%)")
        axes[3].set_xlabel("Fold")
        plt.tight_layout()
        plt.savefig(output_dir / "walkforward_metrics.png")
        plt.close()


# =====================================================================
# CLI
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest walk-forward avec reconstruction du carnet depuis "
            "book.csv et panel de métriques de risque complet."
        )
    )
    parser.add_argument("--data-dir", default="data", help="Racine CSV live.")
    parser.add_argument("--n-folds", type=int, default=3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    settings = Settings(output_dir=data_dir)
    results_df, curve_df = walkforward_backtest(data_dir, settings, args.n_folds)
    output_dir = data_dir / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "walkforward_results.csv", index=False)
    if not curve_df.empty:
        curve_df.to_csv(output_dir / "walkforward_curve.csv", index=False)
    plot_walkforward(results_df, curve_df, output_dir)
    if not results_df.empty:
        perf_cols = [
            "fold",
            "final_pnl",
            "max_drawdown_usd",
            "observation_minutes",
            "fills",
            "round_trips",
        ]
        mm_cols = [
            "fold",
            "fill_rate_pct",
            "queue_drag_pct",
            "avg_edge_bps",
            "edge_weighted_bps",
            "spread_captured_usd",
            "realized_per_round_trip_usd",
            "avg_holding_time_sec",
            "fills_per_hour",
            "inventory_turnover_btc",
        ]
        perf_cols = [c for c in perf_cols if c in results_df.columns]
        mm_cols = [c for c in mm_cols if c in results_df.columns]

        print("\n=== Performance ===")
        print(results_df[perf_cols].to_string(index=False))
        print("\n=== Métriques market making ===")
        print(results_df[mm_cols].to_string(index=False))
    else:
        print("Aucun fold calculé.")


if __name__ == "__main__":
    main()

"""Backtest framework avec reconstruction du carnet d'ordres.

Ce module :

1. Reconstruit dynamiquement un OrderBook au cours du replay à partir des
   snapshots `raw/book.csv` (top N niveaux par timestamp),
2. Rejoue les trades `raw/trades.csv` dans l'ordre chronologique, en les
   interleavant avec les mises à jour du carnet,
3. Calcule les signaux microstructure (OFI, VPIN, trade-flow) en direct,
4. Produit un panel complet de métriques de risque pour la stratégie.

Métriques produites :
- final_pnl (USD)
- total_return (%)
- max_drawdown (USD et %)
- sharpe_ratio (annualisé, returns par minute)
- sortino_ratio (annualisé)
- calmar_ratio
- fills, fill_rate (fills / trades rencontrés éligibles)
- hit_ratio (% de fills profitables sur clôture)
- avg_pnl_per_fill
- inventory_std (volatilité de la position BTC)
- time_in_loss_pct (% du temps en drawdown)
- reduce_only_activation_pct
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

    equity_rows: list[dict] = []
    trades_encountered = 0
    trades_eligible_for_fill = 0  # prix touche nos quotes
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
            best_bid = book.best_bid()
            best_ask = book.best_ask()
            mid = book.mid_price()
            # Vérifier si le trade est éligible pour nous toucher.
            if (
                mm.current_bid is not None
                and trade.side == "BUY"
                and best_bid is not None
            ):
                if mm.current_bid.price >= trade.price - 1e-9:
                    trades_eligible_for_fill += 1
            elif (
                mm.current_ask is not None
                and trade.side == "SELL"
                and best_ask is not None
            ):
                if mm.current_ask.price <= trade.price + 1e-9:
                    trades_eligible_for_fill += 1

            # Mise à jour des signaux VPIN / trade-flow.
            tracker.on_trade(
                ts_ms=trade.time.timestamp() * 1000.0,
                side=trade.side,
                size=trade.size,
            )

            mm.on_trade(
                trade,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_touch_depth=(
                    book.touch_depth("bid") if best_bid is not None else 0.0
                ),
                ask_touch_depth=(
                    book.touch_depth("ask") if best_ask is not None else 0.0
                ),
                mid_price=mid,
            )

    curve = pd.DataFrame(equity_rows)
    metrics = compute_risk_metrics(
        curve,
        settings,
        fills=mm.executions,
        trades_encountered=trades_encountered,
        trades_eligible_for_fill=trades_eligible_for_fill,
        reduce_only_ticks=reduce_only_ticks,
        total_ticks=total_ticks,
    )
    return curve, metrics


# =====================================================================
# Helpers métriques
# =====================================================================
def _build_minute_equity_returns(curve: pd.DataFrame) -> tuple[pd.Series, float]:
    """Construit les returns minute à partir de la courbe d'equity.

    Retourne :
    - returns_per_min : série de returns par minute
    - observation_minutes : durée totale observée en minutes
    """
    if curve.empty or "timestamp" not in curve.columns or "equity" not in curve.columns:
        return pd.Series(dtype=float), 0.0

    ts = pd.to_datetime(curve["timestamp"], utc=True, errors="coerce", format="ISO8601")
    tmp = pd.DataFrame({"ts": ts, "equity": curve["equity"]}).dropna(subset=["ts"])
    if tmp.empty:
        return pd.Series(dtype=float), 0.0

    tmp = tmp.sort_values("ts").drop_duplicates(subset="ts", keep="last")
    if len(tmp) < 2:
        return pd.Series(dtype=float), 0.0

    start_ts = tmp["ts"].iloc[0]
    end_ts = tmp["ts"].iloc[-1]
    observation_minutes = max(
        (end_ts - start_ts).total_seconds() / 60.0,
        0.0,
    )

    equity_1m = tmp.set_index("ts")["equity"].resample("1min").last().ffill()

    returns_per_min = equity_1m.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return returns_per_min, observation_minutes


def _annualized_sharpe_any_length(
    returns_per_period: pd.Series,
    periods_per_year: float,
) -> float:
    """Sharpe annualisé calculable même sur petit échantillon.

    Retourne 0.0 si :
    - moins de 2 points
    - volatilité nulle ou quasi nulle
    """
    if returns_per_period is None:
        return 0.0

    r = pd.Series(returns_per_period, dtype=float)
    r = r.replace([np.inf, -np.inf], np.nan).dropna()

    if len(r) < 2:
        return 0.0

    mean_r = float(r.mean())
    std_r = float(r.std(ddof=0))
    if not np.isfinite(std_r) or std_r <= 1e-12:
        return 0.0

    sharpe = mean_r / std_r * math.sqrt(periods_per_year)
    if not np.isfinite(sharpe):
        return 0.0
    return float(sharpe)


def _annualized_sortino_any_length(
    returns_per_period: pd.Series,
    periods_per_year: float,
    target_return: float = 0.0,
) -> float:
    """Sortino annualisé calculable même sur petit échantillon.

    On utilise la downside deviation définie sur TOUS les points :
        downside_dev = sqrt(mean(min(r - target, 0)^2))

    Cela évite d'exiger un nombre minimal de returns négatifs.
    """
    if returns_per_period is None:
        return 0.0

    r = pd.Series(returns_per_period, dtype=float)
    r = r.replace([np.inf, -np.inf], np.nan).dropna()

    if len(r) < 2:
        return 0.0

    excess = r - target_return
    downside = np.minimum(excess, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))

    if not np.isfinite(downside_dev) or downside_dev <= 1e-12:
        return 0.0

    mean_excess = float(excess.mean())
    sortino = mean_excess / downside_dev * math.sqrt(periods_per_year)
    if not np.isfinite(sortino):
        return 0.0
    return float(sortino)


# =====================================================================
# Métriques de risque
# =====================================================================
def compute_risk_metrics(
    curve: pd.DataFrame,
    settings: Settings,
    fills: list[dict],
    trades_encountered: int,
    trades_eligible_for_fill: int,
    reduce_only_ticks: int,
    total_ticks: int,
) -> dict:
    """Panel complet de métriques de risque/performance."""
    base = {
        "final_pnl": 0.0,
        "total_return_pct": 0.0,
        "max_drawdown_usd": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "calmar_ratio": 0.0,
        "fills": len(fills),
        "trades_encountered": trades_encountered,
        "trades_eligible_for_fill": trades_eligible_for_fill,
        "fill_rate_pct": 0.0,
        "hit_ratio_pct": 0.0,
        "avg_pnl_per_fill": 0.0,
        "inventory_std_btc": 0.0,
        "time_in_loss_pct": 0.0,
        "reduce_only_pct": 0.0,
        "observation_minutes": 0.0,
        "return_points": 0,
        "nonzero_return_points": 0,
        "metrics_reliability": "none",
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

    # Returns par minute sur equity resamplée proprement.
    returns_per_min, observation_minutes = _build_minute_equity_returns(curve)

    # Crypto 24/7 : 525600 minutes par an.
    periods_per_year = 525600.0

    # Calcul même sur petit fold.
    sharpe = _annualized_sharpe_any_length(
        returns_per_min,
        periods_per_year=periods_per_year,
    )

    sortino = _annualized_sortino_any_length(
        returns_per_min,
        periods_per_year=periods_per_year,
        target_return=0.0,
    )

    calmar = 0.0
    if max_dd_pct < -1e-12:
        calmar = total_return_pct / abs(max_dd_pct)

    # Fill rate : fills effectifs / trades qui auraient pu nous toucher.
    fill_rate_pct = 0.0
    if trades_eligible_for_fill > 0:
        fill_rate_pct = 100.0 * len(fills) / trades_eligible_for_fill

    # Hit ratio : % de fills qui ont été clôturés en profit.
    hit_ratio_pct = _hit_ratio_fifo(fills)
    avg_pnl_per_fill = final_pnl / len(fills) if fills else 0.0

    inventory_std = float(pd.Series(curve["position_btc"]).std(ddof=0))

    time_in_loss = float((pd.Series(equity) < initial).mean() * 100.0)
    reduce_only_pct = (
        100.0 * reduce_only_ticks / total_ticks if total_ticks > 0 else 0.0
    )

    nonzero_return_points = int((returns_per_min.abs() > 1e-15).sum())

    if len(returns_per_min) >= 60:
        metrics_reliability = "high"
    elif len(returns_per_min) >= 15:
        metrics_reliability = "medium"
    elif len(returns_per_min) >= 2:
        metrics_reliability = "low"
    else:
        metrics_reliability = "none"

    return {
        "final_pnl": final_pnl,
        "total_return_pct": total_return_pct,
        "max_drawdown_usd": max_dd_usd,
        "max_drawdown_pct": max_dd_pct,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "fills": len(fills),
        "trades_encountered": trades_encountered,
        "trades_eligible_for_fill": trades_eligible_for_fill,
        "fill_rate_pct": fill_rate_pct,
        "hit_ratio_pct": hit_ratio_pct,
        "avg_pnl_per_fill": avg_pnl_per_fill,
        "inventory_std_btc": inventory_std,
        "time_in_loss_pct": time_in_loss,
        "reduce_only_pct": reduce_only_pct,
        "observation_minutes": observation_minutes,
        "return_points": int(len(returns_per_min)),
        "nonzero_return_points": nonzero_return_points,
        "metrics_reliability": metrics_reliability,
    }


def _hit_ratio_fifo(fills: list[dict]) -> float:
    """% de fills "gagnants" en appariant BUYs et SELLs FIFO.

    Approximation simple : on garde deux files (longs/shorts), quand un
    SELL arrive on consomme les BUYs antérieurs et on compte ceux dont
    le prix d'achat était < prix de vente comme gagnants (et inversement).
    """
    if not fills:
        return 0.0
    long_queue: list[tuple[float, float]] = []  # (price, remaining_size)
    short_queue: list[tuple[float, float]] = []
    wins = 0
    total_closes = 0
    for f in fills:
        side = str(f.get("side", "")).upper()
        price = float(f.get("price", 0.0))
        size = float(f.get("size", 0.0))
        if side == "BUY":
            # Clôture des shorts puis renforcement des longs.
            while size > 1e-12 and short_queue:
                sp, ss = short_queue[0]
                close = min(size, ss)
                total_closes += 1
                if sp > price:
                    wins += 1
                size -= close
                if close >= ss - 1e-12:
                    short_queue.pop(0)
                else:
                    short_queue[0] = (sp, ss - close)
            if size > 1e-12:
                long_queue.append((price, size))
        else:
            while size > 1e-12 and long_queue:
                lp, ls = long_queue[0]
                close = min(size, ls)
                total_closes += 1
                if price > lp:
                    wins += 1
                size -= close
                if close >= ls - 1e-12:
                    long_queue.pop(0)
                else:
                    long_queue[0] = (lp, ls - close)
            if size > 1e-12:
                short_queue.append((price, size))
    if total_closes == 0:
        return 0.0
    return 100.0 * wins / total_closes


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

            # Score de sélection plus robuste : on évite de sur-optimiser
            # un Sharpe absurde sur échantillon trop court.
            score = (
                0.35 * m["sharpe_ratio"]
                + 0.15 * m["sortino_ratio"]
                + 0.25 * m["total_return_pct"]
                - 0.20 * abs(m["max_drawdown_pct"])
                + 0.05 * m["fill_rate_pct"]
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
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].bar(results_df["fold"], results_df["final_pnl"])
        axes[0].set_title("Final PnL par fold (USD)")
        axes[0].set_xlabel("Fold")
        axes[1].bar(results_df["fold"], results_df["sharpe_ratio"])
        axes[1].set_title("Sharpe ratio (ann.)")
        axes[1].set_xlabel("Fold")
        axes[2].bar(results_df["fold"], results_df["fill_rate_pct"])
        axes[2].set_title("Fill rate (%)")
        axes[2].set_xlabel("Fold")
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
        # Affichage synthèse : on sélectionne les colonnes clés.
        cols = [
            "fold",
            "final_pnl",
            "total_return_pct",
            "max_drawdown_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "fills",
            "fill_rate_pct",
            "hit_ratio_pct",
            "inventory_std_btc",
            "observation_minutes",
            "return_points",
            "nonzero_return_points",
        ]
        cols = [c for c in cols if c in results_df.columns]
        print(results_df[cols].to_string(index=False))
    else:
        print("Aucun fold calculé.")


if __name__ == "__main__":
    main()

"""Analyse post-run : reconstruit le P&L depuis les fills CSV.

Permet aussi de tracer la courbe de spread à partir de spread_history.csv.

Usage :
    python -m crypto_mm.analyze --data-dir data
    python -m crypto_mm.analyze --fills data/simulation/fills.csv \
        --spreads data/analytics/spread_history.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _parse_float(x, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def summarize_fills(fills: pd.DataFrame) -> None:
    print("\n=== FILLS ===")
    if fills.empty:
        print("Aucun fill.")
        return
    buys = fills[fills["side"].str.upper() == "BUY"]
    sells = fills[fills["side"].str.upper() == "SELL"]
    print(f"Nombre de fills : {len(fills)}")
    print(f"  - Buys  : {len(buys):>5d}  volume ~ {buys['size'].abs().sum():.6f} BTC")
    print(f"  - Sells : {len(sells):>5d}  volume ~ {sells['size'].abs().sum():.6f} BTC")
    if "time" in fills.columns and not fills["time"].empty:
        print(f"Période : {fills['time'].iloc[0]}  →  {fills['time'].iloc[-1]}")


def compute_pnl_from_fills(
    fills: pd.DataFrame, max_inventory: float | None = None
) -> pd.DataFrame:
    """Reconstruit le P&L ligne par ligne à partir des fills.

    Comptabilité FIFO sur la position courante : quand on réduit, on réalise
    (price - avg_entry) * size ; quand on renforce, on met à jour avg_entry
    pondéré. Identique à la logique du MarketMaker en live.
    """
    if fills.empty:
        return pd.DataFrame()

    rows = fills.sort_values("time").to_dict(orient="records")
    q = 0.0
    ac = 0.0
    cash = 0.0
    realized = 0.0
    out: list[dict] = []

    for r in rows:
        side = str(r.get("side", "")).upper()
        price = _parse_float(r.get("price"))
        size = abs(_parse_float(r.get("size")))
        if not (math.isfinite(price) and math.isfinite(size)):
            continue

        s = +1.0 if side.startswith("B") else -1.0
        desired_q = q + s * size
        if (
            max_inventory is not None
            and math.isfinite(max_inventory)
            and max_inventory > 0
        ):
            new_q = max(-max_inventory, min(desired_q, +max_inventory))
        else:
            new_q = desired_q
        dq = new_q - q

        if abs(dq) < 1e-15:
            mark = price
            unrealized = (mark - ac) * q
            out.append(
                {
                    "time": r.get("time"),
                    "side": side,
                    "price": price,
                    "size": 0.0,
                    "position": q,
                    "avg_entry_price": ac,
                    "cash": cash,
                    "realized_pnl": realized,
                    "unrealized_pnl": unrealized,
                    "total_pnl": realized + unrealized,
                    "mark_price": mark,
                }
            )
            continue

        cash -= dq * price

        if dq > 0:  # net BUY
            if q < 0:
                close = min(-q, dq)
                realized += (ac - price) * close
                q += close
                dq -= close
                if abs(q) < 1e-12:
                    ac = 0.0
            if dq > 0:
                new_long = q + dq
                ac = (q * ac + dq * price) / new_long if new_long else 0.0
                q = new_long
        else:  # dq < 0, net SELL
            sell_qty = -dq
            if q > 0:
                close = min(q, sell_qty)
                realized += (price - ac) * close
                q -= close
                sell_qty -= close
                if abs(q) < 1e-12:
                    ac = 0.0
            if sell_qty > 0:
                new_short = (-q) + sell_qty
                ac = ((-q) * ac + sell_qty * price) / new_short if new_short else 0.0
                q -= sell_qty

        mark = price
        unrealized = (mark - ac) * q
        out.append(
            {
                "time": r.get("time"),
                "side": side,
                "price": price,
                "size": abs(dq),
                "position": q,
                "avg_entry_price": ac,
                "cash": cash,
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "total_pnl": realized + unrealized,
                "mark_price": mark,
            }
        )

    return pd.DataFrame(out)


def plot_pnl(pnl_df: pd.DataFrame, output_path: Path) -> None:
    if pnl_df.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.to_datetime(pnl_df["time"], utc=True, errors="coerce", format="ISO8601")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(times, pnl_df["total_pnl"], label="Total P&L")
    ax.plot(times, pnl_df["realized_pnl"], label="Realized P&L")
    ax.plot(times, pnl_df["unrealized_pnl"], label="Unrealized P&L")
    ax.axhline(0, linestyle=":", alpha=0.5)
    ax.legend()
    ax.set_title("P&L reconstruit depuis les fills")
    ax.set_xlabel("Time")
    ax.set_ylabel("USD")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_spreads(spreads_df: pd.DataFrame, output_path: Path) -> None:
    if spreads_df.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spreads_df = spreads_df.copy()
    spreads_df["timestamp"] = pd.to_datetime(
        spreads_df["timestamp"], utc=True, errors="coerce", format="ISO8601"
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    for size, sub in spreads_df.groupby("size_btc"):
        ax.plot(sub["timestamp"], sub["spread_abs"], label=f"{size:g} BTC")
    ax.legend(title="Taille")
    ax.set_title("Historique du spread par taille")
    ax.set_xlabel("Time")
    ax.set_ylabel("Spread absolu (USD)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse post-run : reconstruit P&L et trace les spreads depuis les CSV."
    )
    parser.add_argument("--data-dir", default="data", help="Répertoire racine des CSV.")
    parser.add_argument(
        "--fills", default=None, help="Chemin direct vers fills.csv (override)."
    )
    parser.add_argument(
        "--spreads",
        default=None,
        help="Chemin direct vers spread_history.csv (override).",
    )
    parser.add_argument(
        "--max-inventory", type=float, default=None, help="Inventaire max (optionnel)."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    fills_path = (
        Path(args.fills) if args.fills else (data_dir / "simulation" / "fills.csv")
    )
    spreads_path = (
        Path(args.spreads)
        if args.spreads
        else (data_dir / "analytics" / "spread_history.csv")
    )

    fills = pd.read_csv(fills_path) if fills_path.exists() else pd.DataFrame()
    spreads = pd.read_csv(spreads_path) if spreads_path.exists() else pd.DataFrame()

    summarize_fills(fills)

    pnl_df = compute_pnl_from_fills(fills, max_inventory=args.max_inventory)
    output_dir = data_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not pnl_df.empty:
        pnl_df.to_csv(output_dir / "pnl_from_fills.csv", index=False)
        plot_pnl(pnl_df, output_dir / "pnl_from_fills.png")
        print(f"\n[✓] P&L reconstruit : {output_dir / 'pnl_from_fills.csv'}")
    else:
        print("\n[i] Aucun P&L reconstruit (pas de fills).")

    if not spreads.empty:
        plot_spreads(spreads, output_dir / "spread_history.png")
        print(f"[✓] Graphe spread : {output_dir / 'spread_history.png'}")

    print("\nTerminé.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _read_csv_tree(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_csv(path)
    if path.is_dir():
        files = sorted(path.rglob("*.csv"))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return pd.DataFrame()


def generate_analysis_plots(data_dir: Path) -> None:
    """Génère les graphiques d'analyse à partir des CSV du run live."""
    output_dir = data_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    state_df = _read_csv_tree(data_dir / "analytics" / "state.csv")
    fills_df = _read_csv_tree(data_dir / "simulation" / "fills.csv")

    if not state_df.empty:
        state_df = state_df.copy()
        state_df["timestamp"] = pd.to_datetime(
            state_df["timestamp"], utc=True, errors="coerce", format="ISO8601"
        )
        state_df = state_df.dropna(subset=["timestamp"]).sort_values("timestamp")

        # PnL dans le temps
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(state_df["timestamp"], state_df["equity"], label="Equity")
        if "realized_pnl" in state_df.columns:
            ax.plot(
                state_df["timestamp"], state_df["realized_pnl"], label="Realized PnL"
            )
        if "unrealized_pnl" in state_df.columns:
            ax.plot(
                state_df["timestamp"],
                state_df["unrealized_pnl"],
                label="Unrealized PnL",
            )
        ax.legend()
        ax.set_title("P&L dans le temps")
        ax.set_xlabel("Time")
        ax.set_ylabel("USD")
        fig.tight_layout()
        fig.savefig(output_dir / "pnl_timeseries.png")
        plt.close(fig)

        # Inventaire
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(state_df["timestamp"], state_df["position_btc"], label="Position BTC")
        ax.set_title("Inventaire dans le temps")
        ax.set_xlabel("Time")
        ax.set_ylabel("BTC")
        fig.tight_layout()
        fig.savefig(output_dir / "inventory_timeseries.png")
        plt.close(fig)

        # Paramètres de quote
        cols = [
            c
            for c in [
                "half_spread_bps",
                "volatility_bps",
                "signal_shift_bps",
                "rolling_inside_spread_bps",
                "ewma_inside_spread_bps",
            ]
            if c in state_df.columns
        ]
        if cols:
            fig, ax = plt.subplots(figsize=(11, 4))
            for col in cols:
                ax.plot(state_df["timestamp"], state_df[col], label=col)
            ax.legend()
            ax.set_title("Paramètres de quote dans le temps")
            ax.set_xlabel("Time")
            ax.set_ylabel("bps")
            fig.tight_layout()
            fig.savefig(output_dir / "quote_parameters_timeseries.png")
            plt.close(fig)

        # Signal micro + toxicité
        micro_cols = [
            c
            for c in ["micro_signal_bps", "ofi_ewma", "vpin", "fast_vol_bps"]
            if c in state_df.columns
        ]
        if micro_cols:
            fig, ax = plt.subplots(figsize=(11, 4))
            for col in micro_cols:
                ax.plot(state_df["timestamp"], state_df[col], label=col)
            ax.legend()
            ax.set_title("Signal microstructure")
            ax.set_xlabel("Time")
            fig.tight_layout()
            fig.savefig(output_dir / "microstructure_timeseries.png")
            plt.close(fig)

        if "bandit_arm" in state_df.columns:
            arm_series = pd.to_numeric(state_df["bandit_arm"], errors="coerce").ffill()
            if arm_series.notna().any():
                fig, ax = plt.subplots(figsize=(11, 4))
                ax.plot(state_df["timestamp"], arm_series)
                ax.set_title("Arm choisi par le contextual bandit")
                ax.set_xlabel("Time")
                ax.set_ylabel("Arm")
                fig.tight_layout()
                fig.savefig(output_dir / "bandit_arm_timeseries.png")
                plt.close(fig)

    if not fills_df.empty:
        fills_df = fills_df.copy()
        fills_df["time"] = pd.to_datetime(
            fills_df["time"], utc=True, errors="coerce", format="ISO8601"
        )
        fills_df = fills_df.dropna(subset=["time"])
        if not fills_df.empty:
            fill_counts = (
                fills_df.groupby([fills_df["time"].dt.floor("1min"), "side"])
                .size()
                .unstack(fill_value=0)
            )
            fig, ax = plt.subplots(figsize=(11, 4))
            fill_counts.plot(ax=ax)
            ax.set_title("Nombre de fills par minute")
            ax.set_xlabel("Time")
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(output_dir / "fill_counts_per_minute.png")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère des graphiques d'analyse à partir des CSV."
    )
    parser.add_argument(
        "--data-dir", default="data", help="Répertoire racine des CSV live."
    )
    args = parser.parse_args()
    generate_analysis_plots(Path(args.data_dir))
    print(f"Graphiques générés dans {Path(args.data_dir) / 'plots'}")


if __name__ == "__main__":
    main()

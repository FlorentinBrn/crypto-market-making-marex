from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend non interactif, sans dépendance écran.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .storage import write_dataframe_like


@dataclass(slots=True)
class SpreadSummary:
    """KPIs agrégés par taille de clip (0.1/1/5/10 BTC)."""

    size_btc: float
    average: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    observations: int


class SpreadTracker:
    """Suit les spreads par taille + l'état microstructure complet.

    Ce module concentre *tous* les signaux de microstructure :
    - spread à la touch et VWAP par taille,
    - microprice et edge microprice / mid,
    - imbalance L1/L3 et imbalance par volume,
    - OFI instantané et OFI EWMA (Cont/Kukanov/Stoikov),
    - trade-flow imbalance (pression agresseurs),
    - VPIN-lite (toxicité),
    - vol courte (EWMA sur log-returns mid),
    - micro-signal combiné (edge de quoting recommandé en bps).

    Le micro-signal combiné est exposé à la stratégie pour décaler le fair
    price et le skew.
    """

    def __init__(
        self,
        sizes: list[float],
        maxlen: int = 50_000,
        ofi_ewma_alpha: float = 0.15,
        trade_flow_window_ms: int = 4_000,
        vpin_bucket_size: float = 0.5,
        vpin_history_size: int = 50,
        fast_vol_alpha: float = 0.10,
    ) -> None:
        self.sizes = sizes
        self.values: dict[float, deque[float]] = {
            size: deque(maxlen=maxlen) for size in sizes
        }
        self.timeline_rows: list[dict] = []
        self.microstructure_rows: list[dict] = []

        # OFI
        self.ofi_ewma_alpha = ofi_ewma_alpha
        self.ofi_ewma: float = 0.0

        # Trade flow (fenêtre glissante)
        self.trade_flow_window_ms = trade_flow_window_ms
        self._trade_flow_buffer: deque[tuple[float, str, float]] = deque()
        self._buy_volume_window: float = 0.0
        self._sell_volume_window: float = 0.0

        # VPIN-lite : buckets de volume de taille fixe (en BTC),
        # on calcule sur les derniers `vpin_history_size` buckets.
        self.vpin_bucket_size = vpin_bucket_size
        self.vpin_history_size = vpin_history_size
        self._vpin_current_bucket_buy = 0.0
        self._vpin_current_bucket_sell = 0.0
        self._vpin_current_bucket_total = 0.0
        self._vpin_buckets: deque[float] = deque(maxlen=vpin_history_size)
        self.last_vpin: float = 0.0

        # Vol rapide sur log-returns du mid
        self.fast_vol_alpha = fast_vol_alpha
        self._last_mid: float | None = None
        self._fast_vol_bps: float = 0.0

    # ------------------------------------------------------------------
    # Updates côté carnet
    # ------------------------------------------------------------------
    def update(self, timestamp: str, book) -> dict[float, SpreadSummary]:
        """Mise à jour quand un événement level2 arrive.

        Calcule toutes les métriques dépendantes du carnet et stocke une
        ligne dans la timeline microstructure.
        """
        # 1) Spreads par taille (VWAP)
        for size in self.sizes:
            metrics = book.depth_metrics(size)
            if metrics.spread_abs is not None:
                self.values[size].append(metrics.spread_abs)
                self.timeline_rows.append(
                    {
                        "timestamp": timestamp,
                        "size_btc": size,
                        "spread_abs": metrics.spread_abs,
                        "spread_bps": metrics.spread_bps,
                        "buy_vwap": metrics.buy_vwap,
                        "sell_vwap": metrics.sell_vwap,
                    }
                )

        # 2) Mid / microprice
        mid_price = book.mid_price()
        microprice = book.microprice()
        microprice_edge_bps: float | None = None
        if mid_price is not None and microprice is not None and mid_price > 0:
            microprice_edge_bps = (microprice - mid_price) / mid_price * 10_000

        # 3) Fast vol EWMA sur log-returns du mid
        if mid_price is not None and self._last_mid is not None and self._last_mid > 0:
            r = np.log(mid_price / self._last_mid) * 10_000  # bps
            alpha = self.fast_vol_alpha
            # EWMA sur le carré des returns, puis racine → vol en bps.
            self._fast_vol_bps = float(
                np.sqrt(alpha * r * r + (1.0 - alpha) * self._fast_vol_bps**2)
            )
        if mid_price is not None:
            self._last_mid = mid_price

        # 4) OFI
        ofi_event = book.compute_ofi_delta()
        a = self.ofi_ewma_alpha
        self.ofi_ewma = a * ofi_event + (1.0 - a) * self.ofi_ewma

        # 5) Imbalances
        imb_l1 = book.imbalance_by_levels(1)
        imb_l3 = book.imbalance_by_levels(3)
        imb_1btc = book.imbalance_by_volume(1.0)

        # 6) Signal combiné (edge de quoting recommandé en bps)
        combined_bps = self._combine_signals(
            microprice_edge_bps=microprice_edge_bps or 0.0,
            imbalance_l3=imb_l3,
            ofi_ewma=self.ofi_ewma,
            trade_flow=self._trade_flow_signed_ratio(),
        )

        row = {
            "timestamp": timestamp,
            "mid_price": mid_price,
            "microprice": microprice,
            "microprice_edge_bps": microprice_edge_bps,
            "imbalance_l1": imb_l1,
            "imbalance_l3": imb_l3,
            "imbalance_1btc": imb_1btc,
            "ofi_event": ofi_event,
            "ofi_ewma": self.ofi_ewma,
            "trade_flow_signed": self._trade_flow_signed_ratio(),
            "vpin": self.last_vpin,
            "fast_vol_bps": self._fast_vol_bps,
            "micro_signal_bps": combined_bps,
        }
        self.microstructure_rows.append(row)
        return self.summarize()

    # ------------------------------------------------------------------
    # Updates côté trades
    # ------------------------------------------------------------------
    def on_trade(self, ts_ms: float, side: str, size: float) -> None:
        """Mise à jour des métriques dépendantes des trades (flow + VPIN).

        `side` est la convention Coinbase market_trades : côté du *maker*.
        → si maker=BUY, l'agresseur est un SELLER (pression vendeuse).
        → si maker=SELL, l'agresseur est un BUYER (pression acheteuse).

        On ajoute le volume dans le bon compteur pour que `trade_flow` et
        VPIN reflètent bien la pression agresseur.
        """
        aggressor = "SELL" if side.upper() == "BUY" else "BUY"

        # 1) Fenêtre glissante pour le trade-flow imbalance
        self._trade_flow_buffer.append((ts_ms, aggressor, size))
        if aggressor == "BUY":
            self._buy_volume_window += size
        else:
            self._sell_volume_window += size
        self._trim_trade_flow_window(ts_ms)

        # 2) VPIN-lite : bucket de taille fixe
        self._vpin_current_bucket_total += size
        if aggressor == "BUY":
            self._vpin_current_bucket_buy += size
        else:
            self._vpin_current_bucket_sell += size

        # On découpe en plusieurs buckets si on a sauté plusieurs bornes.
        while self._vpin_current_bucket_total >= self.vpin_bucket_size:
            # On enregistre le bucket courant (|buy-sell| / total).
            num = abs(self._vpin_current_bucket_buy - self._vpin_current_bucket_sell)
            denom = max(self._vpin_current_bucket_total, 1e-12)
            self._vpin_buckets.append(num / denom)
            # Reset pour le prochain bucket, avec le résidu éventuel.
            residual = self._vpin_current_bucket_total - self.vpin_bucket_size
            # Répartir le résidu de façon simple : on garde le même ratio
            # buy/sell que le bucket en cours.
            if self._vpin_current_bucket_total > 0:
                buy_ratio = (
                    self._vpin_current_bucket_buy / self._vpin_current_bucket_total
                )
            else:
                buy_ratio = 0.5
            self._vpin_current_bucket_total = max(0.0, residual)
            self._vpin_current_bucket_buy = buy_ratio * self._vpin_current_bucket_total
            self._vpin_current_bucket_sell = (
                self._vpin_current_bucket_total - self._vpin_current_bucket_buy
            )

        if self._vpin_buckets:
            self.last_vpin = float(np.mean(self._vpin_buckets))

    def _trim_trade_flow_window(self, current_ts_ms: float) -> None:
        """Retire les trades sortis de la fenêtre temporelle."""
        cutoff = current_ts_ms - self.trade_flow_window_ms
        while self._trade_flow_buffer and self._trade_flow_buffer[0][0] < cutoff:
            _, aggressor, size = self._trade_flow_buffer.popleft()
            if aggressor == "BUY":
                self._buy_volume_window = max(0.0, self._buy_volume_window - size)
            else:
                self._sell_volume_window = max(0.0, self._sell_volume_window - size)

    def _trade_flow_signed_ratio(self) -> float:
        """Imbalance signée des agresseurs sur la fenêtre (-1..+1)."""
        b = self._buy_volume_window
        s = self._sell_volume_window
        denom = b + s
        if denom <= 0:
            return 0.0
        return (b - s) / denom

    # ------------------------------------------------------------------
    # Signal combiné
    # ------------------------------------------------------------------
    @staticmethod
    def _combine_signals(
        microprice_edge_bps: float,
        imbalance_l3: float,
        ofi_ewma: float,
        trade_flow: float,
    ) -> float:
        """Combine les signaux en un shift de fair price recommandé (bps).

        Pondérations choisies par défaut :
        - microprice edge : poids 1.0 (déjà exprimé en bps) ;
        - imbalance L3    : poids 2.0 bps par unité d'imbalance ;
        - OFI EWMA        : poids faible (0.05 bps) car l'OFI est en BTC,
                            sa magnitude dépend du marché ;
        - trade flow      : poids 1.5 bps par unité de ratio signé.

        Le résultat est clippé à ±6 bps pour éviter des signaux aberrants
        en période de liquidité anormale.
        """
        raw = (
            1.0 * microprice_edge_bps
            + 2.0 * imbalance_l3
            + 0.05 * ofi_ewma
            + 1.5 * trade_flow
        )
        return float(max(-6.0, min(6.0, raw)))

    # ------------------------------------------------------------------
    # Accesseurs
    # ------------------------------------------------------------------
    def summarize(self) -> dict[float, SpreadSummary]:
        summaries: dict[float, SpreadSummary] = {}
        for size in self.sizes:
            values = list(self.values[size])
            if values:
                series = pd.Series(values, dtype=float)
                summaries[size] = SpreadSummary(
                    size_btc=size,
                    average=float(series.mean()),
                    median=float(series.median()),
                    minimum=float(series.min()),
                    maximum=float(series.max()),
                    observations=len(values),
                )
            else:
                summaries[size] = SpreadSummary(size, None, None, None, None, 0)
        return summaries

    def latest_microstructure(self) -> dict:
        """Retourne la dernière ligne microstructure (ou des zéros si vide)."""
        if not self.microstructure_rows:
            return {
                "mid_price": None,
                "microprice": None,
                "microprice_edge_bps": None,
                "imbalance_l1": 0.0,
                "imbalance_l3": 0.0,
                "imbalance_1btc": 0.0,
                "ofi_event": 0.0,
                "ofi_ewma": 0.0,
                "trade_flow_signed": 0.0,
                "vpin": 0.0,
                "fast_vol_bps": 0.0,
                "micro_signal_bps": 0.0,
            }
        return dict(self.microstructure_rows[-1])

    # ------------------------------------------------------------------
    # Exports CSV
    # ------------------------------------------------------------------
    def export_csv(self, output_path: Path) -> None:
        write_dataframe_like(output_path, self.timeline_rows)

    def export_microstructure_csv(self, output_path: Path) -> None:
        write_dataframe_like(output_path, self.microstructure_rows)

    # ------------------------------------------------------------------
    # Graphiques
    # ------------------------------------------------------------------
    def plot(self, output_path: Path) -> None:
        if not self.timeline_rows:
            return
        df = pd.DataFrame(self.timeline_rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for size in sorted(df["size_btc"].unique()):
            subset = df[df["size_btc"] == size].copy()
            subset["timestamp"] = pd.to_datetime(
                subset["timestamp"], utc=True, format="ISO8601"
            )
            plt.figure(figsize=(10, 4))
            plt.plot(subset["timestamp"], subset["spread_abs"])
            plt.xlabel("Timestamp")
            plt.ylabel("Spread absolu (USD)")
            plt.title(f"Historique du spread - taille {size} BTC")
            plt.tight_layout()
            plt.savefig(
                output_path.parent / f"spread_{str(size).replace('.', '_')}.png"
            )
            plt.close()

    def plot_microstructure(self, output_path: Path) -> None:
        if not self.microstructure_rows:
            return
        df = pd.DataFrame(self.microstructure_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 4))
        plt.plot(df["timestamp"], df["imbalance_l1"], label="Imbalance L1")
        plt.plot(df["timestamp"], df["imbalance_l3"], label="Imbalance L3")
        plt.xlabel("Timestamp")
        plt.ylabel("Imbalance")
        plt.title("Imbalance du carnet")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path.parent / "imbalance.png")
        plt.close()

        if "microprice_edge_bps" in df.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(
                df["timestamp"], df["microprice_edge_bps"], label="Microprice edge"
            )
            plt.plot(
                df["timestamp"], df["micro_signal_bps"], label="Micro signal combiné"
            )
            plt.xlabel("Timestamp")
            plt.ylabel("bps")
            plt.title("Micro-signal")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path.parent / "micro_signal.png")
            plt.close()

        if "vpin" in df.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(df["timestamp"], df["vpin"], label="VPIN")
            plt.plot(df["timestamp"], df["fast_vol_bps"], label="Fast vol (bps)")
            plt.xlabel("Timestamp")
            plt.ylabel("")
            plt.title("Toxicité et volatilité rapide")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path.parent / "toxicity_vol.png")
            plt.close()

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Alias de type : les annotations génériques de sortedcontainers
    # ne sont pas reconnues par tous les checkers, ``dict[float, float]``
    # reflète fidèlement le contenu.
    SortedDict = dict

try:
    from sortedcontainers import SortedDict as _SortedDictImpl  # type: ignore

    _HAS_SORTED = True
except ImportError:
    _SortedDictImpl = dict  # type: ignore
    _HAS_SORTED = False


@dataclass(slots=True)
class DepthMetrics:
    """Métriques de profondeur pour une taille donnée."""

    size_btc: float
    buy_vwap: float | None
    sell_vwap: float | None
    spread_abs: float | None
    spread_bps: float | None


class OrderBook:
    """Carnet d'ordres niveau 2 maintenu localement.

    Choix techniques :
    - bids/asks stockés en SortedDict (si sortedcontainers installé) →
      best_bid / best_ask en O(log n), itération triée sans sort global.
      Fallback sur dict + tri si sortedcontainers absent.
    - Cache invalidé par apply_coinbase_event : best_bid / best_ask /
      best_bid_size / best_ask_size / mid_price / microprice / top_levels
      / imbalances sont calculés une seule fois par mise à jour et
      réutilisés tant que le carnet ne bouge pas. Pour N appels dans la
      même mise à jour, on passe de O(N·k) à O(1).
    - OFI incrémental calculé au moment de la mise à jour elle-même, pas
      dans un getter séparé — la mémoire des meilleurs niveaux précédents
      est maintenue à chaque `apply_coinbase_event`.
    """

    def __init__(self) -> None:
        self.bids: dict[float, float] = _SortedDictImpl()
        self.asks: dict[float, float] = _SortedDictImpl()
        self.sequence_num: int | None = None
        self.last_timestamp: str | None = None

        # Mémoire OFI
        self._prev_best_bid: float | None = None
        self._prev_best_bid_size: float | None = None
        self._prev_best_ask: float | None = None
        self._prev_best_ask_size: float | None = None
        self.last_ofi_event: float = 0.0

        # Caches — invalidés à chaque apply_coinbase_event
        self._cache_best_bid: float | None = None
        self._cache_best_ask: float | None = None
        self._cache_best_bid_size: float | None = None
        self._cache_best_ask_size: float | None = None
        self._cache_mid: float | None = None
        self._cache_microprice: float | None = None
        self._cache_top_bids: list[tuple[float, float]] | None = None
        self._cache_top_asks: list[tuple[float, float]] | None = None
        self._cache_top_n: int = 0
        self._cache_imbalance_l1: float | None = None
        self._cache_imbalance_l3: float | None = None
        self._cache_imbalance_1btc: float | None = None
        self._cache_valid: bool = False

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------
    def _invalidate(self) -> None:
        self._cache_valid = False
        self._cache_best_bid = None
        self._cache_best_ask = None
        self._cache_best_bid_size = None
        self._cache_best_ask_size = None
        self._cache_mid = None
        self._cache_microprice = None
        self._cache_top_bids = None
        self._cache_top_asks = None
        self._cache_top_n = 0
        self._cache_imbalance_l1 = None
        self._cache_imbalance_l3 = None
        self._cache_imbalance_1btc = None

    # ------------------------------------------------------------------
    # Ingestion d'événements
    # ------------------------------------------------------------------
    def apply_coinbase_event(self, event: dict) -> None:
        """Applique un snapshot ou un update Coinbase Advanced Trade."""
        event_type = event.get("type")
        updates = event.get("updates", [])
        if event_type == "snapshot":
            self.bids.clear()
            self.asks.clear()
        # Loop en local pour accès rapide aux attributs (micro-optim).
        bids = self.bids
        asks = self.asks
        for update in updates:
            side = update["side"]
            # Comparaison directe plus rapide que .lower() pour chaînes courtes.
            if side == "bid" or side == "BID":
                book = bids
            else:
                book = asks
            price = float(update["price_level"])
            qty = float(update["new_quantity"])
            if qty <= 0:
                if price in book:
                    del book[price]
            else:
                book[price] = qty
        self._invalidate()

    def _set_level(self, side: str, price: float, quantity: float) -> None:
        """API historique, conservée pour compatibilité avec les tests."""
        book = self.bids if side == "bid" else self.asks
        if quantity <= 0:
            if price in book:
                del book[price]
        else:
            book[price] = quantity
        self._invalidate()

    # ------------------------------------------------------------------
    # Accesseurs cachés
    # ------------------------------------------------------------------
    def best_bid(self) -> float | None:
        if self._cache_best_bid is not None:
            return self._cache_best_bid
        if not self.bids:
            return None
        if _HAS_SORTED:
            # peekitem(-1) = dernier (plus grand) en O(log n).
            self._cache_best_bid = self.bids.peekitem(-1)[0]
        else:
            self._cache_best_bid = max(self.bids)
        return self._cache_best_bid

    def best_ask(self) -> float | None:
        if self._cache_best_ask is not None:
            return self._cache_best_ask
        if not self.asks:
            return None
        if _HAS_SORTED:
            self._cache_best_ask = self.asks.peekitem(0)[0]
        else:
            self._cache_best_ask = min(self.asks)
        return self._cache_best_ask

    def best_bid_size(self) -> float | None:
        if self._cache_best_bid_size is not None:
            return self._cache_best_bid_size
        best = self.best_bid()
        if best is None:
            return None
        self._cache_best_bid_size = self.bids[best]
        return self._cache_best_bid_size

    def best_ask_size(self) -> float | None:
        if self._cache_best_ask_size is not None:
            return self._cache_best_ask_size
        best = self.best_ask()
        if best is None:
            return None
        self._cache_best_ask_size = self.asks[best]
        return self._cache_best_ask_size

    def mid_price(self) -> float | None:
        if self._cache_mid is not None:
            return self._cache_mid
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        self._cache_mid = 0.5 * (bb + ba)
        return self._cache_mid

    def microprice(self) -> float | None:
        """Microprice = prix pondéré par la taille opposée au touch."""
        if self._cache_microprice is not None:
            return self._cache_microprice
        bb, ba = self.best_bid(), self.best_ask()
        bs, asz = self.best_bid_size(), self.best_ask_size()
        if bb is None or ba is None or bs is None or asz is None:
            return None
        denom = bs + asz
        if denom <= 0:
            self._cache_microprice = self.mid_price()
        else:
            self._cache_microprice = (ba * bs + bb * asz) / denom
        return self._cache_microprice

    def top_levels(self, n: int = 5) -> dict[str, list[tuple[float, float]]]:
        """Top N niveaux bid (décroissant) et ask (croissant)."""
        if (
            self._cache_valid
            and self._cache_top_n >= n
            and self._cache_top_bids is not None
            and self._cache_top_asks is not None
        ):
            return {
                "bids": self._cache_top_bids[:n],
                "asks": self._cache_top_asks[:n],
            }

        if _HAS_SORTED:
            # Itération triée sans sort global
            bid_items: list[tuple[float, float]] = []
            for price in reversed(self.bids):
                bid_items.append((price, self.bids[price]))
                if len(bid_items) >= n:
                    break
            ask_items: list[tuple[float, float]] = []
            for price in self.asks:
                ask_items.append((price, self.asks[price]))
                if len(ask_items) >= n:
                    break
        else:
            bid_items = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:n]
            ask_items = sorted(self.asks.items(), key=lambda x: x[0])[:n]

        self._cache_top_bids = bid_items
        self._cache_top_asks = ask_items
        self._cache_top_n = n
        self._cache_valid = True
        return {"bids": bid_items, "asks": ask_items}

    def touch_depth(self, side: str) -> float:
        """Taille présente au meilleur niveau d'un côté donné."""
        if side == "bid":
            s = self.best_bid_size()
            return 0.0 if s is None else float(s)
        if side == "ask":
            s = self.best_ask_size()
            return 0.0 if s is None else float(s)
        raise ValueError("side doit valoir 'bid' ou 'ask'")

    # ------------------------------------------------------------------
    # Imbalances
    # ------------------------------------------------------------------
    @staticmethod
    def _ratio(bid_vol: float, ask_vol: float) -> float:
        denom = bid_vol + ask_vol
        if denom <= 0:
            return 0.0
        return (bid_vol - ask_vol) / denom

    def imbalance_by_levels(self, levels: int = 1) -> float:
        """Imbalance bornée [-1, +1] sur les `levels` premiers niveaux."""
        if levels <= 0:
            return 0.0
        if levels == 1 and self._cache_imbalance_l1 is not None:
            return self._cache_imbalance_l1
        if levels == 3 and self._cache_imbalance_l3 is not None:
            return self._cache_imbalance_l3

        top = self.top_levels(levels)
        bid_vol = sum(q for _, q in top["bids"])
        ask_vol = sum(q for _, q in top["asks"])
        result = self._ratio(bid_vol, ask_vol)
        if levels == 1:
            self._cache_imbalance_l1 = result
        elif levels == 3:
            self._cache_imbalance_l3 = result
        return result

    def imbalance_by_volume(self, target_btc: float = 1.0) -> float:
        """Imbalance sur le volume cumulé jusqu'à `target_btc`."""
        if target_btc <= 0:
            return 0.0
        if abs(target_btc - 1.0) < 1e-9 and self._cache_imbalance_1btc is not None:
            return self._cache_imbalance_1btc

        if _HAS_SORTED:
            bid_iter = ((p, self.bids[p]) for p in reversed(self.bids))
            ask_iter = ((p, self.asks[p]) for p in self.asks)
        else:
            bid_iter = iter(sorted(self.bids.items(), key=lambda x: x[0], reverse=True))
            ask_iter = iter(sorted(self.asks.items(), key=lambda x: x[0]))

        def cumulated(it) -> float:
            remaining = target_btc
            total = 0.0
            for _, qty in it:
                if remaining <= 1e-12:
                    break
                take = qty if qty < remaining else remaining
                total += take
                remaining -= take
            return total

        result = self._ratio(cumulated(bid_iter), cumulated(ask_iter))
        if abs(target_btc - 1.0) < 1e-9:
            self._cache_imbalance_1btc = result
        return result

    # ------------------------------------------------------------------
    # Order Flow Imbalance (Cont, Kukanov & Stoikov 2014)
    # ------------------------------------------------------------------
    def compute_ofi_delta(self) -> float:
        """Calcule l'OFI entre l'état précédent et l'état courant.

        Convention standard :
        - bid price up     → +new_bid_size
        - bid price down   → -old_bid_size
        - bid inchangé     → (new_bid_size - old_bid_size)
        - ask price up     → +old_ask_size
        - ask price down   → -new_ask_size
        - ask inchangé     → (old_ask_size - new_ask_size)
        """
        new_bb = self.best_bid()
        new_ba = self.best_ask()
        new_bs = self.best_bid_size() or 0.0
        new_as = self.best_ask_size() or 0.0

        ofi = 0.0
        if self._prev_best_bid is not None and new_bb is not None:
            prev_bs = self._prev_best_bid_size or 0.0
            if new_bb > self._prev_best_bid:
                ofi += new_bs
            elif new_bb < self._prev_best_bid:
                ofi -= prev_bs
            else:
                ofi += new_bs - prev_bs
        if self._prev_best_ask is not None and new_ba is not None:
            prev_as = self._prev_best_ask_size or 0.0
            if new_ba > self._prev_best_ask:
                ofi += prev_as
            elif new_ba < self._prev_best_ask:
                ofi -= new_as
            else:
                ofi += prev_as - new_as

        self._prev_best_bid = new_bb
        self._prev_best_bid_size = new_bs if new_bb is not None else None
        self._prev_best_ask = new_ba
        self._prev_best_ask_size = new_as if new_ba is not None else None
        self.last_ofi_event = ofi
        return ofi

    # ------------------------------------------------------------------
    # VWAP de consommation (spread effectif)
    # ------------------------------------------------------------------
    def _vwap_from_depth(self, side: str, size_btc: float) -> float | None:
        """VWAP d'exécution si on devait consommer `size_btc` d'un côté."""
        if size_btc <= 0:
            raise ValueError("size_btc doit être strictement positif")

        if side == "buy":
            if _HAS_SORTED:
                levels_iter = ((p, self.asks[p]) for p in self.asks)
            else:
                levels_iter = iter(sorted(self.asks.items(), key=lambda x: x[0]))
        elif side == "sell":
            if _HAS_SORTED:
                levels_iter = ((p, self.bids[p]) for p in reversed(self.bids))
            else:
                levels_iter = iter(
                    sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
                )
        else:
            raise ValueError("side doit valoir 'buy' ou 'sell'")

        remaining = size_btc
        notional = 0.0
        for price, quantity in levels_iter:
            take = quantity if quantity < remaining else remaining
            notional += take * price
            remaining -= take
            if remaining <= 1e-12:
                return notional / size_btc
        return None

    def depth_metrics(self, size_btc: float) -> DepthMetrics:
        buy_vwap = self._vwap_from_depth("buy", size_btc)
        sell_vwap = self._vwap_from_depth("sell", size_btc)
        mid = self.mid_price()
        if buy_vwap is None or sell_vwap is None or mid is None:
            return DepthMetrics(size_btc, buy_vwap, sell_vwap, None, None)
        spread_abs = buy_vwap - sell_vwap
        spread_bps = (spread_abs / mid) * 10_000 if mid else None
        return DepthMetrics(size_btc, buy_vwap, sell_vwap, spread_abs, spread_bps)

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def snapshot_rows(self, top_n: int = 5) -> list[dict]:
        """Transforme les meilleurs niveaux en lignes plates (pour CSV)."""
        rows: list[dict] = []
        levels = self.top_levels(top_n)
        for rank, (price, qty) in enumerate(levels["bids"], start=1):
            rows.append({"side": "bid", "rank": rank, "price": price, "quantity": qty})
        for rank, (price, qty) in enumerate(levels["asks"], start=1):
            rows.append({"side": "ask", "rank": rank, "price": price, "quantity": qty})
        return rows

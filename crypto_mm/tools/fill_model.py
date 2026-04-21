"""Modèle de position dans la file d'attente pour la simulation de fills.

Suit, pour chaque côté (bid et ask), la quantité de liquidité présente
devant notre ordre au même niveau de prix. Permet d'estimer les fills de
manière plus réaliste que via des heuristiques continues : un trade ne
nous touche qu'après avoir consommé la file d'attente qui nous précède.

Règles :

1. À l'arrivée d'un trade sur notre côté, la quantité tradée consomme
   d'abord la queue devant nous (FIFO). Le reliquat éventuel matche
   ensuite notre ordre à hauteur de ``min(notre_taille, reliquat)``.

2. Si notre quote change de prix, nous perdons notre place ; la queue
   devant est réinitialisée à la taille affichée au nouveau niveau.
   Si le prix ne change pas, la queue est conservée.

Le modèle suit l'agrégat devant nous, pas chaque ordre individuellement.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QueueState:
    """État de queue pour un côté."""

    price: float | None = None
    queue_ahead_btc: float = 0.0
    my_size_btc: float = 0.0

    def reset_to_new_quote(self, price: float, my_size: float, touch_depth: float) -> None:
        """Nouvelle quote à un prix différent → on arrive en bout de queue.

        ``touch_depth`` est la taille affichée au prix correspondant dans le
        carnet public (la liquidité déjà présente devant nous).
        """
        self.price = price
        self.queue_ahead_btc = max(0.0, float(touch_depth))
        self.my_size_btc = float(my_size)

    def keep_place(self, my_size: float) -> None:
        """On garde notre prix, on met juste à jour notre taille."""
        self.my_size_btc = float(my_size)

    def consume_trade(self, trade_size: float) -> float:
        """Applique un trade qui nous touche et retourne la quantité fillée.

        Le trade consomme d'abord la queue devant nous puis le reliquat va
        contre notre ordre (clippé par notre taille restante).
        """
        if self.my_size_btc <= 0 or trade_size <= 0:
            return 0.0
        remaining = float(trade_size)
        # Consomme la queue devant nous.
        if self.queue_ahead_btc > 0:
            absorbed = min(self.queue_ahead_btc, remaining)
            self.queue_ahead_btc -= absorbed
            remaining -= absorbed
        if remaining <= 0:
            return 0.0
        # Le reliquat nous touche.
        filled = min(self.my_size_btc, remaining)
        self.my_size_btc -= filled
        return filled


class QueuePositionTracker:
    """Suit la queue position pour bid et ask.

    Usage typique dans le replay :

        tracker = QueuePositionTracker()
        # À chaque update_quotes de la stratégie :
        tracker.sync_quote("bid", bid_price, bid_size, book.touch_depth("bid"))
        tracker.sync_quote("ask", ask_price, ask_size, book.touch_depth("ask"))
        # À chaque trade qui matcherait notre côté :
        filled = tracker.on_trade(side, price, size)
    """

    def __init__(self) -> None:
        self.bid = QueueState()
        self.ask = QueueState()

    def sync_quote(
        self,
        side: str,
        price: float | None,
        size: float,
        touch_depth: float,
    ) -> None:
        """Synchronise notre position de queue après un ``update_quotes``.

        - Si le prix a changé → on perd notre place (reset à la queue observée)
        - Si le prix est le même → on garde notre place (ajuste juste notre size)
        - Si ``price`` est None → pas de quote active sur ce côté.
        """
        state = self.bid if side == "bid" else self.ask
        if price is None:
            state.price = None
            state.queue_ahead_btc = 0.0
            state.my_size_btc = 0.0
            return
        if state.price is None or abs(state.price - price) > 1e-9:
            state.reset_to_new_quote(price, size, touch_depth)
        else:
            state.keep_place(size)

    def on_trade(self, aggressor_side: str, trade_price: float, trade_size: float) -> float:
        """Applique un trade et retourne la quantité fillée sur notre quote.

        ``aggressor_side`` : "SELL" si un vendeur agressif consomme des bids,
        "BUY" si un acheteur agressif consomme des asks.
        """
        # Un vendeur agressif touche notre BID si notre prix est ≥ prix du trade.
        if aggressor_side == "SELL":
            if self.bid.price is None or self.bid.price < trade_price - 1e-9:
                return 0.0
            return self.bid.consume_trade(trade_size)
        # Un acheteur agressif touche notre ASK si notre prix est ≤ prix du trade.
        if aggressor_side == "BUY":
            if self.ask.price is None or self.ask.price > trade_price + 1e-9:
                return 0.0
            return self.ask.consume_trade(trade_size)
        return 0.0

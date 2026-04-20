from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

# Types de côté normalisés pour toute l'application.
SideLiteral = Literal["BUY", "SELL"]
BookSideLiteral = Literal["bid", "ask"]


@dataclass(slots=True)
class Trade:
    """Trade public reçu sur le flux market_trades."""

    trade_id: str
    product_id: str
    price: float
    size: float
    side: SideLiteral
    time: datetime


@dataclass(slots=True)
class Quote:
    """Quote que nous postons virtuellement sur le marché."""

    side: BookSideLiteral
    price: float
    size: float
    created_at: datetime


@dataclass(slots=True)
class Fill:
    """Exécution simulée générée à partir des trades publics."""

    time: datetime
    side: SideLiteral
    price: float
    size: float
    reason: str


@dataclass(slots=True)
class OrderBookLevel:
    """Niveau agrégé utilisé pour les snapshots / exports."""

    price: float
    size: float

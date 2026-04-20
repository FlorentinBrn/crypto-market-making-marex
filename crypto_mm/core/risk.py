from __future__ import annotations


class RiskManager:
    """Gère les contraintes de risque de la simulation.

    Contraintes imposées :
    - exposition notionnelle max : $1,000,000
    - perte max : $100,000 (10 % du capital initial)

    Ce manager expose aussi un niveau de risque, des alertes, un health
    score et des recommandations. Il centralise toute la logique de
    "peut-on encore trader ?", "doit-on passer en reduce-only ?", etc.
    """

    def __init__(
        self,
        max_notional_usd: float,
        max_loss_usd: float,
        initial_equity: float,
        reduce_only_loss_utilization: float = 0.75,
        reduce_only_exposure_utilization: float = 0.85,
    ) -> None:
        self.max_notional_usd = max_notional_usd
        self.max_loss_usd = max_loss_usd
        self.initial_equity = initial_equity
        self.reduce_only_loss_utilization = reduce_only_loss_utilization
        self.reduce_only_exposure_utilization = reduce_only_exposure_utilization

    # ------------------------------------------------------------------
    # Bornes
    # ------------------------------------------------------------------
    @property
    def stop_equity(self) -> float:
        return self.initial_equity - self.max_loss_usd

    def remaining_loss_budget(self, equity: float) -> float:
        return equity - self.stop_equity

    def loss_utilization(self, equity: float) -> float:
        if self.max_loss_usd <= 0:
            return 1.0
        drawdown = max(0.0, self.initial_equity - equity)
        return min(1.0, drawdown / self.max_loss_usd)

    def exposure_utilization(self, position_btc: float, price: float | None) -> float:
        if price is None or self.max_notional_usd <= 0:
            return 0.0
        return min(1.0, abs(position_btc * price) / self.max_notional_usd)

    def max_position_btc(self, price: float | None) -> float:
        if price is None or price <= 0:
            return 0.0
        return self.max_notional_usd / price

    # ------------------------------------------------------------------
    # Décisions
    # ------------------------------------------------------------------
    def can_continue(self, equity: float) -> bool:
        return equity >= self.stop_equity

    def should_reduce_only(
        self, equity: float, position_btc: float, price: float | None
    ) -> bool:
        return (
            self.loss_utilization(equity) >= self.reduce_only_loss_utilization
            or self.exposure_utilization(position_btc, price)
            >= self.reduce_only_exposure_utilization
        )

    def quote_size_multiplier(
        self, equity: float, position_btc: float, price: float | None
    ) -> float:
        """Taille dynamique : on réduit la taille quand le risque monte.

        - stress < 0.85 : taille ≈ 1 - 0.85 * stress (linéaire)
        - 0.85 ≤ stress < 0.95 : taille = 15 % pour rester présent
        - stress ≥ 0.95 : taille = 0 (plus de quotes)
        """
        loss_util = self.loss_utilization(equity)
        exposure_util = self.exposure_utilization(position_btc, price)
        stress = max(loss_util, exposure_util)
        if stress >= 0.95:
            return 0.0
        if stress >= 0.85:
            return 0.15
        return max(0.15, 1.0 - 0.85 * stress)

    def can_add_position(
        self, current_position_btc: float, proposed_trade_size_btc: float, price: float
    ) -> bool:
        projected_notional = abs(
            (current_position_btc + proposed_trade_size_btc) * price
        )
        return projected_notional <= self.max_notional_usd + 1e-9

    def can_execute_fill(
        self,
        current_position_btc: float,
        fill_size_btc_signed: float,
        price: float,
        equity: float,
    ) -> bool:
        if not self.can_continue(equity):
            return False
        return self.can_add_position(current_position_btc, fill_size_btc_signed, price)

    # ------------------------------------------------------------------
    # Surveillance
    # ------------------------------------------------------------------
    def _risk_level(
        self, equity: float, position_btc: float, price: float | None
    ) -> str:
        loss_util = self.loss_utilization(equity)
        exposure_util = self.exposure_utilization(position_btc, price)
        stress = max(loss_util, exposure_util)
        if not self.can_continue(equity):
            return "STOP"
        if stress >= 0.85:
            return "HIGH"
        if stress >= 0.60:
            return "MEDIUM"
        return "LOW"

    def _alerts(
        self, equity: float, position_btc: float, price: float | None
    ) -> list[str]:
        alerts: list[str] = []
        if not self.can_continue(equity):
            alerts.append("Kill switch activé : budget de perte atteint.")
        if self.loss_utilization(equity) >= self.reduce_only_loss_utilization:
            alerts.append("Budget de perte tendu : passage en reduce-only.")
        if (
            self.exposure_utilization(position_btc, price)
            >= self.reduce_only_exposure_utilization
        ):
            alerts.append("Exposition proche de la limite : réduction des quotes.")
        return alerts

    def _recommendations(
        self, equity: float, position_btc: float, price: float | None
    ) -> list[str]:
        recos: list[str] = []
        loss_util = self.loss_utilization(equity)
        exposure_util = self.exposure_utilization(position_btc, price)
        if exposure_util > 0.70:
            recos.append("Réduire l'inventaire ou accentuer le skew vers la sortie.")
        if loss_util > 0.60:
            recos.append("Diminuer la taille de quote et élargir le spread.")
        if price is not None and abs(position_btc) > 0.8 * self.max_position_btc(price):
            recos.append("Position proche du maximum notionnel autorisé.")
        return recos

    def health_score(
        self, equity: float, position_btc: float, price: float | None
    ) -> int:
        """Score de santé global (0-100).

        Démarre à 100, retranche pour inventaire/drawdown, ajoute un bonus
        pour les gains réalisés.
        """
        score = 100
        loss_util = self.loss_utilization(equity)
        exposure_util = self.exposure_utilization(position_btc, price)
        if exposure_util > 0.80:
            score -= 30
        elif exposure_util > 0.60:
            score -= 15
        if loss_util > 0.50:
            score -= 25
        elif loss_util > 0.20:
            score -= 10
        pnl = equity - self.initial_equity
        if pnl > 0:
            score += min(10, int(pnl / 1000.0))
        return max(0, min(100, score))

    def status(
        self, equity: float, position_btc: float, price: float | None
    ) -> dict[str, float | bool | None | list[str] | str]:
        return {
            "can_continue": self.can_continue(equity),
            "reduce_only": self.should_reduce_only(equity, position_btc, price),
            "remaining_loss_budget": self.remaining_loss_budget(equity),
            "loss_utilization": self.loss_utilization(equity),
            "exposure_utilization": self.exposure_utilization(position_btc, price),
            "max_position_btc": self.max_position_btc(price),
            "stop_equity": self.stop_equity,
            "risk_level": self._risk_level(equity, position_btc, price),
            "health_score": self.health_score(equity, position_btc, price),
            "alerts": self._alerts(equity, position_btc, price),
            "recommendations": self._recommendations(equity, position_btc, price),
        }

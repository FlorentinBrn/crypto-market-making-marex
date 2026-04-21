from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass(slots=True)
class BanditDecision:
    state_key: str
    arm_idx: int
    spread_multiplier: float
    size_multiplier: float


class ContextualBanditQuoter:
    """Moteur de reinforcement learning optionnel.

    Désactivé par défaut (cf. ``Settings.use_contextual_bandit``). Il
    s'agit d'un bandit contextuel epsilon-greedy avec bonus UCB qui
    choisit à chaque décision un couple (multiplicateur de spread,
    multiplicateur de taille) en fonction de l'état courant de
    microstructure et de risque.

    La récompense est la variation d'equity entre deux décisions,
    pénalisée par la taille d'inventaire afin d'éviter l'accumulation.
    """

    def __init__(
        self,
        spread_multipliers: list[float] | None = None,
        size_multipliers: list[float] | None = None,
        epsilon: float = 0.08,
        inventory_penalty: float = 2.0,
    ) -> None:
        self.spread_multipliers = spread_multipliers or [0.85, 1.0, 1.2, 1.5]
        self.size_multipliers = size_multipliers or [0.75, 1.0, 1.25]
        self.arms = [
            (spread_mult, size_mult)
            for spread_mult in self.spread_multipliers
            for size_mult in self.size_multipliers
        ]
        self.epsilon = epsilon
        self.inventory_penalty = inventory_penalty
        self.counts: dict[str, list[int]] = defaultdict(lambda: [0 for _ in self.arms])
        self.values: dict[str, list[float]] = defaultdict(
            lambda: [0.0 for _ in self.arms]
        )
        self.last_decision: BanditDecision | None = None
        self.last_equity: float | None = None
        self.last_mid: float | None = None

    # ------------------------------------------------------------------
    # État discret
    # ------------------------------------------------------------------
    def _bucket(
        self,
        vol_bps: float,
        imbalance_l3: float,
        position_btc: float,
        loss_utilization: float,
    ) -> str:
        vol_bucket = (
            "highvol" if vol_bps >= 2.5 else "midvol" if vol_bps >= 1.0 else "lowvol"
        )
        imb_bucket = (
            "buy_press"
            if imbalance_l3 >= 0.15
            else "sell_press" if imbalance_l3 <= -0.15 else "neutral"
        )
        inv_bucket = (
            "long"
            if position_btc > 0.05
            else "short" if position_btc < -0.05 else "flat"
        )
        risk_bucket = "stress" if loss_utilization >= 0.6 else "ok"
        return "|".join([vol_bucket, imb_bucket, inv_bucket, risk_bucket])

    # ------------------------------------------------------------------
    # Choix d'action : epsilon-greedy + UCB
    # ------------------------------------------------------------------
    def choose(
        self,
        vol_bps: float,
        imbalance_l3: float,
        position_btc: float,
        loss_utilization: float,
    ) -> BanditDecision:
        state_key = self._bucket(vol_bps, imbalance_l3, position_btc, loss_utilization)
        counts = self.counts[state_key]
        values = self.values[state_key]

        if random.random() < self.epsilon or sum(counts) == 0:
            arm_idx = random.randrange(len(self.arms))
        else:
            total = sum(counts)
            scores = []
            for idx, (count, value) in enumerate(zip(counts, values)):
                if count == 0:
                    scores.append(float("inf"))
                else:
                    bonus = math.sqrt(2.0 * math.log(total + 1.0) / count)
                    scores.append(value + bonus)
            arm_idx = int(max(range(len(scores)), key=lambda i: scores[i]))

        spread_multiplier, size_multiplier = self.arms[arm_idx]
        decision = BanditDecision(
            state_key=state_key,
            arm_idx=arm_idx,
            spread_multiplier=spread_multiplier,
            size_multiplier=size_multiplier,
        )
        self.last_decision = decision
        return decision

    # ------------------------------------------------------------------
    # Update : calcul de la récompense + moyenne courante
    # ------------------------------------------------------------------
    def update(
        self,
        equity: float,
        mid_price: float | None,
        position_btc: float,
    ) -> float | None:
        if self.last_decision is None:
            self.last_equity = equity
            self.last_mid = mid_price
            return None
        if self.last_equity is None:
            self.last_equity = equity
            self.last_mid = mid_price
            return None

        delta_equity = equity - self.last_equity
        inventory_penalty = self.inventory_penalty * abs(position_btc)
        reward = delta_equity - inventory_penalty
        state_key = self.last_decision.state_key
        arm_idx = self.last_decision.arm_idx

        self.counts[state_key][arm_idx] += 1
        n = self.counts[state_key][arm_idx]
        old_value = self.values[state_key][arm_idx]
        self.values[state_key][arm_idx] = old_value + (reward - old_value) / n
        self.last_equity = equity
        self.last_mid = mid_price
        return reward

    def diagnostics(self) -> dict[str, float | int | str | None]:
        if self.last_decision is None:
            return {
                "bandit_state": None,
                "bandit_arm": None,
                "bandit_spread_multiplier": None,
                "bandit_size_multiplier": None,
                "bandit_arm_count": None,
                "bandit_arm_value": None,
            }
        state = self.last_decision.state_key
        arm_idx = self.last_decision.arm_idx
        return {
            "bandit_state": state,
            "bandit_arm": arm_idx,
            "bandit_spread_multiplier": self.last_decision.spread_multiplier,
            "bandit_size_multiplier": self.last_decision.size_multiplier,
            "bandit_arm_count": self.counts[state][arm_idx],
            "bandit_arm_value": self.values[state][arm_idx],
        }

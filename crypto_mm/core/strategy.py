from __future__ import annotations

from collections import deque
from math import exp

from .learning import ContextualBanditQuoter
from .models import Fill, Quote, Trade
from .risk import RiskManager
from .utils import utc_now


class MarketMaker:
    """Stratégie de market making BTC/USD.

    Principes :

    1. On calcule un *fair price* à partir du mid et d'un signal combiné
       issu de la microstructure (microprice edge, imbalance L3, OFI EWMA,
       trade-flow, VPIN). Ce signal déplace le fair price au lieu de
       simplement pondérer un des termes.

    2. On applique un *inventory skew* qui pousse vers la sortie quand on
       accumule : avec inventaire long, on baisse notre ask (on veut vendre
       plus vite) et on recule notre bid (on veut moins acheter).

    3. Le *half-spread* s'élargit quand :
       - la volatilité rapide monte (EWMA sur les log-returns du mid) ;
       - la toxicité (VPIN) dépasse un seuil de référence ;
       - l'imbalance L1 absolu est fort (anticipation d'un mouvement).

    4. Quand le spread intérieur est étroit, on rejoint le *touch* pour
       maximiser la probabilité d'être fill. Si OFI EWMA est clairement
       favorable à notre côté (flux arrivant sur notre niveau), on joint
       encore plus agressivement.

    5. Un *reduce-only* strict s'active quand le risque dépasse les seuils,
       pour forcer la sortie d'inventaire.

    6. La cadence de requote est limitée à ~100 ms pour ne pas saturer.

    7. La simulation de fill utilise la compétitivité du prix et un modèle
       de queue-share à partir de la profondeur au touch.
    """

    def __init__(
        self,
        quote_size_btc: float,
        base_half_spread_bps: float,
        inventory_skew_bps_per_btc: float,
        risk_manager: RiskManager,
        volatility_window: int = 120,
        volatility_spread_multiplier: float = 1.0,
        initial_cash_usd: float = 1_000_000.0,
        min_quote_size_btc: float = 0.02,
        min_half_spread_bps: float = 0.30,
        max_half_spread_bps: float = 8.0,
        touch_join_threshold_bps: float = 2.0,
        queue_ahead_factor: float = 1.25,
        fill_intensity: float = 1.30,
        cooldown_ms_after_fill: int = 150,
        quote_refresh_interval_ms: int = 100,
        microprice_weight: float = 0.85,
        imbalance_shift_bps: float = 3.0,
        imbalance_widening_bps: float = 1.5,
        use_contextual_bandit: bool = False,
        ewma_spread_alpha: float = 0.08,
        rolling_spread_window: int = 250,
        combined_signal_weight: float = 1.0,
        vpin_reference: float = 0.25,
        vpin_widening_bps_per_unit: float = 4.0,
        fast_vol_widening_coef: float = 0.5,
        ofi_aggressive_join_threshold: float = 15.0,
    ) -> None:
        # Dimensions
        self.quote_size_btc = quote_size_btc
        self.min_quote_size_btc = min_quote_size_btc
        self.base_half_spread_bps = base_half_spread_bps
        self.min_half_spread_bps = min_half_spread_bps
        self.max_half_spread_bps = max_half_spread_bps
        self.inventory_skew_bps_per_btc = inventory_skew_bps_per_btc
        self.risk_manager = risk_manager

        # Volatilité / spread dynamique
        self.volatility_window = volatility_window
        self.volatility_spread_multiplier = volatility_spread_multiplier
        self.ewma_spread_alpha = ewma_spread_alpha

        # Fill sim
        self.touch_join_threshold_bps = touch_join_threshold_bps
        self.queue_ahead_factor = queue_ahead_factor
        self.fill_intensity = fill_intensity
        self.cooldown_ms_after_fill = cooldown_ms_after_fill
        self.quote_refresh_interval_ms = quote_refresh_interval_ms

        # Microstructure / signaux
        self.microprice_weight = microprice_weight
        self.imbalance_shift_bps = imbalance_shift_bps
        self.imbalance_widening_bps = imbalance_widening_bps
        self.combined_signal_weight = combined_signal_weight
        self.vpin_reference = vpin_reference
        self.vpin_widening_bps_per_unit = vpin_widening_bps_per_unit
        self.fast_vol_widening_coef = fast_vol_widening_coef
        self.ofi_aggressive_join_threshold = ofi_aggressive_join_threshold

        # Option RL
        self.use_contextual_bandit = use_contextual_bandit
        self.bandit = ContextualBanditQuoter() if use_contextual_bandit else None

        # État portefeuille
        self.cash_usd = initial_cash_usd
        self.position_btc = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0

        # Historiques
        self.mid_history: deque[float] = deque(maxlen=volatility_window)
        self.inside_spread_history: deque[float] = deque(maxlen=rolling_spread_window)
        self.ewma_inside_spread_bps: float | None = None
        self.current_bid: Quote | None = None
        self.current_ask: Quote | None = None
        self.executions: list[dict] = []
        self.last_equity = initial_cash_usd
        self.last_fill_ts_ms = 0.0
        self.last_quote_update_ts_ms = 0.0

        # Accumulateurs pour vol et rolling spread.
        # - _logret_* : sum / sum² des log-returns sur la fenêtre vol.
        # - _last_mid_for_logret : dernier mid utilisé pour produire un
        #   log-return (mis à jour à chaque update_quotes).
        # - _logret_buffer : deque parallèle à mid_history pour pouvoir
        #   retirer proprement les plus anciens log-returns quand la
        #   fenêtre se remplit.
        # - _inside_spread_sum : somme courante du rolling_spread_window,
        #   maintenue incrémentalement.
        self._logret_buffer: deque[float] = deque(maxlen=volatility_window)
        self._logret_sum: float = 0.0
        self._logret_sum2: float = 0.0
        self._logret_count: int = 0
        self._last_mid_for_logret: float | None = None
        self._inside_spread_sum: float = 0.0

        self.last_quote_context: dict[str, float | bool | str | list[str] | None] = {
            "half_spread_bps": None,
            "inventory_skew_bps": None,
            "signal_shift_bps": None,
            "volatility_bps": None,
            "inside_spread_bps": None,
            "quote_size_btc": quote_size_btc,
            "reduce_only": False,
            "loss_utilization": 0.0,
            "exposure_utilization": 0.0,
            "imbalance_l1": 0.0,
            "imbalance_l3": 0.0,
            "microprice_edge_bps": 0.0,
            "ofi_ewma": 0.0,
            "trade_flow_signed": 0.0,
            "vpin": 0.0,
            "fast_vol_bps": 0.0,
            "combined_signal_bps": 0.0,
            "fair_price": None,
            "bandit_state": None,
            "bandit_arm": None,
            "bandit_reward": None,
            "bandit_spread_multiplier": None,
            "bandit_size_multiplier": None,
            "rolling_inside_spread_bps": None,
            "ewma_inside_spread_bps": None,
        }

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------
    def mark_to_market(self, mid_price: float | None) -> dict[str, float]:
        if mid_price is None:
            return {
                "position_btc": self.position_btc,
                "avg_entry_price": self.avg_entry_price,
                "exposure_usd": 0.0,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": 0.0,
                "equity": self.cash_usd,
            }
        exposure = self.position_btc * mid_price
        unrealized = self.position_btc * (mid_price - self.avg_entry_price)
        equity = self.cash_usd + exposure
        self.last_equity = equity
        return {
            "position_btc": self.position_btc,
            "avg_entry_price": self.avg_entry_price,
            "exposure_usd": exposure,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized,
            "equity": equity,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _volatility_bps(self) -> float:
        """Volatilité sur la fenêtre mid, exprimée en bps.

        Calcul Welford-like en ligne : à chaque nouveau mid, on ajoute le
        log-return (log(m/m_prev)) à un buffer deque tournant, et on
        maintient sum / sum² incrémentalement. Cela évite le coût O(N) de
        np.diff + np.std à chaque update_quotes.
        """
        if self._logret_count < 10:
            return 0.0
        n = float(self._logret_count)
        mean = self._logret_sum / n
        var = max(0.0, self._logret_sum2 / n - mean * mean)
        sigma = var**0.5
        return sigma * 10_000 * self.volatility_spread_multiplier

    @staticmethod
    def _inside_spread_bps(
        best_bid: float | None, best_ask: float | None, mid_price: float | None
    ) -> float:
        if best_bid is None or best_ask is None or mid_price is None or mid_price <= 0:
            return 0.0
        return ((best_ask - best_bid) / mid_price) * 10_000

    @staticmethod
    def _microprice_edge_bps(
        mid_price: float | None, microprice: float | None
    ) -> float:
        if mid_price is None or microprice is None or mid_price <= 0:
            return 0.0
        return (microprice - mid_price) / mid_price * 10_000

    def _effective_quote_size(self, mid_price: float | None) -> float:
        multiplier = self.risk_manager.quote_size_multiplier(
            self.last_equity, self.position_btc, mid_price
        )
        effective = self.quote_size_btc * multiplier
        if effective <= 0:
            return 0.0
        return max(self.min_quote_size_btc, min(self.quote_size_btc, effective))

    def _should_requote(self, force: bool) -> bool:
        if force:
            return True
        now_ms = utc_now().timestamp() * 1000
        return (now_ms - self.last_quote_update_ts_ms) >= self.quote_refresh_interval_ms

    def _update_inside_spread_memory(
        self, inside_spread_bps: float
    ) -> tuple[float, float]:
        """Mémoire rolling + EWMA du spread intérieur en bps (O(1) amorti).

        - Rolling mean : on maintient la somme courante incrémentalement
          (ajout du nouveau, retrait du plus ancien si la deque déborde).
        - EWMA : mise à jour scalaire.
        """
        hist = self.inside_spread_history
        if len(hist) == hist.maxlen:
            # On va perdre l'élément le plus ancien : on le retire de la somme.
            self._inside_spread_sum -= hist[0]
        hist.append(inside_spread_bps)
        self._inside_spread_sum += inside_spread_bps

        if self.ewma_inside_spread_bps is None:
            self.ewma_inside_spread_bps = inside_spread_bps
        else:
            a = self.ewma_spread_alpha
            self.ewma_inside_spread_bps = (
                a * inside_spread_bps + (1.0 - a) * self.ewma_inside_spread_bps
            )
        rolling_mean = (
            self._inside_spread_sum / len(hist) if hist else inside_spread_bps
        )
        return rolling_mean, float(self.ewma_inside_spread_bps)

    # ------------------------------------------------------------------
    # Boucle principale : update des quotes
    # ------------------------------------------------------------------
    def update_quotes(
        self,
        mid_price: float | None,
        best_bid: float | None = None,
        best_ask: float | None = None,
        best_bid_size: float | None = None,
        best_ask_size: float | None = None,
        microprice: float | None = None,
        imbalance_l1: float = 0.0,
        imbalance_l3: float = 0.0,
        ofi_ewma: float = 0.0,
        trade_flow_signed: float = 0.0,
        vpin: float = 0.0,
        fast_vol_bps: float = 0.0,
        combined_signal_bps: float | None = None,
        force: bool = False,
    ) -> tuple[Quote | None, Quote | None]:
        if mid_price is None:
            self.current_bid = None
            self.current_ask = None
            return self.current_bid, self.current_ask

        self.mid_history.append(mid_price)

        # Update incrémental des log-returns.
        if (
            self._last_mid_for_logret is not None
            and mid_price > 0
            and self._last_mid_for_logret > 0
        ):
            import math

            r = math.log(mid_price / self._last_mid_for_logret)
            if (
                len(self._logret_buffer) == self._logret_buffer.maxlen
                and self._logret_buffer.maxlen
            ):
                old = self._logret_buffer[0]
                self._logret_sum -= old
                self._logret_sum2 -= old * old
                self._logret_count -= 1
            self._logret_buffer.append(r)
            self._logret_sum += r
            self._logret_sum2 += r * r
            self._logret_count += 1
        self._last_mid_for_logret = mid_price

        mtm = self.mark_to_market(mid_price)
        equity = mtm["equity"]

        # 1) Kill switch
        risk_status = self.risk_manager.status(equity, self.position_btc, mid_price)
        if not bool(risk_status["can_continue"]):
            self.current_bid = None
            self.current_ask = None
            self.last_quote_context.update(risk_status)
            return self.current_bid, self.current_ask

        # 2) Cadence de requote
        if not self._should_requote(force):
            return self.current_bid, self.current_ask

        # 3) Mesures courantes
        inside_spread_bps = self._inside_spread_bps(best_bid, best_ask, mid_price)
        rolling_inside_spread_bps, ewma_inside_spread_bps = (
            self._update_inside_spread_memory(inside_spread_bps)
        )
        volatility_bps = self._volatility_bps()
        microprice_edge_bps = self._microprice_edge_bps(mid_price, microprice)

        # 4) Signal combiné : si l'analytics nous donne un signal, on l'utilise ;
        #    sinon on reconstruit un signal simple à partir de microprice/imbalance
        #    (utile pour les tests unitaires qui n'ont pas l'analytics).
        if combined_signal_bps is None:
            combined_signal_bps = (
                self.microprice_weight * microprice_edge_bps
                + self.imbalance_shift_bps * imbalance_l3
            )
        signal_shift_bps = self.combined_signal_weight * float(combined_signal_bps)

        # 5) Widening dynamique
        dynamic_widening_bps = self.imbalance_widening_bps * abs(imbalance_l1)

        # Toxicité : plus VPIN dépasse la référence, plus on élargit.
        vpin_excess = max(0.0, float(vpin) - self.vpin_reference)
        toxicity_widening_bps = self.vpin_widening_bps_per_unit * vpin_excess

        # Vol rapide : ajout direct pondéré.
        fast_vol_widening_bps = self.fast_vol_widening_coef * float(fast_vol_bps)

        # 6) Option RL (bandit contextuel)
        bandit_reward = None
        spread_bandit_multiplier = 1.0
        size_bandit_multiplier = 1.0
        if self.bandit is not None:
            bandit_reward = self.bandit.update(
                equity=equity, mid_price=mid_price, position_btc=self.position_btc
            )
            decision = self.bandit.choose(
                vol_bps=volatility_bps,
                imbalance_l3=imbalance_l3,
                position_btc=self.position_btc,
                loss_utilization=float(risk_status.get("loss_utilization", 0.0)),
            )
            spread_bandit_multiplier = decision.spread_multiplier
            size_bandit_multiplier = decision.size_multiplier

        # 7) Half-spread agrégé
        baseline_spread_bps = max(
            inside_spread_bps,
            0.50 * (ewma_inside_spread_bps or inside_spread_bps),
            0.35 * (rolling_inside_spread_bps or inside_spread_bps),
        )
        raw_half_spread_bps = (
            self.base_half_spread_bps
            + 0.20 * inside_spread_bps
            + 0.15 * baseline_spread_bps
            + volatility_bps
            + dynamic_widening_bps
            + toxicity_widening_bps
            + fast_vol_widening_bps
        )
        raw_half_spread_bps *= spread_bandit_multiplier
        half_spread_bps = min(
            self.max_half_spread_bps, max(self.min_half_spread_bps, raw_half_spread_bps)
        )

        # 8) Inventory skew + signal shift → reservation price
        inventory_shift_bps = -self.inventory_skew_bps_per_btc * self.position_btc
        if bool(risk_status["reduce_only"]):
            inventory_shift_bps *= 1.5

        fair_price = mid_price * (1 + signal_shift_bps / 10_000)
        reservation_price = fair_price * (1 + inventory_shift_bps / 10_000)
        bid_price = reservation_price * (1 - half_spread_bps / 10_000)
        ask_price = reservation_price * (1 + half_spread_bps / 10_000)

        # 9) Touch-join : rejoindre le top-of-book quand le spread est étroit
        if best_bid is not None and inside_spread_bps <= self.touch_join_threshold_bps:
            # On accepte de rejoindre le meilleur bid, mais pas mieux.
            bid_price = max(bid_price, best_bid)
            bid_price = min(bid_price, best_bid)
        if best_ask is not None and inside_spread_bps <= self.touch_join_threshold_bps:
            ask_price = min(ask_price, best_ask)
            ask_price = max(ask_price, best_ask)

        # 10) Join agressif quand OFI pousse dans notre sens et qu'on est
        #     du bon côté de l'inventaire (pas en train de se mettre short
        #     contre un flux vendeur par exemple).
        if best_bid is not None and ofi_ewma > self.ofi_aggressive_join_threshold:
            # Flux acheteur côté bid → on joint le bid si on n'est pas trop long.
            if self.position_btc < self.risk_manager.max_position_btc(mid_price) * 0.5:
                bid_price = max(bid_price, best_bid)
        if best_ask is not None and ofi_ewma < -self.ofi_aggressive_join_threshold:
            # Flux vendeur côté ask → on joint l'ask si on n'est pas trop short.
            if self.position_btc > -self.risk_manager.max_position_btc(mid_price) * 0.5:
                ask_price = min(ask_price, best_ask)

        # 11) Taille de quote
        quote_size = self._effective_quote_size(mid_price) * size_bandit_multiplier
        quote_size = max(0.0, min(self.quote_size_btc, quote_size))

        # 12) Validation du risque (notionnel)
        bid_allowed = quote_size > 0 and self.risk_manager.can_add_position(
            current_position_btc=self.position_btc,
            proposed_trade_size_btc=quote_size,
            price=bid_price,
        )
        ask_allowed = quote_size > 0 and self.risk_manager.can_add_position(
            current_position_btc=self.position_btc,
            proposed_trade_size_btc=-quote_size,
            price=ask_price,
        )

        # 13) Reduce-only : on coupe le côté qui aggraverait l'inventaire.
        if bool(risk_status["reduce_only"]):
            if self.position_btc > 0:
                bid_allowed = False
                if best_ask is not None:
                    ask_price = best_ask
            elif self.position_btc < 0:
                ask_allowed = False
                if best_bid is not None:
                    bid_price = best_bid

        # 14) Émission
        now = utc_now()
        self.current_bid = (
            Quote("bid", bid_price, quote_size, now) if bid_allowed else None
        )
        self.current_ask = (
            Quote("ask", ask_price, quote_size, now) if ask_allowed else None
        )
        self.last_quote_update_ts_ms = now.timestamp() * 1000
        self.last_quote_context = {
            "half_spread_bps": half_spread_bps,
            "inventory_skew_bps": inventory_shift_bps,
            "signal_shift_bps": signal_shift_bps,
            "volatility_bps": volatility_bps,
            "inside_spread_bps": inside_spread_bps,
            "quote_size_btc": quote_size,
            "imbalance_l1": imbalance_l1,
            "imbalance_l3": imbalance_l3,
            "microprice_edge_bps": microprice_edge_bps,
            "ofi_ewma": ofi_ewma,
            "trade_flow_signed": trade_flow_signed,
            "vpin": vpin,
            "fast_vol_bps": fast_vol_bps,
            "combined_signal_bps": combined_signal_bps,
            "toxicity_widening_bps": toxicity_widening_bps,
            "fast_vol_widening_bps": fast_vol_widening_bps,
            "fair_price": fair_price,
            "bandit_reward": bandit_reward,
            "rolling_inside_spread_bps": rolling_inside_spread_bps,
            "ewma_inside_spread_bps": ewma_inside_spread_bps,
            **(self.bandit.diagnostics() if self.bandit is not None else {}),
            **risk_status,
        }
        return self.current_bid, self.current_ask

    # ------------------------------------------------------------------
    # Simulation de fill à partir d'un trade public
    # ------------------------------------------------------------------
    def _quote_competitiveness(
        self,
        side: str,
        quote_price: float,
        best_bid: float | None,
        best_ask: float | None,
        mid_price: float | None,
    ) -> float:
        """Probabilité relative d'être en tête de queue (0..1, bruit exp).

        Si on est au touch, compétitivité ≈ 1. Plus on est loin, plus elle
        décroît exponentiellement.
        """
        inside_spread_bps = self._inside_spread_bps(best_bid, best_ask, mid_price)
        scale_bps = max(0.10, inside_spread_bps / 2.0)
        if side == "bid" and best_bid is not None and mid_price is not None:
            distance_bps = max(0.0, (best_bid - quote_price) / mid_price * 10_000)
        elif side == "ask" and best_ask is not None and mid_price is not None:
            distance_bps = max(0.0, (quote_price - best_ask) / mid_price * 10_000)
        else:
            distance_bps = 0.0
        return float(exp(-distance_bps / scale_bps))

    def _simulated_fill_qty(
        self,
        trade: Trade,
        quote: Quote,
        best_bid: float | None,
        best_ask: float | None,
        touch_depth_qty: float,
        mid_price: float | None,
    ) -> float:
        """Quantité estimée fillée à partir d'un trade de marché.

        Modèle : quote_size × compétitivité × part_de_queue × intensity.
        """
        competitive = self._quote_competitiveness(
            quote.side, quote.price, best_bid, best_ask, mid_price
        )
        queue_ahead = max(0.0, touch_depth_qty * self.queue_ahead_factor)
        queue_share = quote.size / max(quote.size, queue_ahead + quote.size)
        expected_fill = trade.size * queue_share * competitive * self.fill_intensity
        return min(quote.size, max(0.0, expected_fill))

    def on_trade(
        self,
        trade: Trade,
        best_bid: float | None = None,
        best_ask: float | None = None,
        bid_touch_depth: float = 0.0,
        ask_touch_depth: float = 0.0,
        mid_price: float | None = None,
    ) -> list[Fill]:
        fills: list[Fill] = []
        now_ms = trade.time.timestamp() * 1000
        if now_ms - self.last_fill_ts_ms < self.cooldown_ms_after_fill:
            return fills

        # Convention Coinbase : `side` = côté du MAKER.
        # Si le maker est BUY → l'agresseur vend → peut toucher notre bid.
        if (
            trade.side == "BUY"
            and self.current_bid is not None
            and self.current_bid.size > 0
        ):
            eligible = (
                best_bid is not None and self.current_bid.price >= trade.price - 1e-9
            )
            if eligible:
                qty = self._simulated_fill_qty(
                    trade,
                    self.current_bid,
                    best_bid,
                    best_ask,
                    bid_touch_depth,
                    mid_price,
                )
                if qty > 1e-12 and self.risk_manager.can_execute_fill(
                    self.position_btc, qty, self.current_bid.price, self.last_equity
                ):
                    fills.append(self._execute_buy(trade, qty))
                    self.last_fill_ts_ms = now_ms

        elif (
            trade.side == "SELL"
            and self.current_ask is not None
            and self.current_ask.size > 0
        ):
            eligible = (
                best_ask is not None and self.current_ask.price <= trade.price + 1e-9
            )
            if eligible:
                qty = self._simulated_fill_qty(
                    trade,
                    self.current_ask,
                    best_bid,
                    best_ask,
                    ask_touch_depth,
                    mid_price,
                )
                if qty > 1e-12 and self.risk_manager.can_execute_fill(
                    self.position_btc, -qty, self.current_ask.price, self.last_equity
                ):
                    fills.append(self._execute_sell(trade, qty))
                    self.last_fill_ts_ms = now_ms
        return fills

    # ------------------------------------------------------------------
    # Exécution comptable : avg entry + realized P&L
    # ------------------------------------------------------------------
    def _execute_buy(self, trade: Trade, qty: float) -> Fill:
        price = self.current_bid.price if self.current_bid is not None else trade.price
        new_position = self.position_btc + qty
        if self.position_btc >= 0:
            total_cost = self.avg_entry_price * self.position_btc + price * qty
            self.avg_entry_price = total_cost / new_position if new_position else 0.0
        else:
            # On couvre un short : tout ce qu'on couvre génère du realized.
            closing_qty = min(qty, abs(self.position_btc))
            self.realized_pnl += (self.avg_entry_price - price) * closing_qty
            remainder = qty - closing_qty
            if remainder > 0:
                # Basculement short → long : reset avg au prix du fill.
                self.avg_entry_price = price
            elif abs(new_position) < 1e-12:
                self.avg_entry_price = 0.0
        self.position_btc = new_position
        self.cash_usd -= price * qty
        if self.current_bid is not None:
            self.current_bid.size = max(0.0, self.current_bid.size - qty)
        fill = Fill(
            time=trade.time, side="BUY", price=price, size=qty, reason="trade_hit_bid"
        )
        self.executions.append(
            {
                "time": trade.time.isoformat(),
                "side": fill.side,
                "price": fill.price,
                "size": fill.size,
                "reason": fill.reason,
            }
        )
        return fill

    def _execute_sell(self, trade: Trade, qty: float) -> Fill:
        price = self.current_ask.price if self.current_ask is not None else trade.price
        new_position = self.position_btc - qty
        if self.position_btc <= 0:
            # On renforce un short (ou on passe de flat à short).
            total_basis = self.avg_entry_price * abs(self.position_btc) + price * qty
            self.avg_entry_price = (
                total_basis / abs(new_position) if abs(new_position) > 1e-12 else 0.0
            )
        else:
            # On réduit un long : realized positif = (vente - entrée) * qty.
            closing_qty = min(qty, self.position_btc)
            self.realized_pnl += (price - self.avg_entry_price) * closing_qty
            remainder = qty - closing_qty
            if remainder > 0:
                # Basculement long → short.
                self.avg_entry_price = price
            elif abs(new_position) < 1e-12:
                self.avg_entry_price = 0.0
        self.position_btc = new_position
        self.cash_usd += price * qty
        if self.current_ask is not None:
            self.current_ask.size = max(0.0, self.current_ask.size - qty)
        fill = Fill(
            time=trade.time, side="SELL", price=price, size=qty, reason="trade_lift_ask"
        )
        self.executions.append(
            {
                "time": trade.time.isoformat(),
                "side": fill.side,
                "price": fill.price,
                "size": fill.size,
                "reason": fill.reason,
            }
        )
        return fill

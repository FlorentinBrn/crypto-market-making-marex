from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Settings:
    """Configuration centralisée de l'application."""

    # --------------------------------------------------------------
    # Flux de données
    # --------------------------------------------------------------
    product_id: str = "BTC-USD"
    ws_url: str = "wss://advanced-trade-ws.coinbase.com"

    # --------------------------------------------------------------
    # Quoting
    # --------------------------------------------------------------
    quote_size_btc: float = 0.10
    min_quote_size_btc: float = 0.02
    quote_refresh_interval_ms: int = 50
    base_half_spread_bps: float = 1.0
    min_half_spread_bps: float = 0.30
    max_half_spread_bps: float = 8.0
    inventory_skew_bps_per_btc: float = 6.0
    volatility_window: int = 120
    volatility_spread_multiplier: float = 1.2
    touch_join_threshold_bps: float = 3.0
    queue_ahead_factor: float = 1.0
    fill_intensity: float = 1.50
    cooldown_ms_after_fill: int = 80
    rolling_spread_window: int = 250
    ewma_spread_alpha: float = 0.08

    # --------------------------------------------------------------
    # Signaux de microstructure
    # --------------------------------------------------------------
    microprice_weight: float = 0.85
    imbalance_shift_bps: float = 3.0
    imbalance_widening_bps: float = 1.5

    ofi_ewma_alpha: float = 0.15
    trade_flow_window_ms: int = 4_000
    vpin_bucket_size_btc: float = 0.5
    vpin_history_size: int = 50
    fast_vol_alpha: float = 0.10

    combined_signal_weight: float = 1.0

    # Élargissement du spread en cas de toxicité élevée (VPIN). Exprimé
    # en bps additionnels par unité de VPIN au-dessus du seuil de référence.
    vpin_reference: float = 0.25
    vpin_widening_bps_per_unit: float = 4.0

    # Élargissement du spread en cas de volatilité rapide élevée.
    fast_vol_widening_coef: float = 0.5

    # Seuil d'OFI EWMA au-delà duquel la stratégie peut rejoindre
    # agressivement le touch du côté favorable.
    ofi_aggressive_join_threshold: float = 15.0

    # --------------------------------------------------------------
    # Reinforcement learning (optionnel, désactivé par défaut)
    # --------------------------------------------------------------
    use_contextual_bandit: bool = False

    # --------------------------------------------------------------
    # Risque
    # --------------------------------------------------------------
    reduce_only_loss_utilization: float = 0.75
    reduce_only_exposure_utilization: float = 0.85
    max_notional_usd: float = 1_000_000.0
    initial_cash_usd: float = 1_000_000.0
    max_loss_usd: float = 100_000.0

    # --------------------------------------------------------------
    # Dashboard
    # --------------------------------------------------------------
    top_levels_to_display: int = 8
    recent_trades_to_display: int = 20
    snapshot_interval_sec: float = 0.10
    dashboard_refresh_ms: int = 100
    dashboard_history_points: int = 750
    render_console: bool = True

    # --------------------------------------------------------------
    # Persistance CSV
    # --------------------------------------------------------------
    csv_flush_size: int = 250
    output_dir: Path = field(default_factory=lambda: Path("data"))

    # --------------------------------------------------------------
    # Instrumentation de latence
    # --------------------------------------------------------------
    bench_latency: bool = False

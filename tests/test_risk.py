from crypto_mm.core.risk import RiskManager


def _make_risk() -> RiskManager:
    return RiskManager(
        max_notional_usd=1_000_000,
        max_loss_usd=100_000,
        initial_equity=1_000_000,
    )


def test_can_continue_true_when_not_in_loss() -> None:
    r = _make_risk()
    assert r.can_continue(1_000_000) is True


def test_can_continue_false_when_loss_budget_exhausted() -> None:
    r = _make_risk()
    assert r.can_continue(899_999.0) is False  # drawdown > 100_000


def test_loss_utilization_capped_at_one() -> None:
    r = _make_risk()
    assert r.loss_utilization(500_000) == 1.0
    assert r.loss_utilization(1_000_000) == 0.0


def test_reduce_only_activates_on_high_loss() -> None:
    r = _make_risk()
    # 75% du budget de perte consommé
    assert r.should_reduce_only(1_000_000 - 75_000, 0.0, 100_000) is True


def test_reduce_only_activates_on_high_exposure() -> None:
    r = _make_risk()
    # 85% du notionnel max
    assert r.should_reduce_only(1_000_000, 8.5, 100_000) is True


def test_quote_size_multiplier_scales_down_with_stress() -> None:
    r = _make_risk()
    # Pas de stress
    m_low = r.quote_size_multiplier(1_000_000, 0.0, 100_000)
    # Stress moyen
    m_mid = r.quote_size_multiplier(1_000_000 - 50_000, 0.0, 100_000)
    assert m_low > m_mid
    assert m_low <= 1.0
    assert m_mid > 0


def test_quote_size_multiplier_zero_when_critical() -> None:
    r = _make_risk()
    # Stress >= 0.95 → taille 0
    m = r.quote_size_multiplier(1_000_000 - 96_000, 0.0, 100_000)
    assert m == 0.0


def test_can_add_position_respects_notional_cap() -> None:
    r = RiskManager(max_notional_usd=100_000, max_loss_usd=100_000, initial_equity=1_000_000)
    assert r.can_add_position(0.0, 1.0, 100_000) is True
    assert r.can_add_position(0.0, 1.1, 100_000) is False


def test_status_returns_all_expected_fields() -> None:
    r = _make_risk()
    status = r.status(1_000_000, 0.5, 100_000)
    for key in [
        "can_continue",
        "reduce_only",
        "loss_utilization",
        "exposure_utilization",
        "max_position_btc",
        "risk_level",
        "health_score",
        "alerts",
        "recommendations",
    ]:
        assert key in status


def test_health_score_high_when_flat_and_no_loss() -> None:
    r = _make_risk()
    score = r.health_score(1_000_000, 0.0, 100_000)
    assert score >= 90


def test_health_score_drops_with_drawdown() -> None:
    r = _make_risk()
    score_low_loss = r.health_score(1_000_000, 0.0, 100_000)
    score_high_loss = r.health_score(1_000_000 - 60_000, 0.0, 100_000)
    assert score_high_loss < score_low_loss


def test_risk_level_stop_when_kill_switch() -> None:
    r = _make_risk()
    status = r.status(800_000, 0.0, 100_000)
    assert status["risk_level"] == "STOP"

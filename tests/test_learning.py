from crypto_mm.core.learning import ContextualBanditQuoter


def test_bandit_learns_positive_reward_on_single_arm() -> None:
    bandit = ContextualBanditQuoter(
        spread_multipliers=[1.0], size_multipliers=[1.0], epsilon=0.0
    )
    decision = bandit.choose(
        vol_bps=1.0, imbalance_l3=0.2, position_btc=0.0, loss_utilization=0.0
    )
    assert decision.arm_idx == 0
    bandit.last_equity = 1_000_000.0
    reward = bandit.update(equity=1_000_010.0, mid_price=100_000.0, position_btc=0.0)
    assert reward is not None and reward > 0
    diag = bandit.diagnostics()
    assert diag["bandit_arm_count"] == 1
    assert diag["bandit_arm_value"] > 0


def test_bandit_state_bucketing_is_consistent() -> None:
    bandit = ContextualBanditQuoter()
    key1 = bandit._bucket(0.5, 0.0, 0.0, 0.0)
    key2 = bandit._bucket(0.6, 0.0, 0.0, 0.0)
    # Mêmes buckets (< 1.0 bps → lowvol, neutre, flat, ok)
    assert key1 == key2


def test_bandit_state_bucketing_differs_across_regimes() -> None:
    bandit = ContextualBanditQuoter()
    low = bandit._bucket(0.5, 0.0, 0.0, 0.0)
    high = bandit._bucket(3.0, 0.0, 0.0, 0.0)
    assert low != high


def test_reward_penalizes_inventory() -> None:
    bandit = ContextualBanditQuoter(
        spread_multipliers=[1.0], size_multipliers=[1.0], epsilon=0.0, inventory_penalty=2.0
    )
    bandit.choose(vol_bps=1.0, imbalance_l3=0.0, position_btc=0.0, loss_utilization=0.0)
    bandit.last_equity = 1_000_000.0
    reward = bandit.update(equity=1_000_010.0, mid_price=100_000.0, position_btc=1.0)
    # delta equity = 10, penalty = 2 * 1 = 2 → reward = 8.
    assert reward == 8.0

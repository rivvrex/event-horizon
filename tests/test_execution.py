"""Module 3 verification: weight drift, fee deduction, rebalance masking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_lab.execution import ExecutionConfig, PortfolioEngine, rebalance_mask

N_BARS = 180
SYMBOLS = ["A", "B", "C", "D"]


@pytest.fixture
def returns() -> pd.DataFrame:
    """Dispersed daily returns: drift must actually pull weights apart."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2022-01-03", periods=N_BARS)
    data = rng.normal(0.0004, 0.011, size=(N_BARS, len(SYMBOLS)))
    data += np.linspace(-0.001, 0.001, len(SYMBOLS))  # force weights to diverge
    return pd.DataFrame(data, index=idx, columns=SYMBOLS)


@pytest.fixture
def signals(returns: pd.DataFrame) -> pd.DataFrame:
    """Static long A,B / short C,D. Constant target isolates drift from signal churn."""
    sig = pd.DataFrame(0, index=returns.index, columns=SYMBOLS, dtype="int8")
    sig[["A", "B"]] = 1
    sig[["C", "D"]] = -1
    return sig


def _reference_weights(
    target: pd.DataFrame, rets: pd.DataFrame, rebal: pd.Series
) -> pd.DataFrame:
    """Deliberately slow bar-by-bar book simulation. Obviously correct."""
    out = np.zeros((len(rets), rets.shape[1]))
    w = np.zeros(rets.shape[1])
    cash = 1.0
    for t in range(len(rets)):
        if rebal.iloc[t]:
            w = target.iloc[t].to_numpy().copy()
            cash = 1.0 - w.sum()
        out[t] = w
        r = rets.iloc[t].to_numpy()
        nav = cash + (w * (1.0 + r)).sum()
        w = w * (1.0 + r) / nav          # renormalize to the new NAV
        cash = cash / nav
    return pd.DataFrame(out, index=rets.index, columns=rets.columns)


def test_weight_drift_matches_iterative_reference(
    signals: pd.DataFrame, returns: pd.DataFrame
) -> None:
    """Test 1: closed-form vectorized drift == bar-by-bar reference loop.

    This is the load-bearing test for Module 3. The vectorized `_held_weights`
    replaces a genuine recursion with two grouped cumprods; if that algebra is
    wrong, every downstream number is wrong in a way no smoke test would catch.
    """
    engine = PortfolioEngine(ExecutionConfig(rebalance_freq="M"))
    result = engine.run(signals, returns)

    rebal = rebalance_mask(returns.index, "M")
    target = engine._target_weights(signals.astype("float64"))
    reference = _reference_weights(target, returns, rebal)

    pd.testing.assert_frame_equal(result.weights, reference, atol=1e-12)

    # Drift must be real: between rebalances weights leave their target.
    off_rebal = ~rebal
    max_drift = (result.weights - target)[off_rebal].abs().to_numpy().max()
    assert max_drift > 1e-3, "weights never drifted; fixture is degenerate"

    # On a rebalance bar the book is restored exactly to target.
    on = rebal.to_numpy()
    np.testing.assert_allclose(
        result.weights[on].to_numpy(), target[on].to_numpy(), atol=1e-12
    )


def test_costs_deducted_only_on_rebalance_bars(
    signals: pd.DataFrame, returns: pd.DataFrame
) -> None:
    """Test 2: fee deduction + off-rebalance masking.

    Pins three things: turnover/costs are exactly zero off-rebalance, net is
    gross minus cost bar-for-bar, and turnover is charged against the DRIFTED
    book (not the stale target) so no phantom trades are invented.
    """
    cfg = ExecutionConfig(cost_bps=10.0, slippage_bps=5.0, rebalance_freq="M")
    result = PortfolioEngine(cfg).run(signals, returns)
    rebal = rebalance_mask(returns.index, "M")

    off = ~rebal.to_numpy()
    assert (result.turnover[off] == 0.0).all()
    assert (result.costs[off] == 0.0).all()
    assert result.trade_log["date"].isin(result.rebalance_dates).all()

    # Cost identity, and gross/net divergence in the right direction.
    np.testing.assert_allclose(
        result.costs.to_numpy(),
        result.turnover.to_numpy() * cfg.total_cost_rate,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        result.net_returns.to_numpy(),
        (result.gross_returns - result.costs).to_numpy(),
        atol=1e-15,
    )
    assert result.net_returns.sum() < result.gross_returns.sum()

    # Bar 0 is the initial entry: turnover == gross leverage, fully charged.
    np.testing.assert_allclose(result.turnover.iloc[0], cfg.gross_leverage, atol=1e-12)

    # Turnover is measured vs. the drifted book. Charging vs. the static target
    # would report ZERO here (target never changes) despite real rebalance trades.
    rebal_turnover = result.turnover[rebal.to_numpy()].iloc[1:]
    assert (rebal_turnover > 0).any(), "drift-aware turnover collapsed to zero"

    # Zero-cost config must reproduce gross exactly.
    free = PortfolioEngine(
        ExecutionConfig(cost_bps=0.0, slippage_bps=0.0, rebalance_freq="M")
    ).run(signals, returns)
    pd.testing.assert_series_equal(
        free.net_returns, free.gross_returns, check_names=False
    )


def test_rebalance_frequency_masking(
    signals: pd.DataFrame, returns: pd.DataFrame
) -> None:
    """Test 3: D/W/M schedules produce the right cadence and cost ordering."""
    idx = returns.index
    counts = {f: int(rebalance_mask(idx, f).sum()) for f in ("D", "W", "M")}

    assert counts["D"] == len(idx)
    assert counts["W"] == idx.to_period("W").nunique()
    assert counts["M"] == idx.to_period("M").nunique()
    assert counts["D"] > counts["W"] > counts["M"]

    # Every schedule must trade on bar 0 (initial entry).
    for freq in ("D", "W", "M"):
        assert bool(rebalance_mask(idx, freq).iloc[0]) is True

    # More frequent rebalancing => more turnover => more cost drag.
    drag = {}
    for freq in ("D", "W", "M"):
        res = PortfolioEngine(ExecutionConfig(rebalance_freq=freq)).run(
            signals, returns
        )
        drag[freq] = res.costs.sum()
        assert len(res.rebalance_dates) == counts[freq]
    assert drag["D"] > drag["W"] > drag["M"]

    with pytest.raises(ValueError, match="unsupported rebalance_freq"):
        ExecutionConfig(rebalance_freq="fortnightly")


def test_buy_and_hold_identity() -> None:
    """Guard: a single untouched position must track the asset exactly."""
    idx = pd.bdate_range("2022-01-03", periods=60)
    rets = pd.DataFrame({"A": np.full(60, 0.01)}, index=idx)
    sig = pd.DataFrame(1, index=idx, columns=["A"], dtype="int8")

    res = PortfolioEngine(
        ExecutionConfig(cost_bps=0.0, slippage_bps=0.0, rebalance_freq="M")
    ).run(sig, rets)

    np.testing.assert_allclose(res.gross_returns.to_numpy(), 0.01, atol=1e-12)
    np.testing.assert_allclose(res.portfolio_value.iloc[-1], 100_000 * 1.01**60)
    # 100% in one name: the weight is pinned at 1.0, so there is nothing to drift.
    np.testing.assert_allclose(res.weights["A"].to_numpy(), 1.0, atol=1e-12)
    assert len(res.trade_log) == 1  # entry only, never re-traded

"""Module 4 verification.

Both tests are anchored on CLOSED-FORM cases where the correct answer is known
analytically. A metrics suite that only checks self-consistency will happily
agree with itself while being uniformly wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_lab.metrics import (
    BenchmarkLoader,
    PerformanceAnalyzer,
    Periodicity,
    resample_returns,
)


@pytest.fixture
def analyzer() -> PerformanceAnalyzer:
    return PerformanceAnalyzer()


def test_metrics_match_closed_form_values(analyzer: PerformanceAnalyzer) -> None:
    """Test 1: CAGR, vol, Sharpe, Sortino and max drawdown vs. hand math."""
    # --- Exact CAGR. 24 month-end bars from 2020-01-31 to 2021-12-31 span 700
    #     calendar days; plus one ~30.44-day period, that is 2.0 years to within
    #     0.06 days. The residual is a genuine calendar artifact (2020 is a leap
    #     year, and the 365.25-day convention averages that out), so the
    #     tolerance is 1e-3 rather than machine epsilon.
    monthly_idx = pd.date_range("2020-01-31", periods=24, freq="BME")
    r = 0.01
    m0 = analyzer.analyze(pd.Series(r, index=monthly_idx))
    assert m0.periodicity == "MONTHLY"
    np.testing.assert_allclose(m0.years, 2.0, rtol=1e-3)
    np.testing.assert_allclose(m0.cagr, (1 + r) ** 12 - 1, rtol=1e-3)
    np.testing.assert_allclose(m0.total_return, (1 + r) ** 24 - 1, rtol=1e-12)

    # --- Constant growth: volatility and drawdown are exactly zero.
    idx = pd.bdate_range("2020-01-01", periods=252)
    flat = pd.Series(0.001, index=idx)
    m = analyzer.analyze(flat)

    assert m.periodicity == "DAILY"
    np.testing.assert_allclose(m.total_return, 1.001**252 - 1, rtol=1e-12)
    np.testing.assert_allclose(m.annual_volatility, 0.0, atol=1e-12)
    np.testing.assert_allclose(m.max_drawdown, 0.0, atol=1e-15)
    assert m.hit_rate == 1.0
    assert np.isinf(m.sortino_ratio)   # zero downside, positive mean
    assert np.isnan(m.sharpe_ratio)    # zero stdev -> undefined, not infinite
    # CAGR must compound back to the realized equity over the elapsed span.
    np.testing.assert_allclose((1 + m.cagr) ** m.years, 1.001**252, rtol=1e-10)

    # --- Two-state series: every statistic is computable by hand.
    #     +2% and -1% alternating, starting positive, over 252 bars.
    alt = pd.Series(np.tile([0.02, -0.01], 126), index=idx)
    m2 = analyzer.analyze(alt)

    mean, sd = np.mean(alt), np.std(alt, ddof=1)
    np.testing.assert_allclose(m2.sharpe_ratio, mean / sd * np.sqrt(252), rtol=1e-10)
    np.testing.assert_allclose(m2.annual_volatility, sd * np.sqrt(252), rtol=1e-10)

    # Sortino: downside deviation divides by ALL n, not just losing periods.
    downside = np.sqrt(np.mean(np.minimum(alt, 0.0) ** 2))
    np.testing.assert_allclose(
        m2.sortino_ratio, mean / downside * np.sqrt(252), rtol=1e-10
    )
    # Half the periods are -1%, so dividing by the loss count instead would
    # shrink the denominator by sqrt(2) and inflate Sortino by the same factor.
    wrong = np.sqrt(np.mean(np.minimum(alt, 0.0)[alt < 0] ** 2))
    assert not np.isclose(mean / downside, mean / wrong)

    # --- Known drawdown: +50% then -40% gives exactly -40% peak-to-trough.
    path = pd.Series([0.5, -0.4, 0.1], index=pd.bdate_range("2021-01-01", periods=3))
    m3 = analyzer.analyze(path)
    np.testing.assert_allclose(m3.max_drawdown, -0.4, rtol=1e-12)

    info = analyzer.drawdown_info(path)
    assert info.peak_date == path.index[0]
    assert info.trough_date == path.index[1]
    assert info.recovery_date is None          # 1.5 -> 0.9 -> 0.99, never recovers
    assert not info.is_recovered

    # Drawdown must be consistent with the plotted equity curve, by construction.
    eq = analyzer.equity_curve(path)
    dd = analyzer.drawdown_series(eq)
    np.testing.assert_allclose(dd.min(), m3.max_drawdown, rtol=1e-15)
    assert (dd <= 1e-15).all() and (dd >= -1.0).all()


def test_annualization_follows_observation_frequency() -> None:
    """Test 2: the sqrt(252)/sqrt(52)/sqrt(12) mapping, and the trap it prevents.

    Annualization must key off how often returns are OBSERVED. Scaling a daily
    series by sqrt(12) because the strategy rebalances monthly inflates Sharpe
    by sqrt(252/12) ~ 4.6x. Resampling first is what makes sqrt(12) legitimate.
    """
    assert Periodicity.DAILY.annualization == pytest.approx(np.sqrt(252))
    assert Periodicity.WEEKLY.annualization == pytest.approx(np.sqrt(52))
    assert Periodicity.MONTHLY.annualization == pytest.approx(np.sqrt(12))

    # Inference from index spacing.
    daily_idx = pd.bdate_range("2020-01-01", periods=100)
    assert Periodicity.infer(daily_idx) is Periodicity.DAILY
    assert Periodicity.infer(
        pd.date_range("2020-01-01", periods=60, freq="W-FRI")
    ) is Periodicity.WEEKLY
    assert Periodicity.infer(
        pd.date_range("2020-01-01", periods=36, freq="BME")
    ) is Periodicity.MONTHLY

    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2015-01-01", periods=252 * 6)
    daily = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)

    correct = PerformanceAnalyzer().analyze(daily)
    assert correct.periodicity == "DAILY"

    # The bug: daily observations scaled as if monthly.
    buggy = PerformanceAnalyzer(periodicity=Periodicity.MONTHLY).analyze(daily)
    np.testing.assert_allclose(
        buggy.sharpe_ratio, correct.sharpe_ratio * np.sqrt(12 / 252), rtol=1e-10
    )
    assert abs(buggy.sharpe_ratio) < abs(correct.sharpe_ratio)

    # The fix: resample first, then sqrt(12) is genuinely correct.
    monthly = PerformanceAnalyzer(resample_to=Periodicity.MONTHLY).analyze(daily)
    assert monthly.periodicity == "MONTHLY"
    assert monthly.n_periods == pytest.approx(72, abs=2)

    # CAGR is frequency-INVARIANT: it depends only on endpoints and elapsed time.
    np.testing.assert_allclose(monthly.cagr, correct.cagr, rtol=0.02)
    # Sharpe is not invariant, but must stay in the same ballpark, not 4.6x off.
    assert 0.4 < monthly.sharpe_ratio / correct.sharpe_ratio < 2.5

    # Resampling compounds, never sums.
    compounded = resample_returns(daily, Periodicity.MONTHLY)
    np.testing.assert_allclose(
        (1 + compounded).prod(), (1 + daily).prod(), rtol=1e-10
    )
    with pytest.raises(ValueError, match="cannot upsample"):
        resample_returns(compounded, Periodicity.DAILY)


def test_benchmark_comparison_recovers_known_beta() -> None:
    """Guard: a synthetic 1.5-beta stream must regress back to beta 1.5."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2019-01-01", periods=1000)
    bench = pd.Series(rng.normal(0.0004, 0.009, len(idx)), index=idx)
    noise = pd.Series(rng.normal(0.0, 0.001, len(idx)), index=idx)
    strat = 1.5 * bench + 0.0002 + noise  # beta 1.5, ~5%/yr alpha

    cmp = PerformanceAnalyzer().compare(strat, bench)

    np.testing.assert_allclose(cmp.beta, 1.5, rtol=0.05)
    np.testing.assert_allclose(cmp.alpha, 0.0002 * 252, rtol=0.15)
    assert cmp.r_squared > 0.98
    assert cmp.up_capture > 1.0 and cmp.down_capture > 1.0  # levered both ways

    # Identical streams: beta 1, zero alpha, no tracking error.
    same = PerformanceAnalyzer().compare(bench, bench)
    np.testing.assert_allclose(same.beta, 1.0, atol=1e-10)
    np.testing.assert_allclose(same.alpha, 0.0, atol=1e-10)
    np.testing.assert_allclose(same.tracking_error, 0.0, atol=1e-12)
    np.testing.assert_allclose(same.excess_cagr, 0.0, atol=1e-12)

    # Comparison must align on shared dates only.
    shifted = PerformanceAnalyzer().compare(strat.iloc[200:], bench)
    np.testing.assert_allclose(shifted.beta, 1.5, rtol=0.08)

    with pytest.raises(ValueError, match="overlapping"):
        PerformanceAnalyzer().compare(strat.iloc[:2], bench.iloc[900:])


def test_equity_curve_is_the_single_source_of_truth(
    analyzer: PerformanceAnalyzer,
) -> None:
    """Guard: reported total return and drawdown must match the plotted curve."""
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2020-01-01", periods=500)
    rets = pd.Series(rng.normal(0.0003, 0.012, 500), index=idx)

    m = analyzer.analyze(rets)
    eq = analyzer.equity_curve(rets)

    np.testing.assert_allclose(m.total_return, eq.iloc[-1] - 1.0, rtol=1e-12)
    np.testing.assert_allclose(
        m.max_drawdown, analyzer.drawdown_series(eq).min(), rtol=1e-12
    )
    np.testing.assert_allclose(
        (1 + m.cagr) ** m.years, eq.iloc[-1], rtol=1e-10
    )

    # A wiped-out curve must not produce a complex/nonsense CAGR.
    ruin = pd.Series([-0.5, -0.6, -1.0], index=pd.bdate_range("2020-01-01", periods=3))
    assert np.isnan(analyzer.analyze(ruin).cagr)

    # Equal-weight benchmark helper.
    panel = pd.DataFrame({"A": rets, "B": rets * 2.0})
    np.testing.assert_allclose(
        BenchmarkLoader.equal_weight(panel).to_numpy(),
        (rets * 1.5).to_numpy(),
        rtol=1e-12,
    )

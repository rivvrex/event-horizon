"""Module 2 verification.

Test 1 proves the lookahead contract holds end-to-end (including composites).
Test 2 proves the vectorized rolling math equals a naive reference loop, which
is the claim that justifies banning loops in the first place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_lab.data import MarketData
from factor_lab.signals import (
    CompositeFactor,
    Factor,
    MeanReversion,
    Momentum,
    SignalConfig,
    SignalGenerator,
    Volatility,
)

N_BARS = 300
N_SYMBOLS = 8


@pytest.fixture
def market() -> MarketData:
    """Deterministic multi-asset panel with dispersed drifts (non-degenerate ranks)."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=N_BARS)
    symbols = [f"S{i}" for i in range(N_SYMBOLS)]
    drift = np.linspace(-0.0006, 0.0010, N_SYMBOLS)
    log_rets = rng.normal(0.0, 0.012, size=(N_BARS, N_SYMBOLS)) + drift
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(log_rets, axis=0)), index=dates, columns=symbols
    )
    return MarketData.from_prices(prices)


def _build_composite() -> Factor:
    return CompositeFactor(
        {Momentum(60, 10): 1.0, MeanReversion(15): 1.0, Volatility(20): 0.5}
    )


@pytest.mark.parametrize(
    "make_factor",
    [
        lambda: Momentum(60, 10),
        lambda: MeanReversion(15),
        lambda: Volatility(20),
        _build_composite,
    ],
    ids=["momentum", "mean_reversion", "volatility", "composite"],
)
def test_future_prices_cannot_change_past_signals(
    market: MarketData, make_factor: object
) -> None:
    """Test 1: the lookahead firewall.

    Triple every price from bar `k` onward -- a change no real factor could
    possibly anticipate -- and assert every signal up to AND INCLUDING bar `k`
    is bit-identical. Signal[k] is only allowed to see data through bar k-1, so
    perturbing bar k must not move it.

    The composite case additionally pins down that blending does not apply the
    lag twice: a double shift would still pass the equality check but would
    shift the first valid signal one bar later, which the tail assertion catches.
    """
    factor: Factor = make_factor()  # type: ignore[operator]
    gen = SignalGenerator(SignalConfig(min_universe=2))

    baseline = gen.from_factor(factor, market)

    k = 200
    tampered_prices = market.prices.copy()
    # The multiplier MUST vary by symbol. A uniform scale is, in log space, an
    # identical additive shift to every name's score, which a cross-sectional
    # z-score cancels exactly -- ranks would be unchanged and the sanity check
    # below would fire against perfectly correct code.
    tampered_prices.iloc[k:] *= np.linspace(0.4, 3.0, N_SYMBOLS)
    tampered = gen.from_factor(factor, MarketData.from_prices(tampered_prices))

    pd.testing.assert_frame_equal(baseline.iloc[: k + 1], tampered.iloc[: k + 1])

    # Sanity: the tamper must actually bite somewhere after k, otherwise this
    # test would pass trivially against a factor that ignores prices entirely.
    assert not baseline.iloc[k + 1 :].equals(tampered.iloc[k + 1 :])

    # The lag is applied exactly once: scores exist from `warmup`-ish onward, so
    # the first non-flat bar must not be pushed later than the factor's warmup.
    first_active = int(np.argmax((baseline != 0).any(axis=1).to_numpy()))
    assert first_active <= factor.warmup, (
        f"first signal at bar {first_active} exceeds warmup {factor.warmup}: "
        "the lag was likely applied more than once"
    )


def test_vectorized_momentum_matches_reference_loop(market: MarketData) -> None:
    """Test 2: vectorized rolling math == naive loop, and signals are well-formed."""
    lookback, skip = 10, 3
    factor = Momentum(lookback, skip)
    vectorized = factor._raw_scores(market)

    # Deliberately slow, obviously-correct reference implementation.
    lr = market.log_returns
    reference = pd.DataFrame(np.nan, index=lr.index, columns=lr.columns)
    for t in range(len(lr)):
        end = t - skip
        start = end - lookback + 1
        if start < 0:
            continue
        window = lr.iloc[start : end + 1]
        if window.isna().to_numpy().any():
            continue
        reference.iloc[t] = window.sum().to_numpy()

    pd.testing.assert_frame_equal(vectorized, reference, atol=1e-12)

    # Signal well-formedness.
    signals = SignalGenerator(SignalConfig(0.25, 0.25)).generate(
        factor.compute(market)
    )
    assert set(np.unique(signals.to_numpy())) <= {-1, 0, 1}
    assert signals.dtypes.eq("int8").all()

    # A long/short book must be name-balanced: 8 valid symbols, 25% per side.
    live = signals.loc[(signals != 0).any(axis=1)]
    np.testing.assert_array_equal(
        (live > 0).sum(axis=1).to_numpy(), (live < 0).sum(axis=1).to_numpy()
    )
    assert (live > 0).sum(axis=1).eq(2).all()  # floor(8 * 0.25) == 2


def test_long_only_and_tiny_universe_degrade_gracefully() -> None:
    """Guard: 3-name universe must still produce a short book (the pct-rank trap)."""
    dates = pd.bdate_range("2021-01-01", periods=40)
    prices = pd.DataFrame(
        {
            "A": np.linspace(100, 130, 40),
            "B": np.linspace(100, 100, 40) + np.sin(np.arange(40)),
            "C": np.linspace(100, 80, 40),
        },
        index=dates,
    )
    md = MarketData.from_prices(prices)
    scores = MeanReversion(5).compute(md)

    ls = SignalGenerator(SignalConfig(0.3, 0.3, allow_short=True)).generate(scores)
    live = ls.loc[(ls != 0).any(axis=1)]
    assert not live.empty, "3-name universe produced no positions at all"
    assert (live > 0).sum(axis=1).eq(1).all()
    assert (live < 0).sum(axis=1).eq(1).all()

    long_only = SignalGenerator(
        SignalConfig(0.3, 0.3, allow_short=False)
    ).generate(scores)
    assert (long_only >= 0).all().all()

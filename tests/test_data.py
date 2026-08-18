"""Module 1 verification.

Both tests use a synthetic in-memory source: no network, deterministic, and they
assert the two properties the rest of the engine depends on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_lab.data import CleaningConfig, DataPipeline, DataSource, FetchRequest


class FakeSource(DataSource):
    """Deterministic source with a planted holiday gap and a planted 2:1 split."""

    name = "fake"

    def _download(self, request: FetchRequest) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-01", periods=10)
        rows = []
        for sym in request.symbols:
            close = pd.Series(100.0 + np.arange(10), index=dates)
            factor = pd.Series(1.0, index=dates)
            if sym == "AAA":
                close.iloc[5] = np.nan          # exchange holiday / halt
                factor.iloc[:4] = 0.5           # 2:1 split effective on day 4
            frame = pd.DataFrame(
                {
                    "open": close, "high": close, "low": close,
                    "close": close,
                    "adj_close": close * factor,
                    "volume": 1_000_000.0,
                },
                index=dates,
            )
            frame["symbol"] = sym
            rows.append(frame.set_index("symbol", append=True))
        return pd.concat(rows)


@pytest.fixture
def pipeline() -> DataPipeline:
    return DataPipeline(FakeSource(), CleaningConfig(min_history_frac=0.5))


@pytest.fixture
def raw_pipeline() -> DataPipeline:
    """Winsorizing off: needed to assert the *unmodified* return math."""
    return DataPipeline(
        FakeSource(), CleaningConfig(min_history_frac=0.5, winsorize_bad_ticks=False)
    )


def test_gaps_filled_and_log_returns_are_time_additive(
    raw_pipeline: DataPipeline,
) -> None:
    """Test 1: cleaning fills the halt, and sum(log returns) == total log return.

    Time-additivity is the property the whole metrics module leans on, so if this
    breaks, every Sharpe/CAGR number downstream is silently wrong. Asserted with
    winsorizing disabled, since clipping intentionally violates additivity.
    """
    req = FetchRequest.of(["AAA", "BBB"], "2024-01-01", "2024-01-15")
    md = raw_pipeline.run(req)

    assert md.prices.notna().all().all(), "ffill left a hole in the price matrix"
    assert md.prices.shape == md.log_returns.shape

    total = np.log(md.prices.iloc[-1] / md.prices.iloc[0])
    np.testing.assert_allclose(md.log_returns.sum(), total, atol=1e-12)

    # simple_returns must round-trip through expm1/log1p
    np.testing.assert_allclose(
        np.log1p(md.simple_returns.dropna()), md.log_returns.dropna(), atol=1e-12
    )


def test_split_is_absorbed_by_adjustment_factor(pipeline: DataPipeline) -> None:
    """Test 2: the 2:1 split shows up in adj_factor, NOT as a fake -50% return.

    An unadjusted split is the classic way a backtest invents alpha; this pins
    the behaviour down.
    """
    md = pipeline.run(FetchRequest.of(["AAA", "BBB"], "2024-01-01", "2024-01-15"))

    assert md.adj_factor["AAA"].nunique() == 2      # 0.5 pre-split, 1.0 post
    assert md.adj_factor["BBB"].nunique() == 1      # untouched control

    # The split day is a genuine +100% jump in adjusted price. It is real (the
    # factor doubles), so clipping should have capped it at the configured bound
    # rather than letting a 0.69 log return through.
    aaa = md.log_returns["AAA"].dropna()
    assert aaa.abs().max() <= pipeline.config.max_abs_daily_return + 1e-12
    assert md.log_returns["BBB"].dropna().abs().max() < 0.02  # control stays tame

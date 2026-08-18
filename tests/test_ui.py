"""Module 5 verification.

Both tests run entirely OFFLINE against a synthetic `DataSource`. A UI test that
needs the network is a test that fails on a train, and yfinance rate limits will
eventually make it fail everywhere.

The point of these tests is not "does Streamlit render" -- Streamlit is not the
part that can be silently wrong. The point is that the orchestration layer wires
Modules 1-4 together without breaking the invariants each module guarantees, and
that every chart builder can consume the real result object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from factor_lab.data import DataSource, FetchRequest
from factor_lab.metrics import PerformanceAnalyzer
from factor_lab.signals import CompositeFactor
from factor_lab.ui import charts
from factor_lab.ui.app import RunSpec, _metrics_table, _pooled_factor_panel
from factor_lab.ui.backtest import BacktestResult, build_factor, run_backtest

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
BENCH = "BENCH"
START = pd.Timestamp("2018-01-02").date()
END = pd.Timestamp("2023-12-29").date()


class SyntheticSource(DataSource):
    """Deterministic geometric-Brownian prices with per-symbol drift.

    Dispersed drifts matter: an undispersed universe gives the cross-sectional
    rank nothing to separate, so the book would be arbitrary and the test would
    pass regardless of whether the factor math worked.
    """

    name = "synthetic"

    def _download(self, request: FetchRequest) -> pd.DataFrame:
        idx = pd.bdate_range(request.start, request.end)
        rng = np.random.default_rng(42)
        frames = []
        for i, symbol in enumerate(request.symbols):
            drift = 0.0002 + 0.0004 * (i / max(len(request.symbols) - 1, 1))
            rets = rng.normal(drift, 0.013, len(idx))
            close = 50.0 * (1.0 + i) * np.exp(np.cumsum(rets))
            frames.append(
                pd.DataFrame(
                    {
                        "open": close, "high": close * 1.004, "low": close * 0.996,
                        "close": close, "adj_close": close,
                        "volume": np.full(len(idx), 1e6),
                        "symbol": symbol,
                    },
                    index=idx,
                ).set_index("symbol", append=True)
            )
        return pd.concat(frames)


@pytest.fixture(scope="module")
def result() -> BacktestResult:
    return run_backtest(
        symbols=SYMBOLS,
        start=START,
        end=END,
        factor_names=["Momentum", "Volatility"],
        factor_params={"mom_lookback": 126, "mom_skip": 21, "vol_window": 63},
        rebalance_freq="W",
        cost_bps=10.0,
        slippage_bps=5.0,
        benchmark_symbol=BENCH,
        source=SyntheticSource(),
    )


def test_orchestration_preserves_module_invariants(result: BacktestResult) -> None:
    """Test 1: the pipeline composes without violating any module's contract."""
    # --- Module 2: the lookahead lag survives composition. A composite applies
    #     the shift once at the top, so the first tradeable bar must be strictly
    #     later than the first bar with a finite raw score.
    assert result.factor.lag == 1
    assert result.factor.name == "Momentum + Volatility"
    raw_first = result.factor._normalized_scores(result.market).first_valid_index()
    assert result.scores.first_valid_index() > raw_first

    # --- Module 3: gross/net/cost identity holds bar for bar.
    ex = result.execution
    np.testing.assert_allclose(
        ex.net_returns.to_numpy(), (ex.gross_returns - ex.costs).to_numpy(), atol=1e-15
    )
    assert (ex.turnover[~ex.turnover.index.isin(ex.rebalance_dates)] == 0.0).all()
    assert ex.total_cost_paid > 0.0

    # --- Module 4: reported metrics describe the equity curve the UI plots.
    #     This is the invariant most easily broken by a trimming bug: slice the
    #     returns for the chart but not for the metrics and the two disagree.
    equity = result.equity
    np.testing.assert_allclose(
        result.metrics.total_return,
        equity.iloc[-1] / ex.config.initial_capital - 1.0,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        result.metrics.max_drawdown,
        (equity / equity.cummax() - 1.0).min(),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        (1.0 + result.metrics.cagr) ** result.metrics.years,
        equity.iloc[-1] / ex.config.initial_capital,
        rtol=1e-8,
    )
    recomputed = PerformanceAnalyzer().analyze(result.net_returns)
    assert recomputed.sharpe_ratio == pytest.approx(result.metrics.sharpe_ratio)

    # --- Warm-up trimming: the analysis window starts once the book is live,
    #     and costs make net strictly worse than gross.
    assert result.net_returns.index[0] > ex.net_returns.index[0]
    assert (result.signals.loc[result.net_returns.index[0]] != 0).any()
    assert result.metrics.cagr < result.gross_metrics.cagr

    # --- Benchmark alignment: identical index, no NaNs, comparison well-formed.
    assert result.benchmark.index.equals(result.net_returns.index)
    assert result.equal_weight.index.equals(result.net_returns.index)
    assert not result.benchmark.isna().any()
    assert -1.0 <= result.comparison.correlation <= 1.0
    assert set(result.component_scores) == {"Momentum", "Volatility"}

    # A single-factor run must NOT be wrapped in a composite.
    solo = build_factor(["Momentum"], {"mom_lookback": 126, "mom_skip": 21})
    assert not isinstance(solo, CompositeFactor)
    assert solo.name.startswith("Momentum")
    with pytest.raises(ValueError, match="at least one factor"):
        build_factor([], {})


def test_charts_and_helpers_consume_the_result(result: BacktestResult) -> None:
    """Test 2: every figure builds from the real result, with correct data.

    Plotly fails late and quietly -- a bad column name yields an empty trace, not
    an exception -- so each figure is checked for actual data, not just for type.
    """
    equity, bench = result.equity, result.benchmark
    figs: dict[str, go.Figure] = {
        "equity": charts.equity_curve(
            {"Strategy": equity, BENCH: (1.0 + bench).cumprod()}
        ),
        "equity_log": charts.equity_curve({"Strategy": equity}, log_scale=True),
        "drawdown": charts.drawdown_chart({"Strategy": equity / equity.cummax() - 1.0}),
        "rolling": charts.rolling_performance(result.net_returns, bench, window=126),
        "corr": charts.correlation_heatmap(
            result.market.simple_returns.dropna(how="all"), "Assets"
        ),
        "weights": charts.weight_area(result.execution.weights),
        "exposure": charts.exposure_chart(result.execution.weights),
        "turnover": charts.turnover_chart(
            result.execution.turnover, result.execution.costs
        ),
        "distribution": charts.return_distribution(
            result.net_returns, result.metrics.var_95
        ),
        "monthly": charts.monthly_heatmap(result.net_returns),
        "scores": charts.factor_score_chart(result.scores),
    }
    for name, fig in figs.items():
        assert isinstance(fig, go.Figure), name
        assert fig.data, f"{name} has no traces"
        assert all(
            np.isfinite(np.asarray(t.z, dtype="float64")).any()
            if getattr(t, "z", None) is not None
            else len(t.x) > 0
            for t in fig.data
        ), f"{name} has an empty trace"

    assert figs["equity_log"].layout.yaxis.type == "log"
    # Rebasing is what makes a $100k curve comparable to a $450 share price.
    for trace in figs["equity"].data:
        assert trace.y[0] == pytest.approx(100.0)

    # Diverging scale centred at zero, or the SIGN of a correlation is invisible.
    heat = figs["corr"].data[0]
    assert (heat.zmid, heat.zmin, heat.zmax) == (0.0, -1.0, 1.0)

    # Pooled factor panel: one column per factor, one row per (date, symbol).
    panel = _pooled_factor_panel(result.component_scores)
    assert list(panel.columns) == ["Momentum", "Volatility"]
    assert isinstance(panel.index, pd.MultiIndex)
    assert not panel.isna().to_numpy().any()
    assert _pooled_factor_panel({}).empty

    table = _metrics_table(
        {
            "net": result.metrics,
            "gross": result.gross_metrics,
            BENCH: result.benchmark_metrics,
        }
    )
    assert list(table.columns) == ["net", "gross", BENCH]
    assert table.loc["CAGR", "net"].endswith("%")
    assert "n/a" not in table.loc["Sharpe ratio", "net"]

    # RunSpec is the cache key: frozen and hashable, so Streamlit can hash it and
    # two identical configurations cannot produce two different backtests.
    spec = RunSpec(
        symbols=tuple(SYMBOLS), start=START, end=END, factor_names=("Momentum",),
        params=(("mom_lookback", 126),), weights=(), long_quantile=0.3,
        short_quantile=0.3, allow_short=True, rebalance_freq="W", cost_bps=10.0,
        slippage_bps=5.0, initial_capital=100_000.0, risk_free_rate=0.0,
        benchmark_symbol=BENCH,
    )
    assert hash(spec) == hash(
        RunSpec(**{**{f: getattr(spec, f) for f in spec.__slots__}})
    )
    with pytest.raises((AttributeError, TypeError)):
        spec.cost_bps = 20.0  # type: ignore[misc]

"""Backtest orchestration: one call that wires Modules 1-4 together.

Kept OUT of `app.py` so the pipeline is testable and callable from a notebook or
a scheduled job without importing Streamlit. `app.py` owns widgets and layout;
this owns the actual run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from factor_lab.data import (
    CachedDataSource,
    CleaningConfig,
    DataPipeline,
    DataSource,
    FetchRequest,
    MarketData,
    YFinanceSource,
)
from factor_lab.execution import ExecutionConfig, ExecutionResult, PortfolioEngine
from factor_lab.metrics import (
    BenchmarkComparison,
    BenchmarkLoader,
    DrawdownInfo,
    PerformanceAnalyzer,
    PerformanceMetrics,
)
from factor_lab.signals import (
    CompositeFactor,
    Factor,
    MeanReversion,
    Momentum,
    SignalConfig,
    SignalGenerator,
    Value,
    Volatility,
)
from factor_lab.types import SignalFrame

FACTOR_BUILDERS: dict[str, type[Factor]] = {
    "Momentum": Momentum,
    "Mean Reversion": MeanReversion,
    "Volatility": Volatility,
    "Value": Value,
}


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Everything the UI renders. Frozen so a cached run cannot be mutated."""

    market: MarketData
    factor: Factor
    scores: pd.DataFrame
    signals: SignalFrame
    execution: ExecutionResult
    net_returns: pd.Series      # trimmed to the live window; what metrics describe
    benchmark: pd.Series
    equal_weight: pd.Series
    metrics: PerformanceMetrics
    gross_metrics: PerformanceMetrics
    benchmark_metrics: PerformanceMetrics
    comparison: BenchmarkComparison
    drawdown: DrawdownInfo
    component_scores: dict[str, pd.DataFrame]

    @property
    def equity(self) -> pd.Series:
        """Equity over the live window only, rebased to the initial capital.

        The full `portfolio_value` includes the factor warm-up, during which the
        book is structurally flat. Charting that produces a long dead segment
        that makes the strategy look calmer than it is.
        """
        capital = self.execution.config.initial_capital
        return (capital * (1.0 + self.net_returns).cumprod()).rename("equity")


def build_factor(
    names: list[str],
    params: dict[str, int],
    weights: dict[str, float] | None = None,
) -> Factor:
    """One factor, or an equal/custom-weighted composite of several."""
    if not names:
        raise ValueError("select at least one factor")

    built: dict[str, Factor] = {}
    for name in names:
        # The registry membership check is the guard, not a constructor source:
        # without it an unknown name would fall through to `else` and silently
        # build a Value factor.
        if name not in FACTOR_BUILDERS:
            raise KeyError(f"unknown factor: {name!r}")
        if name == "Momentum":
            built[name] = Momentum(
                params.get("mom_lookback", 231), params.get("mom_skip", 21)
            )
        elif name == "Mean Reversion":
            built[name] = MeanReversion(params.get("mr_window", 21))
        elif name == "Volatility":
            built[name] = Volatility(params.get("vol_window", 63))
        else:
            built[name] = Value(params.get("value_lookback", 756))

    if len(built) == 1:
        return next(iter(built.values()))

    w = weights or {}
    return CompositeFactor(
        {f: w.get(n, 1.0) for n, f in built.items()},
        name=" + ".join(built),
    )


def default_source(cache_dir: str = ".cache") -> DataSource:
    return CachedDataSource(YFinanceSource(), cache_dir)


def run_backtest(
    *,
    symbols: list[str],
    start: date,
    end: date,
    factor_names: list[str],
    factor_params: dict[str, int],
    factor_weights: dict[str, float] | None = None,
    long_quantile: float = 0.3,
    short_quantile: float = 0.3,
    allow_short: bool = True,
    rebalance_freq: str = "W",
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.0,
    benchmark_symbol: str = "SPY",
    source: DataSource | None = None,
) -> BacktestResult:
    """Full pipeline: fetch -> clean -> score -> size -> cost -> analyze."""
    src = source or default_source()

    market = DataPipeline(src, CleaningConfig()).run(
        FetchRequest.of(symbols, start, end)
    )
    factor = build_factor(factor_names, factor_params, factor_weights)
    scores = factor.compute(market)

    signals = SignalGenerator(
        SignalConfig(
            long_quantile=long_quantile,
            short_quantile=short_quantile,
            allow_short=allow_short,
        )
    ).generate(scores)

    returns = market.simple_returns
    execution = PortfolioEngine(
        ExecutionConfig(
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            rebalance_freq=rebalance_freq,
            initial_capital=initial_capital,
        )
    ).run(signals, returns)

    # Trim the factor warm-up: before it, the book is structurally flat and a
    # long stretch of zero returns drags CAGR and vol toward zero for reasons
    # that have nothing to do with the strategy.
    live = execution.net_returns.loc[(signals != 0).any(axis=1)]
    first_live = live.index[0] if len(live) else execution.net_returns.index[0]
    idx = execution.net_returns.loc[first_live:].index

    benchmark = BenchmarkLoader(src, benchmark_symbol).load(idx)
    equal_weight = BenchmarkLoader.equal_weight(returns).reindex(idx).fillna(0.0)

    analyzer = PerformanceAnalyzer(risk_free_rate=risk_free_rate)
    net = execution.net_returns.loc[idx]

    return BacktestResult(
        market=market,
        factor=factor,
        scores=scores,
        signals=signals,
        execution=execution,
        net_returns=net,
        benchmark=benchmark,
        equal_weight=equal_weight,
        metrics=analyzer.analyze(net),
        gross_metrics=analyzer.analyze(execution.gross_returns.loc[idx]),
        benchmark_metrics=analyzer.analyze(benchmark),
        comparison=analyzer.compare(net, benchmark),
        drawdown=analyzer.drawdown_info(net),
        component_scores={
            # Rebuilt through `build_factor` so each component carries the SAME
            # params as the traded factor. Constructing from the registry with
            # default args instead would make the factor-correlation panel
            # describe factors that are not the ones in the book.
            name: build_factor([name], factor_params).compute(market)
            for name in factor_names
        },
    )

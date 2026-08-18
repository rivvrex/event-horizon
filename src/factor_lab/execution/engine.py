"""Portfolio construction, weight drift, transaction costs and PnL accounting.

TIMING CONVENTION (must match Module 2's lookahead contract)
------------------------------------------------------------
At the start of bar `t` the book is set to `weights[t]`; the bar's return
`returns[t]` then accrues against it. Module 2 already guarantees `signal[t]`
was knowable at the close of `t-1`, so:

    gross_return[t] = sum_i weights[t, i] * returns[t, i]

is correct with NO further shift. Shifting again here would double-lag the
strategy and understate performance.

THE DRIFT PROBLEM
-----------------
Between rebalances the book is untouched, but the *weights* still move: a
position that rallies becomes a larger share of NAV. Charging turnover against
the stale target instead of the drifted book invents trades that never happened
and overstates costs -- badly, at monthly frequency in a trending market.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from factor_lab.execution.schedule import normalize_freq, rebalance_mask
from factor_lab.types import ReturnFrame, SignalFrame, WeightFrame

logger = logging.getLogger(__name__)

BPS: float = 1e-4
_TRADE_TOL: float = 1e-12  # below this a "trade" is float noise, not an order


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Cost model and rebalancing policy. Frozen -> a run is reproducible."""

    cost_bps: float = 10.0        # commission + fees, per unit notional traded
    slippage_bps: float = 5.0     # adverse fill vs. the decision price
    rebalance_freq: str = "D"     # 'D' | 'W' | 'M' (long forms also accepted)
    initial_capital: float = 100_000.0
    gross_leverage: float = 1.0   # target sum(|w|); 1.0 = fully invested, no leverage

    def __post_init__(self) -> None:
        if self.cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("cost_bps and slippage_bps must be non-negative")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.gross_leverage <= 0:
            raise ValueError("gross_leverage must be positive")
        normalize_freq(self.rebalance_freq)  # fail fast on a bad alias

    @property
    def total_cost_rate(self) -> float:
        """Round-trip-agnostic cost per unit of notional traded."""
        return (self.cost_bps + self.slippage_bps) * BPS


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Everything Module 4 needs to compute analytics, and nothing more."""

    gross_returns: pd.Series      # simple returns, before costs
    net_returns: pd.Series        # after commission + slippage
    turnover: pd.Series           # sum|w - w_drift| per bar; 0 off-rebalance
    portfolio_value: pd.Series    # initial_capital * cumprod(1 + net)
    weights: WeightFrame          # actual held weights per bar
    drifted_weights: WeightFrame  # book at start of bar, pre-trade
    costs: pd.Series              # return drag per bar
    trade_log: pd.DataFrame       # one row per executed order
    rebalance_dates: pd.DatetimeIndex
    config: ExecutionConfig = field(repr=False)

    @property
    def total_cost_paid(self) -> float:
        """Cash cost in currency units, integrated over the path."""
        nav_start = self.portfolio_value.shift(1).fillna(
            self.config.initial_capital
        )
        return float((self.costs * nav_start).sum())

    def __repr__(self) -> str:
        pv = self.portfolio_value
        if pv.empty:
            return "ExecutionResult(empty)"
        return (
            f"ExecutionResult({len(pv)} bars, "
            f"final={pv.iloc[-1]:,.0f}, "
            f"trades={len(self.trade_log)}, "
            f"avg_turnover={self.turnover.mean():.3f})"
        )


class PortfolioEngine:
    """Signals -> weights -> drifted book -> costed PnL. Fully vectorized."""

    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()

    # ------------------------------------------------------------------ public

    def run(self, signals: SignalFrame, returns: ReturnFrame) -> ExecutionResult:
        """Backtest `signals` against simple (NOT log) returns.

        Simple returns are required here because portfolio aggregation is a
        weighted ARITHMETIC mean across assets: a portfolio's return is
        sum(w_i * r_i), which holds for simple returns and does not for logs.
        Module 1 stores logs for time-additivity; `MarketData.simple_returns`
        converts. Mixing the two is the single easiest way to get a subtly wrong
        equity curve, so the parameter name is explicit.
        """
        signals, returns = self._align(signals, returns)
        if signals.empty:
            raise ValueError("no overlapping bars between signals and returns")

        rebal = rebalance_mask(signals.index, self.config.rebalance_freq)
        target = self._target_weights(signals)
        weights = self._held_weights(target, returns, rebal)
        drifted = self._drifted_weights(weights, returns)

        # Off-rebalance bars are untouched by construction (weights == drifted),
        # but we mask explicitly so float noise can never bleed into costs.
        turnover = (weights - drifted).abs().sum(axis=1).where(rebal, 0.0)
        costs = turnover * self.config.total_cost_rate

        gross = (weights * returns).sum(axis=1)
        net = gross - costs
        portfolio_value = self.config.initial_capital * (1.0 + net).cumprod()

        result = ExecutionResult(
            gross_returns=gross.rename("gross_return"),
            net_returns=net.rename("net_return"),
            turnover=turnover.rename("turnover"),
            portfolio_value=portfolio_value.rename("portfolio_value"),
            weights=weights,
            drifted_weights=drifted,
            costs=costs.rename("cost"),
            trade_log=self._build_trade_log(
                weights, drifted, rebal, portfolio_value
            ),
            rebalance_dates=pd.DatetimeIndex(signals.index[rebal.to_numpy()]),
            config=self.config,
        )
        logger.info("execution complete: %r", result)
        return result

    # ------------------------------------------------------------------ stages

    @staticmethod
    def _align(
        signals: SignalFrame, returns: ReturnFrame
    ) -> tuple[SignalFrame, ReturnFrame]:
        """Intersect on both axes so every downstream op is shape-safe."""
        cols = signals.columns.intersection(returns.columns)
        idx = signals.index.intersection(returns.index)
        return (
            signals.loc[idx, cols].astype("float64"),
            returns.loc[idx, cols].astype("float64").fillna(0.0),
        )

    def _target_weights(self, signals: SignalFrame) -> WeightFrame:
        """Normalize {-1,0,1} to weights summing to `gross_leverage` in absolute value.

        Equal-weighting within each side is deliberate. Scaling by signal
        strength would smuggle the factor's cross-sectional dispersion into
        position sizing, which conflates "is this name in the book" with "how
        confident am I" -- and makes cost attribution unreadable.
        """
        gross = signals.abs().sum(axis=1).replace(0.0, np.nan)
        weights = signals.div(gross, axis=0) * self.config.gross_leverage
        return weights.fillna(0.0)  # flat bars (no signal) -> all cash

    @staticmethod
    def _held_weights(
        target: WeightFrame, returns: ReturnFrame, rebal: pd.Series
    ) -> WeightFrame:
        """Actual weights held on each bar, closed-form (no Python loop).

        Within a rebalance period the book is frozen, so weights evolve purely
        by compounding. Let period p start at bar s with target w and residual
        cash c = 1 - sum(w). For any bar t in the period:

            A_i[t] = prod_{u=s}^{t-1} (1 + r_i[u])        (asset growth so far)
            w_i[t] = w_i * A_i[t] / (c + sum_j w_j * A_j[t])

        The denominator is NAV growth: cash plus the grown positions. This
        collapses a bar-by-bar recursion into two grouped cumprods, and is
        provably identical to iterating (see tests). Cash is assumed to earn
        zero -- a simplification worth remembering in a high-rate regime, where
        it understates a low-exposure book.
        """
        period_id = rebal.cumsum()

        # Broadcast each period's target across every bar in that period.
        w_start = target.where(rebal, np.nan).ffill().fillna(0.0)
        cash_start = 1.0 - w_start.sum(axis=1)

        growth = (1.0 + returns).fillna(1.0)
        cumulative = growth.groupby(period_id).cumprod()
        # Growth through t-1: shift within the period, 1.0 on the period's first bar.
        asset_growth = cumulative.groupby(period_id).shift(1).fillna(1.0)

        held = w_start * asset_growth
        nav_multiple = cash_start + held.sum(axis=1)
        return held.div(nav_multiple, axis=0)

    @staticmethod
    def _drifted_weights(weights: WeightFrame, returns: ReturnFrame) -> WeightFrame:
        """The book at the START of bar t, before any trade.

        One-step drift of the prior bar's holdings:

            w_drift[t] = w[t-1] * (1 + r[t-1]) / (1 + r_p[t-1])

        Dividing by portfolio NAV growth is what keeps this a *weight* rather
        than a position value. On bar 0 the book is empty, so drift is zero and
        the full initial entry is correctly charged as turnover.
        """
        prev_w = weights.shift(1).fillna(0.0)
        prev_r = returns.shift(1).fillna(0.0)
        nav_growth = 1.0 + (prev_w * prev_r).sum(axis=1)
        return (prev_w * (1.0 + prev_r)).div(nav_growth, axis=0)

    def _build_trade_log(
        self,
        weights: WeightFrame,
        drifted: WeightFrame,
        rebal: pd.Series,
        portfolio_value: pd.Series,
    ) -> pd.DataFrame:
        """One row per executed order. Vectorized via stack, no per-date loop."""
        delta = (weights - drifted).where(rebal, 0.0)
        # future_stack=True keeps NaNs rather than silently dropping them, so the
        # size filter below is the ONLY thing deciding what counts as a trade.
        stacked = delta.stack(future_stack=True).dropna()
        stacked = stacked[stacked.abs() > _TRADE_TOL]
        if stacked.empty:
            return pd.DataFrame(
                columns=[
                    "date", "symbol", "action", "weight_before", "weight_after",
                    "delta_weight", "notional", "cost", "nav_before",
                ]
            )

        # NAV at the start of the bar is what the order is sized against.
        nav_before = portfolio_value.shift(1).fillna(self.config.initial_capital)
        dates = stacked.index.get_level_values(0)
        symbols = stacked.index.get_level_values(1)

        before = drifted.stack(future_stack=True).reindex(stacked.index)
        after = weights.stack(future_stack=True).reindex(stacked.index)
        navs = nav_before.reindex(dates).to_numpy()
        notional = stacked.abs().to_numpy() * navs

        log = pd.DataFrame(
            {
                "date": dates,
                "symbol": symbols,
                "action": np.where(stacked.to_numpy() > 0, "BUY", "SELL"),
                "weight_before": before.to_numpy(),
                "weight_after": after.to_numpy(),
                "delta_weight": stacked.to_numpy(),
                "notional": notional,
                "cost": notional * self.config.total_cost_rate,
                "nav_before": navs,
            }
        )
        return log.sort_values(["date", "symbol"]).reset_index(drop=True)

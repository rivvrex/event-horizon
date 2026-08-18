"""Cleaning + transformation: dirty vendor frame -> analysis-ready `MarketData`.

Every step is vectorized and order-dependent; see `DataPipeline.run` for the
sequence and the reasoning behind it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from factor_lab.data.sources import DataSource, FetchRequest
from factor_lab.types import Field as MarketField
from factor_lab.types import PriceFrame, ReturnFrame

logger = logging.getLogger(__name__)

FillMethod = Literal["ffill", "drop", "none"]


@dataclass(frozen=True, slots=True)
class CleaningConfig:
    """Knobs for the cleaning stage. Frozen so a run is reproducible from config."""

    fill_method: FillMethod = "ffill"
    max_ffill_days: int = 5
    min_history_frac: float = 0.80   # drop a symbol with <80% coverage of the window
    max_abs_daily_return: float = 0.50  # |log ret| above this = suspected bad tick
    winsorize_bad_ticks: bool = True


@dataclass(frozen=True, slots=True)
class MarketData:
    """The single immutable object every downstream module consumes.

    All frames share the identical (DatetimeIndex, columns) shape, which is what
    lets the factor engine do pure matrix math with zero alignment logic.
    """

    prices: PriceFrame        # adjusted close
    log_returns: ReturnFrame  # ln(p_t / p_{t-1})
    volume: PriceFrame
    adj_factor: PriceFrame    # adj_close / close  (corporate-action multiplier)
    config: CleaningConfig

    @classmethod
    def from_prices(
        cls, prices: PriceFrame, config: CleaningConfig | None = None
    ) -> MarketData:
        """Build from an adjusted-price frame alone.

        For synthetic fixtures and for benchmark series, where there is no raw
        close to derive an adjustment factor from -- the prices are already
        total-return, so the factor is identically 1.
        """
        prices = prices.sort_index()
        return cls(
            prices=prices,
            log_returns=np.log(prices).diff(),
            volume=pd.DataFrame(
                np.nan, index=prices.index, columns=prices.columns
            ),
            adj_factor=pd.DataFrame(
                1.0, index=prices.index, columns=prices.columns
            ),
            config=config or CleaningConfig(),
        )

    @property
    def symbols(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def simple_returns(self) -> ReturnFrame:
        """expm1 is the numerically stable inverse of log1p for small returns."""
        return np.expm1(self.log_returns)

    def __len__(self) -> int:
        return len(self.prices)

    def __repr__(self) -> str:
        idx = self.prices.index
        return (
            f"MarketData({len(self.symbols)} symbols, {len(idx)} bars, "
            f"{idx.min():%Y-%m-%d}..{idx.max():%Y-%m-%d})"
        )


class DataPipeline:
    """Orchestrates source -> clean -> transform."""

    def __init__(self, source: DataSource, config: CleaningConfig | None = None) -> None:
        self.source = source
        self.config = config or CleaningConfig()

    def run(self, request: FetchRequest) -> MarketData:
        long = self.source.fetch(request)

        close = self._pivot(long, MarketField.CLOSE.value)
        adj_close = self._pivot(long, MarketField.ADJ_CLOSE.value)
        volume = self._pivot(long, MarketField.VOLUME.value)

        # Fall back to raw close if the provider gave no adjusted series.
        adj_close = adj_close.where(adj_close.notna(), close)
        adj_factor = self._adjustment_factor(close, adj_close)

        prices = self._drop_sparse_symbols(adj_close)
        prices = self._fill_gaps(prices)

        log_returns = self._log_returns(prices)
        log_returns = self._handle_bad_ticks(log_returns)

        volume = volume.reindex(columns=prices.columns, index=prices.index).fillna(0.0)
        adj_factor = adj_factor.reindex(columns=prices.columns, index=prices.index)

        md = MarketData(prices, log_returns, volume, adj_factor, self.config)
        logger.info("pipeline complete: %r", md)
        return md

    # ---------------------------------------------------------------- steps

    @staticmethod
    def _pivot(long: pd.DataFrame, field: str) -> pd.DataFrame:
        """(date, symbol) x fields -> date x symbol for one field."""
        return long[field].unstack(level="symbol").sort_index()

    @staticmethod
    def _adjustment_factor(close: pd.DataFrame, adj_close: pd.DataFrame) -> pd.DataFrame:
        """Corporate-action multiplier.

        adj_close = close * factor, where factor accumulates split ratios and
        reinvested dividends. We keep it explicitly rather than discarding raw
        close because execution needs the *tradeable* price (what you actually
        paid per share) while signals need the *total-return* price. A factor
        that jumps between consecutive days is a split/dividend event, which is
        exactly what the trade log should annotate.
        """
        return (adj_close / close.replace(0.0, np.nan)).astype("float64")

    def _drop_sparse_symbols(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Remove tickers that barely traded over the window.

        A symbol IPO'ing mid-window or delisting early poisons cross-sectional
        ranking: its factor score is computed on a different information set
        than its peers. Cheaper and more honest to drop it than to impute.
        """
        coverage = prices.notna().mean()
        keep = coverage[coverage >= self.config.min_history_frac].index
        dropped = sorted(set(prices.columns) - set(keep))
        if dropped:
            logger.warning("dropping sparse symbols %s (coverage < %.0f%%)",
                           dropped, self.config.min_history_frac * 100)
        return prices[keep]

    def _fill_gaps(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill holidays/halts, bounded by `max_ffill_days`.

        Forward fill is the only *causal* imputation: it uses information that
        existed at time t. Interpolation or bfill would leak the future into a
        signal and silently inflate backtest Sharpe. The `limit` cap prevents a
        long halt from manufacturing a flat, zero-volatility stretch that the
        volatility factor would read as a great risk-adjusted asset.
        """
        if self.config.fill_method == "none":
            return prices
        if self.config.fill_method == "drop":
            return prices.dropna(how="any")

        filled = prices.ffill(limit=self.config.max_ffill_days)
        # Leading NaNs (pre-IPO) are never filled: no price existed to carry.
        return filled.dropna(how="all")

    @staticmethod
    def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """r_t = ln(P_t) - ln(P_{t-1}).

        Log returns over simple returns because they are time-additive
        (sum of daily logs == cumulative log return), which makes multi-period
        aggregation and the volatility/Sharpe math a matrix sum instead of a
        compounding loop. We convert back with expm1 only at reporting time.
        `np.log(...).diff()` avoids the division that `pct_change` performs.
        """
        return np.log(prices).diff()

    def _handle_bad_ticks(self, log_returns: pd.DataFrame) -> pd.DataFrame:
        """Clip implausible single-day moves.

        A |log return| > 50% in one session on a liquid name is far more often a
        vendor error (unadjusted split, decimal shift) than a real move. We clip
        rather than drop so the index stays rectangular; clipping caps the damage
        a single bad print does to a rolling mean or z-score.
        """
        cap = self.config.max_abs_daily_return
        offenders = int((log_returns.abs() > cap).sum().sum())
        if offenders and self.config.winsorize_bad_ticks:
            logger.warning("winsorizing %d returns beyond +/-%.0f%%",
                           offenders, cap * 100)
            return log_returns.clip(-cap, cap)
        return log_returns

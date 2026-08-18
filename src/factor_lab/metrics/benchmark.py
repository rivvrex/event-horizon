"""Benchmark series loading, aligned to a strategy's own trading calendar."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from factor_lab.data import DataPipeline, DataSource, FetchRequest

logger = logging.getLogger(__name__)

SP500_PROXY: str = "SPY"


class BenchmarkLoader:
    """Fetches a benchmark and aligns it to the strategy index.

    SPY is used as the S&P 500 proxy rather than ^GSPC because the index is
    PRICE-only: it excludes dividends. Comparing a total-return strategy
    against a price-only index flatters the strategy by roughly 1.5-2% a year,
    which is larger than most factor alphas. SPY's adjusted close is
    total-return, so the comparison is apples to apples.
    """

    def __init__(self, source: DataSource, symbol: str = SP500_PROXY) -> None:
        self.source = source
        self.symbol = symbol

    def load(
        self, index: pd.DatetimeIndex, *, pad_days: int = 7
    ) -> pd.Series:
        """Return benchmark simple returns reindexed onto `index`.

        `pad_days` pulls a little extra history so the first bar has a genuine
        prior close to difference against, rather than a NaN that would silently
        drop the strategy's first day from every comparison.
        """
        if len(index) == 0:
            raise ValueError("cannot load a benchmark for an empty index")

        request = FetchRequest.of(
            self.symbol,
            (index[0] - pd.Timedelta(days=pad_days)).date(),
            index[-1].date(),
        )
        md = DataPipeline(self.source).run(request)
        prices = md.prices[self.symbol]

        # Reindex onto the strategy calendar, carrying the last known price over
        # any date the strategy trades but the benchmark does not.
        aligned = prices.reindex(prices.index.union(index)).ffill().reindex(index)
        return aligned.pct_change().fillna(0.0).rename(self.symbol)

    @staticmethod
    def from_prices(prices: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        """Build a benchmark from an existing price series (offline/testing)."""
        aligned = prices.reindex(prices.index.union(index)).ffill().reindex(index)
        return aligned.pct_change().fillna(0.0)

    @staticmethod
    def equal_weight(returns: pd.DataFrame) -> pd.Series:
        """Equal-weighted universe return: the honest 'did factor timing help?' test.

        Beating SPY may just mean the universe outperformed. Beating an
        equal-weight basket of the SAME names isolates the factor's selection
        skill from the universe's own drift.
        """
        return returns.mean(axis=1).rename("equal_weight").replace(np.nan, 0.0)

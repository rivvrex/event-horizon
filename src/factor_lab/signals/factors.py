"""Concrete factor implementations. All rolling-window math, no Python loops.

Every `_raw_scores` here returns a frame whose value at index `t` is derived
only from information through the close of `t`; `Factor.compute` then applies
the 1-bar tradeability lag. See `base.py` for the full contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_lab.data import MarketData
from factor_lab.signals.base import Factor, Normalizer

TRADING_DAYS: int = 252


class Momentum(Factor):
    """Cross-sectional price momentum, classic 12-1 formulation.

    score_t = sum(log_returns[t - skip - lookback + 1 .. t - skip])

    Because log returns are time-additive, cumulative momentum is a plain
    rolling SUM -- no compounding loop, no price division. That is the whole
    reason Module 1 stores logs.

    The `skip` gap (default 21 bars ~ 1 month) omits the most recent month. Short
    horizon returns exhibit *reversal* driven by microstructure effects such as
    bid-ask bounce and liquidity provision, which partially cancels the 12-month
    continuation effect. Jegadeesh-Titman style 12-1 momentum excludes it so the
    two opposing signals are not blended into mush.
    """

    def __init__(
        self,
        lookback: int = 231,
        skip: int = 21,
        *,
        normalizer: Normalizer | None = None,
        lag: int = 1,
    ) -> None:
        super().__init__(name=f"Momentum({lookback},{skip})",
                         normalizer=normalizer, lag=lag)
        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}")
        if skip < 0:
            raise ValueError(f"skip must be >= 0, got {skip}")
        self.lookback = lookback
        self.skip = skip

    @property
    def warmup(self) -> int:
        return self.lookback + self.skip + self.lag

    def _raw_scores(self, md: MarketData) -> pd.DataFrame:
        cumulative = md.log_returns.rolling(
            self.lookback, min_periods=self.lookback
        ).sum()
        # shift(skip) is a MODELLING gap (skip the reversal month), not the
        # lookahead lag. It still only looks backwards, so the contract holds.
        return cumulative.shift(self.skip)


class MeanReversion(Factor):
    """Short-horizon reversal: distance below a rolling mean, in sigmas.

    score_t = -(P_t - MA_w(P)_t) / SD_w(P)_t

    The negation is the entire thesis: a price stretched *below* its own recent
    mean scores HIGH (buy the dip), stretched above scores LOW. Scaling by the
    trailing standard deviation makes the score comparable across a $15 and a
    $600 stock, which a raw price gap would not be.

    Note this is a time-series z-score used as a raw score, then standardized
    cross-sectionally by the normalizer. The time-series window is strictly
    trailing, so no future information enters.
    """

    def __init__(
        self,
        window: int = 21,
        *,
        normalizer: Normalizer | None = None,
        lag: int = 1,
    ) -> None:
        super().__init__(name=f"MeanReversion({window})",
                         normalizer=normalizer, lag=lag)
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.window = window

    @property
    def warmup(self) -> int:
        return self.window + self.lag

    def _raw_scores(self, md: MarketData) -> pd.DataFrame:
        roll = md.prices.rolling(self.window, min_periods=self.window)
        mean = roll.mean()
        std = roll.std().replace(0.0, np.nan)  # flat window -> undefined, not inf
        return -(md.prices - mean) / std


class Volatility(Factor):
    """Low-volatility anomaly: realized vol, negated so calm names score HIGH.

    score_t = -sqrt(252) * SD_w(log_returns)_t

    Annualization by sqrt(252) assumes i.i.d. daily returns. That assumption is
    imperfect (returns cluster in volatility), but the scaling is monotonic, so
    for a purely cross-sectional RANKING it changes nothing -- it is applied so
    the raw units stay human-readable in the dashboard.

    Empirically, low-beta/low-vol stocks have delivered higher risk-adjusted
    returns than CAPM predicts, commonly attributed to leverage constraints
    pushing return-seeking investors into high-beta names instead.
    """

    def __init__(
        self,
        window: int = 63,
        *,
        normalizer: Normalizer | None = None,
        lag: int = 1,
    ) -> None:
        super().__init__(name=f"Volatility({window})",
                         normalizer=normalizer, lag=lag)
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.window = window

    @property
    def warmup(self) -> int:
        return self.window + self.lag

    def _raw_scores(self, md: MarketData) -> pd.DataFrame:
        realized = md.log_returns.rolling(
            self.window, min_periods=self.window
        ).std() * np.sqrt(TRADING_DAYS)
        return -realized


class Value(Factor):
    """Long-horizon reversal, a price-only proxy for value.

    score_t = -sum(log_returns[t - lookback + 1 .. t])

    True value needs fundamentals (B/P, E/P) that neither yfinance nor Alpha
    Vantage supply cleanly as a point-in-time series -- and using *current*
    fundamentals against *historical* prices is a textbook lookahead trap. This
    proxy uses the well-documented 3-5 year reversal effect instead: sustained
    long-horizon losers tend to be cheap. Named honestly as a proxy, not as B/P.
    """

    def __init__(
        self,
        lookback: int = 756,
        *,
        normalizer: Normalizer | None = None,
        lag: int = 1,
    ) -> None:
        super().__init__(name=f"Value({lookback})", normalizer=normalizer, lag=lag)
        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}")
        self.lookback = lookback

    @property
    def warmup(self) -> int:
        return self.lookback + self.lag

    def _raw_scores(self, md: MarketData) -> pd.DataFrame:
        return -md.log_returns.rolling(
            self.lookback, min_periods=self.lookback
        ).sum()

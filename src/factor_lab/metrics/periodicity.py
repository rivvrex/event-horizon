"""Annualization scaling.

THE TRAP THIS MODULE EXISTS TO PREVENT
--------------------------------------
The annualization factor is a property of how often the return series is
OBSERVED, not of how often the portfolio is rebalanced. Module 3 emits one row
per trading bar whatever the rebalance cadence, so a monthly-rebalanced
strategy still yields ~252 daily observations per year.

Scaling that daily series by sqrt(12) because "we rebalance monthly" divides
volatility by sqrt(21) and multiplies Sharpe by ~4.6. To legitimately quote a
monthly Sharpe you must first RESAMPLE the returns to monthly (see
`resample_returns`), which is what makes sqrt(12) the correct factor.

`Periodicity.infer` reads the index spacing, so the default is always coherent
with the data actually handed in.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class Periodicity(Enum):
    """Observation frequency of a return series -> periods per year."""

    DAILY = 252
    WEEKLY = 52
    MONTHLY = 12
    QUARTERLY = 4
    ANNUAL = 1

    @property
    def periods_per_year(self) -> int:
        return int(self.value)

    @property
    def annualization(self) -> float:
        """sqrt(periods_per_year): the scalar for standard-deviation quantities.

        Volatility scales with the SQUARE ROOT of time under the i.i.d.
        assumption (variance is additive, so sigma_annual = sigma_period *
        sqrt(n)). Mean returns scale linearly with n. Getting these two mixed up
        is the other classic annualization error.
        """
        return float(np.sqrt(self.value))

    @property
    def pandas_alias(self) -> str:
        """Resampling alias for this frequency."""
        return {
            Periodicity.DAILY: "B",
            Periodicity.WEEKLY: "W-FRI",
            Periodicity.MONTHLY: "BME",
            Periodicity.QUARTERLY: "BQE",
            Periodicity.ANNUAL: "BYE",
        }[self]

    @classmethod
    def from_alias(cls, alias: str) -> Periodicity:
        """Accept the same 'D'/'W'/'M' shorthand Module 3 uses."""
        key = str(alias).strip().upper()
        table = {
            "D": cls.DAILY, "DAILY": cls.DAILY, "B": cls.DAILY,
            "W": cls.WEEKLY, "WEEKLY": cls.WEEKLY,
            "M": cls.MONTHLY, "MONTHLY": cls.MONTHLY,
            "Q": cls.QUARTERLY, "QUARTERLY": cls.QUARTERLY,
            "A": cls.ANNUAL, "Y": cls.ANNUAL, "ANNUAL": cls.ANNUAL,
        }
        if key not in table:
            raise ValueError(
                f"unsupported periodicity {alias!r}; expected one of "
                f"{sorted(set(table))}"
            )
        return table[key]

    @classmethod
    def infer(cls, index: pd.DatetimeIndex) -> Periodicity:
        """Detect frequency from the median gap between observations.

        Median rather than mean: a single long data gap (suspended trading, a
        missing vendor month) would drag the mean into the wrong bucket.
        """
        if len(index) < 3:
            return cls.DAILY  # too short to infer; daily is the engine default
        gaps = np.diff(index.to_numpy()).astype("timedelta64[h]").astype(float) / 24.0
        median_days = float(np.median(gaps))

        if median_days <= 4.0:
            return cls.DAILY
        if median_days <= 10.0:
            return cls.WEEKLY
        if median_days <= 45.0:
            return cls.MONTHLY
        if median_days <= 135.0:
            return cls.QUARTERLY
        return cls.ANNUAL


def resample_returns(returns: pd.Series, target: Periodicity) -> pd.Series:
    """Compound a return series up to a lower frequency.

    Uses `prod(1 + r) - 1` per bucket, NOT a sum: returns compound
    geometrically, and summing them overstates gains and understates losses.
    Only downsampling is meaningful -- you cannot manufacture daily
    observations from monthly ones.
    """
    source = Periodicity.infer(pd.DatetimeIndex(returns.index))
    if target.periods_per_year > source.periods_per_year:
        raise ValueError(
            f"cannot upsample {source.name} returns to {target.name}: "
            "higher-frequency observations do not exist"
        )
    if target is source:
        return returns
    compounded = (1.0 + returns).resample(target.pandas_alias).prod() - 1.0
    return compounded.dropna().rename(returns.name)

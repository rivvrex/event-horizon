"""Continuous factor scores -> discrete {-1, 0, +1} signals.

Scope boundary: this module emits a signal for EVERY bar. It does not know about
rebalancing frequency -- Module 3 samples this frame on its own schedule and
turns signals into sized positions. Keeping the two apart means you can change
rebalance frequency without recomputing a single factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factor_lab.data import MarketData
from factor_lab.signals.base import Factor
from factor_lab.types import SignalFrame


@dataclass(frozen=True, slots=True)
class SignalConfig:
    """Selection rules for turning a cross-section of scores into positions."""

    long_quantile: float = 0.3
    short_quantile: float = 0.3
    allow_short: bool = True
    min_universe: int = 2  # below this, sit flat rather than bet on 1 name

    def __post_init__(self) -> None:
        for label, q in (
            ("long_quantile", self.long_quantile),
            ("short_quantile", self.short_quantile),
        ):
            if not 0.0 < q <= 1.0:
                raise ValueError(f"{label} must be in (0, 1], got {q}")
        if self.min_universe < 1:
            raise ValueError("min_universe must be >= 1")


class SignalGenerator:
    """Rank-based cross-sectional selector.

    Quantiles are converted to integer NAME COUNTS rather than applied as raw
    percentile thresholds. With a 5-20 ticker dashboard universe, a naive
    `rank(pct=True) <= 0.3` breaks down: for 3 names the minimum percentile rank
    is 0.333, so the short book would silently be empty forever. Counting names
    degrades gracefully -- you always get at least one name per active side.
    """

    def __init__(self, config: SignalConfig | None = None) -> None:
        self.config = config or SignalConfig()

    def generate(self, scores: pd.DataFrame) -> SignalFrame:
        """Vectorized: two ranks and two comparisons, no iteration over dates."""
        cfg = self.config
        n_valid = scores.notna().sum(axis=1)

        # Each side is capped at half the cross-section when shorting, which
        # guarantees the long and short books can never overlap.
        cap = n_valid // 2 if cfg.allow_short else n_valid
        active = n_valid >= cfg.min_universe

        n_long = self._side_count(n_valid, cfg.long_quantile, cap, active)
        n_short = (
            self._side_count(n_valid, cfg.short_quantile, cap, active)
            if cfg.allow_short
            else pd.Series(0, index=scores.index, dtype="int64")
        )

        # method="first" makes tie-breaking deterministic (column order), so a
        # backtest is reproducible bar-for-bar.
        desc = scores.rank(axis=1, ascending=False, method="first")
        asc = scores.rank(axis=1, ascending=True, method="first")

        # NaN scores rank NaN; NaN.le(...) is False, so they fall through to 0.
        long_mask = desc.le(n_long, axis=0)
        short_mask = asc.le(n_short, axis=0)
        return (long_mask.astype("int8") - short_mask.astype("int8")).astype("int8")

    def from_factor(self, factor: Factor, md: MarketData) -> SignalFrame:
        """Convenience: compute (lag already applied by Factor) then discretize."""
        return self.generate(factor.compute(md))

    @staticmethod
    def _side_count(
        n_valid: pd.Series, quantile: float, cap: pd.Series, active: pd.Series
    ) -> pd.Series:
        """Names to hold on one side: >=1 when active, never more than `cap`."""
        raw = np.floor(n_valid * quantile).clip(lower=1)
        return np.minimum(raw, cap).where(active, 0).astype("int64")

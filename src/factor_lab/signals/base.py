"""Factor abstraction and cross-sectional normalizers.

THE LOOKAHEAD CONTRACT (read this before touching anything downstream)
---------------------------------------------------------------------
`Factor.compute()` returns a frame where the value at index `t` is tradeable
*during* bar `t` -- i.e. it was fully determined by information available at or
before the close of bar `t-1`.

That guarantee is produced by exactly ONE `.shift(lag)` inside the base class
template method. Subclasses implement `_raw_scores`, which is allowed (and
expected) to use information through the close of bar `t`; the base class
applies the delay. Consequences:

  * A subclass must NEVER call `.shift()` for lookahead reasons. Shifting inside
    a factor for *modelling* reasons (e.g. momentum's skip-month gap) is fine
    and is separate from the lag.
  * Module 3 (execution) must NEVER shift again. `signal[t] * return[t]` is the
    correct PnL expression, because signal[t] was knowable before bar t began.
  * `CompositeFactor` blends children via `_normalized_scores` (un-shifted) so
    the lag is applied once at the top, not once per child.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from factor_lab.data import MarketData
from factor_lab.types import SignalFrame


@runtime_checkable
class Normalizer(Protocol):
    """Maps raw factor scores to a comparable cross-sectional scale.

    Normalization is always CROSS-SECTIONAL (axis=1, across symbols within a
    date), never time-series. Standardizing along the time axis would use future
    observations to scale the present -- a subtle and very common lookahead leak.
    """

    def __call__(self, scores: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class ZScore:
    """(x - cross-sectional mean) / cross-sectional std, then clipped.

    Clipping at +/-`clip` sigma stops one extreme name from dominating a
    composite blend. Applied after standardizing so the threshold is in units of
    cross-sectional dispersion rather than raw factor units.
    """

    clip: float = 3.0

    def __call__(self, scores: pd.DataFrame) -> pd.DataFrame:
        mean = scores.mean(axis=1)
        # A zero cross-section std (all names identical) would yield +/-inf.
        std = scores.std(axis=1).replace(0.0, np.nan)
        z = scores.sub(mean, axis=0).div(std, axis=0)
        return z.clip(-self.clip, self.clip)


@dataclass(frozen=True, slots=True)
class RankNormalizer:
    """Percentile rank in [-0.5, 0.5]. Outlier-proof; discards magnitude."""

    def __call__(self, scores: pd.DataFrame) -> pd.DataFrame:
        return scores.rank(axis=1, pct=True) - 0.5


@dataclass(frozen=True, slots=True)
class Identity:
    """Pass-through, for when raw factor units are already comparable."""

    def __call__(self, scores: pd.DataFrame) -> pd.DataFrame:
        return scores


class Factor(ABC):
    """Base class for every factor. Subclasses implement `_raw_scores` only."""

    def __init__(
        self,
        *,
        name: str | None = None,
        normalizer: Normalizer | None = None,
        lag: int = 1,
    ) -> None:
        if lag < 1:
            raise ValueError(
                f"lag must be >= 1 to avoid lookahead bias, got {lag}. "
                "lag=0 would trade on a signal derived from the same bar's close."
            )
        self.name = name or type(self).__name__
        self.normalizer: Normalizer = normalizer or ZScore()
        self.lag = lag

    @abstractmethod
    def _raw_scores(self, md: MarketData) -> pd.DataFrame:
        """Raw factor values. Index `t` may use information through close of `t`.

        Must be fully vectorized: rolling/ewm/matrix ops only, no `.apply` over
        rows and no Python-level iteration over dates or symbols.
        """

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Bars of history consumed before the first finite score.

        Used by the UI/metrics layer to trim the burn-in period rather than
        silently reporting a stretch where the strategy was structurally flat.
        """

    def _normalized_scores(self, md: MarketData) -> pd.DataFrame:
        """Raw -> masked -> normalized. NO lag applied (composites rely on this)."""
        raw = self._raw_scores(md)
        # Mask names with no tradeable price: a score on a non-existent quote is
        # not actionable, and leaving it in would skew the cross-sectional stats.
        raw = raw.where(md.prices.notna())
        return self.normalizer(raw)

    def compute(self, md: MarketData) -> SignalFrame:
        """Public entry point. The single point where the lookahead lag applies."""
        return self._normalized_scores(md).shift(self.lag)

    def __repr__(self) -> str:
        return f"{self.name}(lag={self.lag}, warmup={self.warmup})"


class CompositeFactor(Factor):
    """Weighted blend of sub-factors.

    Children are combined at the *normalized, un-shifted* stage so the lag is
    applied exactly once by this class's `compute`. Blending post-normalization
    is what makes the weights meaningful -- raw momentum (a log return, ~0.2)
    and raw volatility (~0.3 annualized) are not on comparable scales.
    """

    def __init__(
        self,
        factors: dict[Factor, float] | list[Factor],
        *,
        name: str = "Composite",
        normalizer: Normalizer | None = None,
        lag: int = 1,
    ) -> None:
        super().__init__(name=name, normalizer=normalizer, lag=lag)
        if isinstance(factors, list):
            factors = dict.fromkeys(factors, 1.0)
        if not factors:
            raise ValueError("CompositeFactor requires at least one sub-factor")
        total = sum(abs(w) for w in factors.values())
        if total == 0:
            raise ValueError("sub-factor weights sum to zero")
        # Normalize weights so a composite's scale is independent of how many
        # sub-factors it holds.
        self.factors = {f: w / total for f, w in factors.items()}

    @property
    def warmup(self) -> int:
        return max(f.warmup for f in self.factors)

    def _raw_scores(self, md: MarketData) -> pd.DataFrame:
        blended: pd.DataFrame | None = None
        for factor, weight in self.factors.items():
            contribution = factor._normalized_scores(md) * weight
            blended = contribution if blended is None else blended.add(contribution)
        assert blended is not None  # guaranteed non-empty by __init__
        return blended

"""Shared type aliases and enums used across the whole engine."""

from __future__ import annotations

from enum import StrEnum
from typing import TypeAlias

import pandas as pd

# A wide, date-indexed frame: index = DatetimeIndex (tz-naive, normalized),
# columns = ticker symbols. This is the ONE data shape the engine speaks.
PriceFrame: TypeAlias = pd.DataFrame
ReturnFrame: TypeAlias = pd.DataFrame
SignalFrame: TypeAlias = pd.DataFrame
WeightFrame: TypeAlias = pd.DataFrame


class Field(StrEnum):
    """Canonical OHLCV field names (lower snake case, provider-agnostic)."""

    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"          # raw close, NOT adjusted
    ADJ_CLOSE = "adj_close"  # split + dividend adjusted
    VOLUME = "volume"


class Rebalance(StrEnum):
    DAILY = "D"
    WEEKLY = "W-FRI"
    MONTHLY = "BME"  # business month end

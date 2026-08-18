"""Rebalance calendars.

A rebalance date is the FIRST trading bar of each new period. Convention chosen
deliberately: signals are already lagged by Module 2, so the first bar of a new
month is the earliest bar on which a month-end observation is actionable.
Rebalancing on the *last* bar of the period would require knowing it is the last
bar, which at the time is unknowable -- a calendar-shaped lookahead.
"""

from __future__ import annotations

import pandas as pd

# User-facing shorthand -> pandas period alias used to bucket the index.
_FREQ_ALIASES: dict[str, str] = {
    "D": "D",
    "DAILY": "D",
    "W": "W",
    "WEEKLY": "W",
    "M": "M",
    "MONTHLY": "M",
    "Q": "Q",
    "QUARTERLY": "Q",
}


def normalize_freq(freq: str) -> str:
    """Map 'D'/'W'/'M' (and long forms) onto a pandas period alias."""
    key = str(freq).strip().upper()
    if key not in _FREQ_ALIASES:
        raise ValueError(
            f"unsupported rebalance_freq {freq!r}; "
            f"expected one of {sorted(set(_FREQ_ALIASES))}"
        )
    return _FREQ_ALIASES[key]


def rebalance_mask(index: pd.DatetimeIndex, freq: str) -> pd.Series:
    """Boolean Series: True on bars where the portfolio is re-set to target.

    The first bar is always a rebalance -- that is the initial entry, and it
    must be charged transaction costs like any other trade.
    """
    alias = normalize_freq(freq)
    if len(index) == 0:
        return pd.Series(dtype="bool", index=index)

    if alias == "D":
        mask = pd.Series(True, index=index)
    else:
        periods = index.to_period(alias)
        # First bar of each new period: the period label differs from the prior bar.
        changed = periods[1:] != periods[:-1]
        mask = pd.Series([True, *changed], index=index)

    mask.iloc[0] = True
    return mask.rename("rebalance")

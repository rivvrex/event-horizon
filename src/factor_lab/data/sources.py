"""Provider-agnostic data source abstraction.

Design rule: a `DataSource` is ONLY responsible for network I/O + reshaping a
vendor payload into the canonical long frame. It does no cleaning, no returns
math, no caching. That separation is what lets us swap yfinance -> Alpha Vantage
-> an internal tick store without touching a single line downstream.

Canonical output contract (`fetch`):
    pd.DataFrame with a MultiIndex (date, symbol) and columns == Field values.
    date is tz-naive, normalized to midnight. Sorted. No duplicate index rows.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

import pandas as pd

from factor_lab.types import Field as MarketField

logger = logging.getLogger(__name__)

_CANONICAL_COLUMNS: Final[list[str]] = [f.value for f in MarketField]


class DataSourceError(RuntimeError):
    """Raised when a provider returns nothing usable for the whole request."""


@dataclass(frozen=True, slots=True)
class FetchRequest:
    """Immutable description of what to pull. Hashable -> usable as a cache key."""

    symbols: tuple[str, ...]
    start: date
    end: date

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("symbols must not be empty")
        if self.start >= self.end:
            raise ValueError(f"start {self.start} must be < end {self.end}")

    @classmethod
    def of(
        cls, symbols: Sequence[str] | str, start: str | date, end: str | date
    ) -> FetchRequest:
        """Ergonomic constructor: accepts strings, dedupes and upper-cases symbols."""
        if isinstance(symbols, str):
            symbols = [symbols]
        # dict.fromkeys preserves order while deduping
        clean = tuple(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
        return cls(
            symbols=clean,
            start=pd.Timestamp(start).date(),
            end=pd.Timestamp(end).date(),
        )

    @property
    def cache_key(self) -> str:
        return f"{'-'.join(self.symbols)}_{self.start:%Y%m%d}_{self.end:%Y%m%d}"


class DataSource(ABC):
    """Base class every provider adapter implements."""

    name: str = "abstract"

    @abstractmethod
    def _download(self, request: FetchRequest) -> pd.DataFrame:
        """Vendor call. Must return the canonical MultiIndex frame (may be dirty)."""

    def fetch(self, request: FetchRequest) -> pd.DataFrame:
        """Template method: download, then enforce the output contract."""
        raw = self._download(request)
        if raw.empty:
            raise DataSourceError(
                f"{self.name} returned no rows for {request.symbols} "
                f"[{request.start} .. {request.end}]"
            )
        return self._conform(raw)

    @staticmethod
    def _conform(df: pd.DataFrame) -> pd.DataFrame:
        """Enforce column set, dtypes, index type/order and uniqueness."""
        missing = set(_CANONICAL_COLUMNS) - set(df.columns)
        for col in missing:  # tolerate providers without adj_close / volume
            df[col] = pd.NA
        df = df[_CANONICAL_COLUMNS].astype("float64")

        dates = pd.to_datetime(df.index.get_level_values(0)).tz_localize(None).normalize()
        symbols = df.index.get_level_values(1).astype(str).str.upper()
        df.index = pd.MultiIndex.from_arrays([dates, symbols], names=["date", "symbol"])

        # Keep the last observation for any duplicated (date, symbol) pair: vendors
        # occasionally emit a partial bar then a corrected one.
        df = df[~df.index.duplicated(keep="last")]
        return df.sort_index()


class YFinanceSource(DataSource):
    """Yahoo Finance adapter.

    yfinance quirks handled here:
      * `group_by="ticker"` gives a (symbol, field) column MultiIndex for
        multi-symbol pulls but a flat one for a single symbol -> we normalize.
      * `auto_adjust=False` is REQUIRED: we want raw `close` AND `adj_close`
        so the pipeline can compute a corporate-action adjustment factor
        (adj_close / close) instead of blindly trusting the vendor.
      * `end` is exclusive in Yahoo's API; we add a day so the user's `end` is
        inclusive, matching every other date range in this codebase.
    """

    name = "yfinance"

    def __init__(self, *, timeout: int = 30, retries: int = 2) -> None:
        self.timeout = timeout
        self.retries = retries

    def _download(self, request: FetchRequest) -> pd.DataFrame:
        import yfinance as yf  # local import: keeps module import cheap/testable

        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                raw = yf.download(
                    list(request.symbols),
                    start=request.start,
                    end=request.end + pd.Timedelta(days=1),
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                    timeout=self.timeout,
                )
                break
            except Exception as exc:  # noqa: BLE001 - retry any transport failure
                last_err = exc
                logger.warning("yfinance attempt %d/%d failed: %s",
                               attempt + 1, self.retries + 1, exc)
        else:
            raise DataSourceError(f"yfinance download failed: {last_err}") from last_err

        if raw is None or raw.empty:
            return pd.DataFrame()
        return self._to_long(raw, request.symbols)

    @staticmethod
    def _to_long(raw: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
        """Wide (date x [symbol, field]) -> long (date, symbol) x field."""
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_product([[symbols[0]], raw.columns])

        long = raw.stack(level=0, future_stack=True)
        long.index.names = ["date", "symbol"]
        # "Adj Close" -> adj_close
        long.columns = [str(c).lower().replace(" ", "_") for c in long.columns]
        # Drop symbols Yahoo silently returned as all-NaN (delisted / bad ticker).
        return long.dropna(how="all")


class AlphaVantageSource(DataSource):
    """Alpha Vantage TIME_SERIES_DAILY_ADJUSTED adapter.

    Free tier is 25 req/day and one symbol per call, so we loop and tolerate
    per-symbol failure rather than aborting the whole universe.
    """

    name = "alpha_vantage"
    _URL: Final[str] = "https://www.alphavantage.co/query"
    _FIELD_MAP: Final[dict[str, str]] = {
        "1. open": MarketField.OPEN.value,
        "2. high": MarketField.HIGH.value,
        "3. low": MarketField.LOW.value,
        "4. close": MarketField.CLOSE.value,
        "5. adjusted close": MarketField.ADJ_CLOSE.value,
        "6. volume": MarketField.VOLUME.value,
    }

    def __init__(self, api_key: str, *, timeout: int = 30) -> None:
        if not api_key:
            raise ValueError("Alpha Vantage requires an API key")
        self.api_key = api_key
        self.timeout = timeout

    def _download(self, request: FetchRequest) -> pd.DataFrame:
        import requests  # local import

        frames: list[pd.DataFrame] = []
        for symbol in request.symbols:
            resp = requests.get(
                self._URL,
                params={
                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol": symbol,
                    "outputsize": "full",
                    "apikey": self.api_key,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            series = payload.get("Time Series (Daily)")
            if not series:
                logger.warning("alpha_vantage: no data for %s (%s)", symbol,
                               payload.get("Note") or payload.get("Error Message"))
                continue

            df = pd.DataFrame.from_dict(series, orient="index").rename(
                columns=self._FIELD_MAP
            )
            df.index = pd.to_datetime(df.index)
            df["symbol"] = symbol
            frames.append(df.set_index("symbol", append=True))

        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames)
        mask = (out.index.get_level_values(0).date >= request.start) & (
            out.index.get_level_values(0).date <= request.end
        )
        return out.loc[mask]

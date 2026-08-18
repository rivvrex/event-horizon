"""Parquet-backed disk cache keyed by the immutable FetchRequest."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from factor_lab.data.sources import DataSource, FetchRequest

logger = logging.getLogger(__name__)


class CachedDataSource(DataSource):
    """Decorator that memoizes any `DataSource` to Parquet on disk.

    Wrapping (rather than subclassing each provider) keeps caching orthogonal:
    `CachedDataSource(YFinanceSource())` and `CachedDataSource(AlphaVantageSource(k))`
    both work, and tests can drop the wrapper entirely.
    """

    def __init__(self, inner: DataSource, cache_dir: str | Path = ".cache") -> None:
        self.inner = inner
        self.name = f"cached[{inner.name}]"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, request: FetchRequest) -> Path:
        return self.cache_dir / f"{self.inner.name}_{request.cache_key}.parquet"

    def _download(self, request: FetchRequest) -> pd.DataFrame:
        path = self._path(request)
        if path.exists():
            logger.info("cache hit: %s", path.name)
            return pd.read_parquet(path)

        df = self.inner.fetch(request)
        df.to_parquet(path)
        logger.info("cache write: %s (%d rows)", path.name, len(df))
        return df

    def clear(self) -> int:
        """Delete every cached file. Returns the number removed."""
        files = list(self.cache_dir.glob("*.parquet"))
        for f in files:
            f.unlink()
        return len(files)

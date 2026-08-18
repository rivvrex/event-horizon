"""Performance and risk analytics.

Every statistic is derived from ONE object: the equity curve
`E_t = prod(1 + r_t)`. Computing CAGR from the equity curve while computing
drawdown from a separately-accumulated series is how a dashboard ends up
reporting a max drawdown that never appears on the plotted curve. Deriving
everything from `E_t` makes that class of inconsistency structurally impossible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from factor_lab.metrics.periodicity import Periodicity, resample_returns

_EPS: float = 1e-12
DAYS_PER_YEAR: float = 365.25  # includes the leap-year quarter-day


@dataclass(frozen=True, slots=True)
class DrawdownInfo:
    """Worst peak-to-trough decline and its recovery profile."""

    max_drawdown: float          # negative, e.g. -0.34
    peak_date: pd.Timestamp | None
    trough_date: pd.Timestamp | None
    recovery_date: pd.Timestamp | None
    drawdown_days: int           # peak -> trough
    recovery_days: int | None    # trough -> new high; None if still underwater
    longest_underwater_days: int

    @property
    def is_recovered(self) -> bool:
        return self.recovery_date is not None


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Full analytics bundle for one return stream."""

    total_return: float
    cagr: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    skew: float
    kurtosis: float
    hit_rate: float
    best_period: float
    worst_period: float
    n_periods: int
    years: float
    periodicity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_series(self) -> pd.Series:
        return pd.Series(self.to_dict())


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Strategy measured against a benchmark (S&P 500 by default)."""

    alpha: float                 # annualized Jensen's alpha
    beta: float
    correlation: float
    r_squared: float
    tracking_error: float        # annualized stdev of active return
    information_ratio: float
    up_capture: float
    down_capture: float
    excess_cagr: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PerformanceAnalyzer:
    """Computes every metric from a net return series.

    Stateless and periodicity-aware. Pass `periodicity=None` (default) to infer
    from the index -- the safe choice, since it cannot disagree with the data.
    """

    def __init__(
        self,
        *,
        risk_free_rate: float = 0.0,
        periodicity: Periodicity | str | None = None,
        resample_to: Periodicity | str | None = None,
    ) -> None:
        """
        Args:
            risk_free_rate: ANNUAL rate (0.04 = 4%), de-annualized internally.
            periodicity: Override the inferred observation frequency. Only set
                this if you know the index is misleading.
            resample_to: Compound returns to this frequency BEFORE analyzing.
                This is the correct way to quote a monthly Sharpe: resample,
                then sqrt(12) applies legitimately.
        """
        self.risk_free_rate = risk_free_rate
        self._periodicity = self._coerce(periodicity)
        self._resample_to = self._coerce(resample_to)

    @staticmethod
    def _coerce(value: Periodicity | str | None) -> Periodicity | None:
        if value is None or isinstance(value, Periodicity):
            return value
        return Periodicity.from_alias(value)

    # ------------------------------------------------------------------ public

    def analyze(self, returns: pd.Series) -> PerformanceMetrics:
        rets = self._prepare(returns)
        per = self._resolve_periodicity(rets)

        if len(rets) < 2:
            return self._empty(per)

        equity = self.equity_curve(rets)
        dd = self.drawdown_series(equity)

        total_return = float(equity.iloc[-1] - 1.0)
        years = self._years(pd.DatetimeIndex(rets.index), per)
        cagr = self._cagr(equity, years)
        vol = float(rets.std(ddof=1)) * per.annualization
        max_dd = float(dd.min())

        # scipy's moment estimators warn and return garbage on a (near-)constant
        # series; there is no meaningful shape to report when there is no spread.
        has_spread = float(rets.std(ddof=1)) > _EPS

        return PerformanceMetrics(
            total_return=total_return,
            cagr=cagr,
            annual_volatility=vol,
            sharpe_ratio=self.sharpe(rets, per),
            sortino_ratio=self.sortino(rets, per),
            calmar_ratio=cagr / abs(max_dd) if abs(max_dd) > _EPS else float("nan"),
            max_drawdown=max_dd,
            var_95=float(np.percentile(rets, 5)),
            cvar_95=self._cvar(rets),
            skew=float(stats.skew(rets, bias=False)) if has_spread else float("nan"),
            kurtosis=(
                float(stats.kurtosis(rets, bias=False))  # excess
                if has_spread
                else float("nan")
            ),
            hit_rate=float((rets > 0).mean()),
            best_period=float(rets.max()),
            worst_period=float(rets.min()),
            n_periods=len(rets),
            years=years,
            periodicity=per.name,
        )

    @staticmethod
    def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
        """E_t = initial * prod(1 + r_u) for u <= t. The single source of truth."""
        return (initial * (1.0 + returns).cumprod()).rename("equity")

    @staticmethod
    def drawdown_series(equity: pd.Series) -> pd.Series:
        """dd_t = E_t / cummax(E_u) - 1, in [-1, 0]. Vectorized, no loop.

        `cummax` is the running high-water mark. Expressed as E/peak - 1 rather
        than 1 - E/peak so the sign convention is negative-is-bad, matching how
        drawdown is plotted and quoted everywhere else.
        """
        return (equity / equity.cummax() - 1.0).rename("drawdown")

    def sharpe(self, returns: pd.Series, per: Periodicity | None = None) -> float:
        """(mean excess return / stdev) * sqrt(periods_per_year).

        ddof=1 (sample stdev) because a backtest is a sample of a return
        generating process, not its population. The difference is immaterial at
        n=1500 and material at n=24.
        """
        per = per or self._resolve_periodicity(returns)
        excess = returns - self._periodic_rf(per)
        sd = float(excess.std(ddof=1))
        if sd < _EPS:
            return float("nan")
        return float(excess.mean() / sd) * per.annualization

    def sortino(self, returns: pd.Series, per: Periodicity | None = None) -> float:
        """Excess return over DOWNSIDE deviation.

        Downside deviation = sqrt(mean(min(r - MAR, 0)^2)), with the mean taken
        over ALL n periods, not just the losing ones. Dividing by the count of
        negative periods instead is a common bug: it would reward a strategy for
        having few (but catastrophic) losses, and makes the statistic
        non-comparable across strategies with different loss frequencies.

        MAR is the de-annualized risk-free rate (zero by default).
        """
        per = per or self._resolve_periodicity(returns)
        mar = self._periodic_rf(per)
        excess = returns - mar
        downside = np.minimum(excess, 0.0)
        dd = float(np.sqrt(np.mean(np.square(downside))))
        if dd < _EPS:
            # No downside at all: the ratio is mathematically infinite. Report
            # inf rather than a huge finite number so the UI can flag it.
            return float("inf") if excess.mean() > 0 else float("nan")
        return float(excess.mean() / dd) * per.annualization

    def drawdown_info(self, returns: pd.Series) -> DrawdownInfo:
        """Locate the worst decline and measure time underwater."""
        rets = self._prepare(returns)
        if len(rets) < 2:
            return DrawdownInfo(0.0, None, None, None, 0, None, 0)

        equity = self.equity_curve(rets)
        dd = self.drawdown_series(equity)
        trough = dd.idxmin()
        # The peak is the last high-water mark at or before the trough.
        peak = equity.loc[:trough].idxmax()

        after = equity.loc[trough:]
        recovered = after[after >= equity.loc[peak]]
        recovery = recovered.index[0] if len(recovered) else None

        return DrawdownInfo(
            max_drawdown=float(dd.min()),
            peak_date=peak,
            trough_date=trough,
            recovery_date=recovery,
            drawdown_days=int((trough - peak).days),
            recovery_days=int((recovery - trough).days) if recovery else None,
            longest_underwater_days=self._longest_underwater(dd),
        )

    def compare(
        self, returns: pd.Series, benchmark: pd.Series
    ) -> BenchmarkComparison:
        """Regress strategy on benchmark over their COMMON dates.

        Aligning on the intersection is not a detail: comparing a strategy that
        starts after a factor warmup against a benchmark measured from an
        earlier date silently credits or penalizes the strategy for a window it
        never traded.
        """
        strat = self._prepare(returns)
        bench = self._prepare(benchmark)
        idx = strat.index.intersection(bench.index)
        if len(idx) < 3:
            raise ValueError(
                f"need >=3 overlapping observations, got {len(idx)}"
            )
        strat, bench = strat.loc[idx], bench.loc[idx]

        per = self._resolve_periodicity(strat)
        ppy = per.periods_per_year
        rf = self._periodic_rf(per)

        # Jensen's alpha: regress EXCESS strategy return on EXCESS benchmark.
        reg = stats.linregress(bench - rf, strat - rf)
        active = strat - bench
        te = float(active.std(ddof=1)) * per.annualization

        up, down = bench > 0, bench < 0
        years = self._years(pd.DatetimeIndex(idx), per)

        return BenchmarkComparison(
            alpha=float(reg.intercept) * ppy,
            beta=float(reg.slope),
            correlation=float(reg.rvalue),
            r_squared=float(reg.rvalue) ** 2,
            tracking_error=te,
            information_ratio=(
                float(active.mean()) / float(active.std(ddof=1)) * per.annualization
                if float(active.std(ddof=1)) > _EPS
                else float("nan")
            ),
            up_capture=self._capture(strat, bench, up),
            down_capture=self._capture(strat, bench, down),
            excess_cagr=(
                self._cagr(self.equity_curve(strat), years)
                - self._cagr(self.equity_curve(bench), years)
            ),
        )

    def summary_table(self, streams: dict[str, pd.Series]) -> pd.DataFrame:
        """Metrics for several streams side by side (strategy vs benchmark)."""
        return pd.DataFrame(
            {name: self.analyze(r).to_series() for name, r in streams.items()}
        )

    # ----------------------------------------------------------------- private

    def _prepare(self, returns: pd.Series) -> pd.Series:
        """Drop NaNs, sort, and apply `resample_to` if configured.

        Resampling happens HERE rather than in `analyze` so that every public
        entry point (sharpe, sortino, compare, drawdown_info) sees the same
        series. Doing it in one place only would let a caller get a daily
        Sharpe and a monthly drawdown out of the same analyzer.
        """
        s = pd.Series(returns).dropna().astype("float64")
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.DatetimeIndex(s.index)
        s = s.sort_index()
        if self._resample_to is not None:
            s = resample_returns(s, self._resample_to)
        return s

    def _resolve_periodicity(self, returns: pd.Series) -> Periodicity:
        # An explicit resample target IS the resulting periodicity -- trust it
        # over inference, which can misread a short or gappy resampled index.
        if self._resample_to is not None:
            return self._resample_to
        if self._periodicity is not None:
            return self._periodicity
        return Periodicity.infer(pd.DatetimeIndex(returns.index))

    def _periodic_rf(self, per: Periodicity) -> float:
        """De-annualize the risk-free rate geometrically, not by division.

        (1 + r_annual)^(1/n) - 1 rather than r_annual / n: the two differ by
        only a few bps at daily frequency, but the geometric form is the one
        that compounds back to exactly the annual rate.
        """
        if self.risk_free_rate == 0.0:
            return 0.0
        return float((1.0 + self.risk_free_rate) ** (1.0 / per.periods_per_year) - 1.0)

    @staticmethod
    def _years(index: pd.DatetimeIndex, per: Periodicity) -> float:
        """Elapsed time in years from the CALENDAR span, not the observation count.

        `n / periods_per_year` assumes the index holds exactly `ppy`
        observations per year. Holidays, halts and any resampled series all
        violate that, which would make CAGR disagree between a daily and a
        monthly view of the SAME track record. The calendar span is
        frequency-invariant, so the two agree.

        One period is added because the first return covers a period ending at
        `index[0]`, so the track record actually begins one period earlier.
        """
        if len(index) < 2:
            return 0.0
        period_days = DAYS_PER_YEAR / per.periods_per_year
        span = (index[-1] - index[0]).days + period_days
        return float(span / DAYS_PER_YEAR)

    @staticmethod
    def _cagr(equity: pd.Series, years: float) -> float:
        """Geometric annual growth implied by the equity curve's endpoints.

        Returns NaN on a wiped-out curve (final <= 0): a fractional power of a
        negative number is complex, and "-180% annualized" is not meaningful.
        """
        if years <= 0:
            return float("nan")
        final = float(equity.iloc[-1])
        if final <= 0:
            return float("nan")
        return float(final ** (1.0 / years) - 1.0)

    @staticmethod
    def _cvar(returns: pd.Series, level: float = 5.0) -> float:
        """Expected shortfall: mean of the worst `level`% of periods."""
        cutoff = np.percentile(returns, level)
        tail = returns[returns <= cutoff]
        return float(tail.mean()) if len(tail) else float("nan")

    @staticmethod
    def _capture(strat: pd.Series, bench: pd.Series, mask: pd.Series) -> float:
        """Strategy's mean return / benchmark's mean return over masked periods."""
        if not mask.any():
            return float("nan")
        denom = float(bench[mask].mean())
        if abs(denom) < _EPS:
            return float("nan")
        return float(strat[mask].mean()) / denom

    @staticmethod
    def _longest_underwater(dd: pd.Series) -> int:
        """Longest consecutive stretch below a prior high, in calendar days.

        Vectorized: each new high starts a new "episode" via cumsum on the
        at-peak flag, then group and measure. No iteration over bars.
        """
        underwater = dd < -_EPS
        if not underwater.any():
            return 0
        episode = (~underwater).cumsum()
        spans = (
            dd.index.to_series()
            .groupby(episode)
            .agg(lambda s: (s.iloc[-1] - s.iloc[0]).days)
        )
        return int(spans.max())

    @staticmethod
    def _empty(per: Periodicity) -> PerformanceMetrics:
        nan = float("nan")
        return PerformanceMetrics(
            total_return=0.0, cagr=nan, annual_volatility=nan, sharpe_ratio=nan,
            sortino_ratio=nan, calmar_ratio=nan, max_drawdown=0.0, var_95=nan,
            cvar_95=nan, skew=nan, kurtosis=nan, hit_rate=nan, best_period=nan,
            worst_period=nan, n_periods=0, years=0.0, periodicity=per.name,
        )

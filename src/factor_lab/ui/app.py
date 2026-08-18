"""Module 5: Streamlit dashboard.

Thin presentation layer. Computation lives in `backtest.run_backtest`, figures
live in `charts`; this file owns widgets, layout and caching only. That split is
what keeps the engine runnable from a notebook or a cron job with no Streamlit
process anywhere in sight.

    streamlit run src/factor_lab/ui/app.py

Design system
-------------
Deep charcoal page (`#0A0A0C`) over a slightly lifted chart/sidebar surface
(`#0E1117`), separated by 1px hairlines (`#1E222B`) instead of shadows or cards.
Muted emerald marks gains, soft rose marks losses, and `#F3F4F6` carries text.

Those two accents measure ΔE 5.6 apart under deuteranopia — below the ΔE 6 floor
for colour-only encoding — so nothing here relies on hue alone: every delta ships
a direction glyph and a written comparison, and every badge ships an icon.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from html import escape
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from factor_lab.metrics import PerformanceMetrics
from factor_lab.ui import charts
from factor_lab.ui.backtest import FACTOR_BUILDERS, BacktestResult, run_backtest

DEFAULT_TICKERS = "AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, XOM, JNJ, PG, WMT, KO"
FREQ_LABELS: dict[str, str] = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
PARAM_SPECS: dict[str, tuple[str, str, int, int, int]] = {
    # factor -> (param key, label, min, max, default)
    "Momentum": ("mom_lookback", "Lookback (bars)", 21, 504, 231),
    "Mean Reversion": ("mr_window", "Window (bars)", 5, 126, 21),
    "Volatility": ("vol_window", "Window (bars)", 10, 252, 63),
    "Value": ("value_lookback", "Lookback (bars)", 252, 1260, 756),
}

UP, DOWN, FLAT = "▲", "▼", "–"   # arrows carry sign without hue
CHECK, WARN_ICON, INFO_ICON = "✓", "⚠", "○"

# ----------------------------------------------------------------------- theme

_CSS = """
<style>
:root{
  --fl-bg:#0A0A0C; --fl-surface:#0E1117; --fl-raise:#12161F; --fl-line:#1E222B;
  --fl-line-hi:#2B313D; --fl-ink:#F3F4F6; --fl-muted:#9CA3AF; --fl-faint:#6B7280;
  --fl-pos:#10B981; --fl-neg:#F43F5E;
}
.stApp{background:var(--fl-bg);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stMain"] .block-container{
  padding:2.1rem 2.4rem 4rem;max-width:1560px;}
[data-testid="stToolbar"]{color:var(--fl-faint);}

/* crisp flat surfaces: hairlines, never shadows */
.stPlotlyChart,[data-testid="stDataFrame"],[data-testid="stExpander"],
[data-testid="stVerticalBlockBorderWrapper"]{box-shadow:none!important;}
.stPlotlyChart,[data-testid="stDataFrame"]{
  border:1px solid var(--fl-line);border-radius:6px;overflow:hidden;}

h1,h2,h3,h4{color:var(--fl-ink);letter-spacing:-.012em;}
h1{font-size:1.45rem!important;font-weight:600!important;padding:0 0 .1rem!important;}

/* page title lockup: the name carries the weight, the descriptor recedes to
   muted ink on the same baseline, so the header reads as one line not two. */
.fl-title{display:flex;align-items:baseline;flex-wrap:wrap;gap:.6rem;
  margin:0 0 .55rem;}
.fl-title-name{font-size:1.6rem;font-weight:600;letter-spacing:-.02em;
  color:var(--fl-ink);line-height:1.15;}
.fl-title-sep{width:1px;height:.95rem;background:var(--fl-line-hi);}
.fl-title-sub{font-size:.72rem;letter-spacing:.13em;text-transform:uppercase;
  color:var(--fl-faint);}
h2,h3{font-size:.76rem!important;font-weight:600!important;
  letter-spacing:.1em!important;text-transform:uppercase;color:var(--fl-muted)!important;
  margin:1.5rem 0 .35rem!important;padding:0!important;}
[data-testid="stCaptionContainer"] p{
  color:var(--fl-faint)!important;font-size:.72rem!important;line-height:1.55;}

/* ---- compact KPI grid: 1px hairline gaps, no cards, no shadows ---- */
.fl-kpi-grid{display:grid;gap:1px;background:var(--fl-line);
  border:1px solid var(--fl-line);border-radius:6px;overflow:hidden;
  margin:.15rem 0 1.1rem;}
.fl-kpi{background:var(--fl-surface);padding:.55rem .75rem .6rem;}
.fl-kpi-l{font-size:.615rem;letter-spacing:.09em;text-transform:uppercase;
  color:var(--fl-faint);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.fl-kpi-v{font-size:1.12rem;font-weight:550;color:var(--fl-ink);line-height:1.45;
  font-variant-numeric:tabular-nums;letter-spacing:-.015em;}
.fl-kpi-d{font-size:.665rem;font-variant-numeric:tabular-nums;line-height:1.3;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.fl-kpi-d.pos{color:var(--fl-pos);} .fl-kpi-d.neg{color:var(--fl-neg);}
.fl-kpi-d.flat{color:var(--fl-faint);}

/* ---- inline badges replace warning/info blocks ---- */
.fl-badges{display:flex;flex-wrap:wrap;gap:.3rem;margin:.1rem 0 .8rem;}
.fl-badge{display:inline-flex;align-items:center;gap:.32rem;
  font-size:.665rem;line-height:1.5;padding:.14rem .45rem;border-radius:3px;
  border:1px solid var(--fl-line);background:var(--fl-surface);
  color:var(--fl-muted);font-variant-numeric:tabular-nums;}
.fl-badge.pos{color:var(--fl-pos);border-color:rgba(16,185,129,.32);}
.fl-badge.neg{color:var(--fl-neg);border-color:rgba(244,63,94,.32);}
.fl-badge.warn{color:#E5B567;border-color:rgba(229,181,103,.32);}
.fl-badge b{color:var(--fl-ink);font-weight:550;}

/* ---- sidebar: flat, compact, collapsible ---- */
[data-testid="stSidebar"]{background:var(--fl-surface);
  border-right:1px solid var(--fl-line);}
/* Streamlit reserves a 3.75rem header band plus a 2rem logo spacer above the
   sidebar body. With no app logo configured that is ~94px of dead space over
   the wordmark. The band still has to exist -- it holds the collapse control,
   the only way to hide the sidebar -- so it shrinks to the button rather than
   being removed, and the empty logo placeholder collapses to nothing. */
[data-testid="stSidebar"] [data-testid="stSidebarHeader"]{
  height:2.1rem!important;min-height:0!important;margin-bottom:0!important;
  padding-top:.35rem!important;padding-bottom:0!important;align-items:flex-start;}
[data-testid="stSidebar"] [data-testid="stLogoSpacer"]{
  height:0!important;min-height:0!important;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{
  padding:0 .95rem 2rem;}
.fl-brand-wrap{position:relative;display:inline-block;cursor:default;
  padding-bottom:.95rem;-webkit-user-select:none;user-select:none;}
.fl-brand{font-size:.82rem;font-weight:600;letter-spacing:.16em;
  color:var(--fl-ink);}
.fl-brand-sub{font-size:.66rem;color:var(--fl-faint);margin:.05rem 0 0;}
/* The 5s delay lives only on the :hover rule, so it counts while the pointer is
   held on the wordmark. The base rule has no delay, so leaving hides the line at
   once instead of fading it out five seconds after the mouse has gone. Absolute
   positioning inside the wrap's padding means nothing below it ever shifts. */
.fl-egg{position:absolute;left:0;bottom:.1rem;white-space:nowrap;
  font-size:.6rem;letter-spacing:.15em;text-transform:uppercase;
  color:var(--fl-pos);pointer-events:none;
  opacity:0;transform:translateY(-3px);
  transition:opacity 70ms linear,transform 70ms linear;}
.fl-brand-wrap:hover .fl-egg{opacity:1;transform:translateY(0);
  transition:opacity 450ms ease-out 5s,transform 450ms ease-out 5s;}
.fl-sub{font-size:.615rem;letter-spacing:.09em;text-transform:uppercase;
  color:var(--fl-faint);margin:.55rem 0 -.35rem;}
.fl-rule{height:1px;background:var(--fl-line);margin:.9rem 0 .75rem;}

[data-testid="stExpander"]{border:1px solid var(--fl-line)!important;
  border-radius:5px!important;background:transparent!important;margin-bottom:.4rem;}
[data-testid="stExpander"] summary{padding:.4rem .6rem!important;}
[data-testid="stExpander"] summary p{font-size:.665rem!important;font-weight:600;
  letter-spacing:.09em;text-transform:uppercase;color:var(--fl-muted)!important;}
[data-testid="stExpander"] summary:hover p{color:var(--fl-ink)!important;}
[data-testid="stExpander"] [data-testid="stExpanderDetails"]{padding:0 .6rem .5rem;}

[data-testid="stSidebar"] label p{font-size:.685rem!important;
  color:var(--fl-muted)!important;margin-bottom:.1rem!important;}
[data-testid="stSidebar"] input,[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"]>div,
[data-testid="stSidebar"] [data-baseweb="input"]{
  background:var(--fl-bg)!important;border:1px solid var(--fl-line)!important;
  border-radius:4px!important;color:var(--fl-ink)!important;
  font-size:.75rem!important;box-shadow:none!important;min-height:0!important;}
[data-testid="stSidebar"] [data-baseweb="tag"]{
  background:var(--fl-raise)!important;border:1px solid var(--fl-line-hi)!important;
  color:var(--fl-ink)!important;font-size:.68rem!important;border-radius:3px!important;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"]{margin-bottom:.05rem;}

/* flat sliders */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]{
  height:11px!important;width:11px!important;background:var(--fl-pos)!important;
  border:none!important;box-shadow:none!important;}
[data-testid="stSlider"] [data-testid="stThumbValue"]{
  font-size:.64rem!important;color:var(--fl-pos)!important;}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"]{
  font-size:.6rem!important;color:var(--fl-faint)!important;}
[data-testid="stSlider"]{padding-bottom:.15rem;}

/* flat buttons */
.stButton>button,.stDownloadButton>button{
  background:transparent;border:1px solid var(--fl-line);border-radius:4px;
  color:var(--fl-muted);font-size:.7rem;font-weight:500;padding:.28rem .7rem;
  box-shadow:none;transition:none;min-height:0;}
.stButton>button:hover,.stDownloadButton>button:hover{
  background:var(--fl-raise);border-color:var(--fl-line-hi);color:var(--fl-ink);}
.stButton>button:focus,.stDownloadButton>button:focus{
  color:var(--fl-ink);border-color:var(--fl-pos);}

/* flat tabs */
.stTabs [data-baseweb="tab-list"]{gap:1.5rem;border-bottom:1px solid var(--fl-line);}
.stTabs [data-baseweb="tab"]{background:transparent;padding:.35rem 0;
  font-size:.78rem;color:var(--fl-faint);}
.stTabs [data-baseweb="tab"]:hover{color:var(--fl-muted);}
.stTabs [aria-selected="true"]{color:var(--fl-ink)!important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--fl-pos);height:2px;}
.stTabs [data-baseweb="tab-border"]{display:none;}
</style>
"""

# --------------------------------------------------------------------- helpers

Delta = tuple[str, str] | None
Kpi = tuple[str, str, Delta]


def _pct(value: float, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}%}"


def _num(value: float, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if np.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:,.{digits}f}"


def _delta(
    value: float,
    reference: float,
    *,
    digits: int = 2,
    unit: str = "",
    label: str = "vs bench",
    higher_is_better: bool = True,
) -> Delta:
    """Signed gap vs. a reference, as (text, tone).

    The arrow states the DIRECTION of the gap and the tone states whether that
    direction is good, which are not the same thing: more volatility than the
    benchmark points up and reads red. Both channels ship with the written
    comparison, so a reader who cannot separate emerald from rose loses nothing.
    """
    if value is None or reference is None:
        return None
    if pd.isna(value) or pd.isna(reference):
        return None
    if np.isinf(value) or np.isinf(reference):
        return None
    gap = float(value) - float(reference)
    if abs(gap) < 0.5 * 10.0**-digits:
        return f"{FLAT} flat {label}", "flat"
    good = gap > 0 if higher_is_better else gap < 0
    glyph = UP if gap > 0 else DOWN
    return f"{glyph} {gap:+,.{digits}f}{unit} {label}", "pos" if good else "neg"


def _kpi_grid(cells: Sequence[Kpi], *, min_width: int = 152) -> None:
    """Render a compact metric row as one hairline-separated CSS grid.

    Streamlit's own `st.metric` boxes are chunky and independently sized, so a
    row of five never lines up. One grid with a shared track template does.
    """
    reserve = any(cell[2] is not None for cell in cells)
    blocks: list[str] = []
    for label, value, delta in cells:
        if delta is not None:
            text, tone = delta
            foot = f'<div class="fl-kpi-d {tone}">{escape(text)}</div>'
        elif reserve:
            foot = '<div class="fl-kpi-d flat">&nbsp;</div>'
        else:
            foot = ""
        blocks.append(
            f'<div class="fl-kpi"><div class="fl-kpi-l">{escape(label)}</div>'
            f'<div class="fl-kpi-v">{escape(value)}</div>{foot}</div>'
        )
    st.markdown(
        f'<div class="fl-kpi-grid" style="grid-template-columns:'
        f'repeat(auto-fit,minmax({min_width}px,1fr))">{"".join(blocks)}</div>',
        unsafe_allow_html=True,
    )


def _badge(text: str, tone: str = "", icon: str = "") -> str:
    """One inline badge. Always icon + words: never a bare colour."""
    lead = f"<span>{escape(icon)}</span>" if icon else ""
    return f'<span class="fl-badge {tone}">{lead}<span>{escape(text)}</span></span>'


def _badges(*items: str, container: Any = None) -> None:
    target = container if container is not None else st
    target.markdown(
        f'<div class="fl-badges">{"".join(items)}</div>', unsafe_allow_html=True
    )


def _csv(frame: pd.DataFrame | pd.Series) -> bytes:
    # `to_csv` is overloaded to return `str | None` (None when writing to a path),
    # so the no-path branch still widens to Any. str() pins it.
    return str(frame.to_csv(index=True)).encode("utf-8")


def _tag(start: date, end: date) -> str:
    return f"{start:%Y%m%d}_{end:%Y%m%d}"


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Every widget value that changes a result, in hashable form.

    Frozen and flat so it doubles as the cache key: two identical specs must
    reuse one backtest, and any change to any field must invalidate it. Passing
    the widgets straight into a cached function instead makes it far too easy to
    add an input that silently fails to bust the cache.
    """

    symbols: tuple[str, ...]
    start: date
    end: date
    factor_names: tuple[str, ...]
    params: tuple[tuple[str, int], ...]
    weights: tuple[tuple[str, float], ...]
    long_quantile: float
    short_quantile: float
    allow_short: bool
    rebalance_freq: str
    cost_bps: float
    slippage_bps: float
    initial_capital: float
    risk_free_rate: float
    benchmark_symbol: str


@st.cache_data(show_spinner=False, ttl=60 * 60, max_entries=16)
def _run(spec: RunSpec) -> BacktestResult:
    """Cached backtest. `RunSpec` is frozen, so Streamlit can hash it directly."""
    return run_backtest(
        symbols=list(spec.symbols),
        start=spec.start,
        end=spec.end,
        factor_names=list(spec.factor_names),
        factor_params=dict(spec.params),
        factor_weights=dict(spec.weights),
        long_quantile=spec.long_quantile,
        short_quantile=spec.short_quantile,
        allow_short=spec.allow_short,
        rebalance_freq=spec.rebalance_freq,
        cost_bps=spec.cost_bps,
        slippage_bps=spec.slippage_bps,
        initial_capital=spec.initial_capital,
        risk_free_rate=spec.risk_free_rate,
        benchmark_symbol=spec.benchmark_symbol,
    )


def _pooled_factor_panel(component_scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack each factor's (date x symbol) scores into one column.

    Correlating the POOLED panel rather than a cross-sectional average is the
    point: averaging across symbols first collapses each factor to a single
    time series and reports a number that says nothing about whether the two
    factors rank the same names the same way on the same day.
    """
    if not component_scores:
        return pd.DataFrame()
    return pd.DataFrame(
        {name: df.stack(future_stack=True) for name, df in component_scores.items()}
    ).dropna()


def _metrics_table(streams: dict[str, PerformanceMetrics]) -> pd.DataFrame:
    """Side-by-side metric comparison with display formatting applied."""
    rows: dict[str, list[str]] = {}
    fmt: list[tuple[str, str, str]] = [
        ("Total return", "total_return", "pct"),
        ("CAGR", "cagr", "pct"),
        ("Annual volatility", "annual_volatility", "pct"),
        ("Sharpe ratio", "sharpe_ratio", "num"),
        ("Sortino ratio", "sortino_ratio", "num"),
        ("Calmar ratio", "calmar_ratio", "num"),
        ("Max drawdown", "max_drawdown", "pct"),
        ("VaR 95% (1 bar)", "var_95", "pct"),
        ("CVaR 95% (1 bar)", "cvar_95", "pct"),
        ("Skew", "skew", "num"),
        ("Excess kurtosis", "kurtosis", "num"),
        ("Hit rate", "hit_rate", "pct"),
        ("Best bar", "best_period", "pct"),
        ("Worst bar", "worst_period", "pct"),
        ("Observations", "n_periods", "int"),
        ("Years", "years", "num"),
    ]
    for label, attr, kind in fmt:
        values: list[str] = []
        for m in streams.values():
            raw: Any = getattr(m, attr)
            if kind == "pct":
                values.append(_pct(raw))
            elif kind == "int":
                values.append(f"{int(raw):,}")
            else:
                values.append(_num(raw))
        rows[label] = values
    return pd.DataFrame(rows, index=list(streams)).T


# --------------------------------------------------------------------- sidebar


def build_sidebar() -> RunSpec | None:
    """Collect inputs. Returns None until the universe is large enough to rank.

    Four collapsible sections instead of one long column: the two that change
    every run stay open, the two that are set once and forgotten start closed.
    """
    sb = st.sidebar
    # The wordmark hides a five-second-hover easter egg. Absolutely positioned
    # inside the wrapper's own bottom padding, so it costs no layout when hidden.
    sb.markdown(
        '<div class="fl-brand-wrap" title="">'
        '<div class="fl-brand">EVENT HORIZON</div>'
        '<div class="fl-brand-sub">Cross-sectional factor backtester</div>'
        '<div class="fl-egg">made by rivvrex</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    with sb.expander("Universe", expanded=True):
        raw = st.text_area(
            "Tickers", DEFAULT_TICKERS, height=70,
            help="Comma or whitespace separated.",
        )
        symbols = tuple(
            dict.fromkeys(
                t.strip().upper()
                for t in raw.replace(",", " ").split()
                if t.strip()
            )
        )
        today = date.today()
        col_a, col_b = st.columns(2)
        start = col_a.date_input("Start", today - timedelta(days=365 * 6),
                                 max_value=today)
        end = col_b.date_input("End", today, max_value=today)
        benchmark = st.text_input("Benchmark", "SPY").strip().upper() or "SPY"
        _badges(
            _badge(f"{len(symbols)} tickers",
                   "pos" if len(symbols) >= 4 else "warn",
                   CHECK if len(symbols) >= 4 else WARN_ICON),
            _badge(f"vs {benchmark}", "", INFO_ICON),
        )

    with sb.expander("Factors", expanded=True):
        factor_names = tuple(
            st.multiselect(
                "Signals", list(FACTOR_BUILDERS), default=["Momentum"],
                label_visibility="collapsed",
            )
        )
        params: dict[str, int] = {}
        weights: dict[str, float] = {}
        for name in factor_names:
            key, label, lo, hi, default = PARAM_SPECS[name]
            st.markdown(
                f'<div class="fl-sub">{escape(name)}</div>', unsafe_allow_html=True
            )
            params[key] = int(
                st.slider(label, lo, hi, default, step=1, key=f"p_{key}")
            )
            if name == "Momentum":
                params["mom_skip"] = int(
                    st.slider("Skip gap (bars)", 0, 42, 21, key="p_mom_skip")
                )
            if len(factor_names) > 1:
                weights[name] = float(
                    st.slider("Blend weight", 0.0, 3.0, 1.0, 0.25, key=f"w_{name}")
                )

    with sb.expander("Portfolio", expanded=False):
        long_q = st.slider("Long quantile", 0.05, 0.50, 0.30, 0.05)
        allow_short = st.checkbox("Allow shorts", value=True)
        short_q = (
            st.slider("Short quantile", 0.05, 0.50, 0.30, 0.05)
            if allow_short
            else 0.0
        )
        freq = FREQ_LABELS[
            st.radio("Rebalance", list(FREQ_LABELS), index=1, horizontal=True)
        ]

    with sb.expander("Costs & capital", expanded=False):
        cost_bps = st.number_input("Commission (bps)", 0.0, 200.0, 10.0, 1.0)
        slip_bps = st.number_input("Slippage (bps)", 0.0, 200.0, 5.0, 1.0)
        capital = st.number_input("Initial capital", 1_000.0, 1e9, 100_000.0, 10_000.0)
        rf = st.number_input(
            "Risk-free rate (annual)", 0.0, 0.20, 0.0, 0.005, format="%.3f"
        )

    sb.markdown('<div class="fl-rule"></div>', unsafe_allow_html=True)
    if sb.button("Clear data cache", width="stretch"):
        _run.clear()
        _badges(_badge("Cache cleared", "pos", CHECK), container=sb)

    problems: list[str] = []
    if len(symbols) < 4:
        problems.append("Needs 4+ tickers: a cross-sectional rank needs a spread")
    if not factor_names:
        problems.append("Select at least one factor")
    if start >= end:
        problems.append("Start date must precede end date")
    if problems:
        _badges(*(_badge(p, "warn", WARN_ICON) for p in problems), container=sb)
        return None

    return RunSpec(
        symbols=symbols,
        start=start,
        end=end,
        factor_names=factor_names,
        params=tuple(sorted(params.items())),
        weights=tuple(sorted(weights.items())),
        long_quantile=long_q,
        short_quantile=short_q,
        allow_short=allow_short,
        rebalance_freq=freq,
        cost_bps=float(cost_bps),
        slippage_bps=float(slip_bps),
        initial_capital=float(capital),
        risk_free_rate=float(rf),
        benchmark_symbol=benchmark,
    )


# ------------------------------------------------------- tab 1: performance


def render_performance(result: BacktestResult, spec: RunSpec) -> None:
    m, bm, cmp_ = result.metrics, result.benchmark_metrics, result.comparison
    bench = spec.benchmark_symbol

    _kpi_grid(
        [
            ("CAGR", _pct(m.cagr),
             _delta(m.cagr * 100, bm.cagr * 100, unit="pt", label=f"vs {bench}")),
            ("Sharpe", _num(m.sharpe_ratio),
             _delta(m.sharpe_ratio, bm.sharpe_ratio, label=f"vs {bench}")),
            ("Sortino", _num(m.sortino_ratio), None),
            ("Max drawdown", _pct(m.max_drawdown),
             _delta(m.max_drawdown * 100, bm.max_drawdown * 100, unit="pt",
                    label=f"vs {bench}")),
            ("Ann. volatility", _pct(m.annual_volatility),
             _delta(m.annual_volatility * 100, bm.annual_volatility * 100,
                    unit="pt", label=f"vs {bench}", higher_is_better=False)),
        ]
    )
    _kpi_grid(
        [
            ("Total return", _pct(m.total_return), None),
            ("Calmar", _num(m.calmar_ratio), None),
            ("Alpha (ann.)", _pct(cmp_.alpha), None),
            ("Beta", _num(cmp_.beta), None),
            ("Info ratio", _num(cmp_.information_ratio), None),
        ]
    )

    # Gross vs net side by side: the gap IS the cost of the strategy's turnover,
    # and it is the first thing that kills an otherwise good-looking factor.
    drag = m.cagr - result.gross_metrics.cagr
    _badges(
        _badge(f"gross CAGR {_pct(result.gross_metrics.cagr)}"),
        _badge(f"{DOWN} {_pct(drag)} cost drag", "neg" if drag < 0 else ""),
        _badge(f"net CAGR {_pct(m.cagr)}", "pos" if m.cagr > 0 else "neg"),
        _badge(f"{result.execution.total_cost_paid:,.0f} paid in costs"),
        _badge(f"{m.n_periods:,} {m.periodicity.lower()} bars / {m.years:.2f}y"),
    )

    log_scale = st.toggle("Log scale", value=False, key="log_scale")
    equity = result.equity
    curves = {
        "Strategy (net)": equity,
        f"{bench}": (1.0 + result.benchmark).cumprod(),
        "Equal weight": (1.0 + result.equal_weight).cumprod(),
    }
    st.plotly_chart(charts.equity_curve(curves, log_scale=log_scale), width="stretch")

    st.plotly_chart(
        charts.drawdown_chart(
            {
                "Strategy": equity / equity.cummax() - 1.0,
                f"{bench}": _dd(result.benchmark),
                "Equal weight": _dd(result.equal_weight),
            }
        ),
        width="stretch",
    )

    left, right = st.columns([2, 1], gap="small")
    with left:
        window = min(252, max(21, len(result.net_returns) // 4))
        st.plotly_chart(
            charts.rolling_performance(
                result.net_returns, result.benchmark, window=window
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(charts.monthly_heatmap(result.net_returns), width="stretch")


def _dd(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax() - 1.0


# ------------------------------------------------------- tab 2: signals


def render_signals(result: BacktestResult) -> None:
    st.subheader("Factor diagnostics")

    # A factor-score correlation needs two factors to correlate. With one
    # selected the panel does not exist, so the asset matrix takes the full
    # width rather than leaving half the row empty -- an empty column reads as a
    # chart that failed to render, not as a chart that has nothing to say.
    panel = _pooled_factor_panel(result.component_scores)
    if panel.shape[1] >= 2:
        left, right = st.columns(2, gap="small")
        with left:
            st.plotly_chart(
                charts.correlation_heatmap(panel, "Factor score correlation (pooled)"),
                width="stretch",
            )
            st.caption(
                "Pooled across every (date, symbol) pair. Two factors above ~0.7 "
                "are largely the same bet: blending them adds cost, not breadth."
            )
        target: Any = right
        asset_height: int | None = None
    else:
        target = st.container()
        asset_height = 520

    with target:
        st.plotly_chart(
            charts.correlation_heatmap(
                result.market.simple_returns.dropna(how="all"),
                "Asset return correlation",
                height=asset_height,
            ),
            width="stretch",
        )
        st.caption(
            "A universe that is uniformly high-correlation gives a "
            "long/short book little to separate, whatever the factor says."
        )
        if panel.shape[1] < 2:
            st.caption(
                "Add a second factor in the sidebar to also see how the two "
                "signals correlate with each other."
            )

    st.plotly_chart(charts.factor_score_chart(result.scores), width="stretch")

    st.subheader("Positions")
    weights = result.execution.weights
    gross, net = weights.abs().sum(axis=1), weights.sum(axis=1)
    held = float((weights.abs() > 1e-12).sum(axis=1).mean())
    _kpi_grid(
        [
            ("Avg gross exposure", _pct(float(gross.mean())), None),
            ("Avg net exposure", _pct(float(net.mean())), None),
            ("Max net exposure", _pct(float(net.abs().max())), None),
            ("Avg names held", _num(held, 1), None),
        ]
    )
    st.plotly_chart(charts.weight_area(weights), width="stretch")
    st.plotly_chart(charts.exposure_chart(weights), width="stretch")

    with st.expander("Latest target weights"):
        latest = weights.iloc[-1]
        active = latest[latest.abs() > 1e-12].sort_values(ascending=False)
        st.dataframe(
            active.rename("weight").to_frame().style.format("{:.2%}"),
            width="stretch",
        )


# ------------------------------------------------------- tab 3: risk & trades


def render_risk(result: BacktestResult, spec: RunSpec) -> None:
    st.subheader("Risk metrics")

    table = _metrics_table(
        {
            "Strategy (net)": result.metrics,
            "Strategy (gross)": result.gross_metrics,
            spec.benchmark_symbol: result.benchmark_metrics,
        }
    )
    st.dataframe(table, width="stretch")

    st.subheader(f"Versus {spec.benchmark_symbol}")
    cmp_ = result.comparison
    _kpi_grid(
        [
            ("Alpha (ann.)", _pct(cmp_.alpha), None),
            ("Beta", _num(cmp_.beta), None),
            ("R-squared", _num(cmp_.r_squared, 3), None),
            ("Tracking error", _pct(cmp_.tracking_error), None),
            ("Excess CAGR", _pct(cmp_.excess_cagr), None),
            ("Up capture", _num(cmp_.up_capture), None),
            ("Down capture", _num(cmp_.down_capture), None),
            ("Correlation", _num(cmp_.correlation, 3), None),
        ],
        min_width=138,
    )
    if cmp_.r_squared < 0.10:
        st.caption(
            f"R-squared of {cmp_.r_squared:.3f} means the regression explains "
            "almost nothing, so beta and alpha here are weakly identified. That "
            "is expected for a dollar-neutral book and is not a bug."
        )

    st.subheader("Drawdown profile")
    dd = result.drawdown
    _kpi_grid(
        [
            ("Max drawdown", _pct(dd.max_drawdown), None),
            ("Peak to trough", f"{dd.drawdown_days:,} days", None),
            (
                "Recovery",
                f"{dd.recovery_days:,} days"
                if dd.recovery_days is not None
                else "not recovered",
                None,
            ),
            ("Longest underwater", f"{dd.longest_underwater_days:,} days", None),
        ]
    )
    if dd.peak_date is not None and dd.trough_date is not None:
        _badges(
            _badge(f"peak {dd.peak_date:%Y-%m-%d}"),
            _badge(f"trough {dd.trough_date:%Y-%m-%d}", "neg", DOWN),
            _badge(
                f"recovered {dd.recovery_date:%Y-%m-%d}"
                if dd.recovery_date is not None
                else "still underwater at end of sample",
                "pos" if dd.recovery_date is not None else "warn",
                CHECK if dd.recovery_date is not None else WARN_ICON,
            ),
        )

    left, right = st.columns(2, gap="small")
    with left:
        st.plotly_chart(
            charts.return_distribution(result.net_returns, result.metrics.var_95),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            charts.turnover_chart(result.execution.turnover, result.execution.costs),
            width="stretch",
        )

    st.subheader("Trade log")
    log = result.execution.trade_log
    _kpi_grid(
        [
            ("Orders", f"{len(log):,}", None),
            ("Rebalances", f"{len(result.execution.rebalance_dates):,}", None),
            ("Avg turnover / bar",
             _num(float(result.execution.turnover.mean()), 3), None),
            ("Total costs", f"{result.execution.total_cost_paid:,.0f}", None),
        ]
    )

    if log.empty:
        _badges(_badge("No trades executed for this configuration", "warn", WARN_ICON))
    else:
        symbols = ["All", *sorted(log["symbol"].unique())]
        picked = st.selectbox("Filter by symbol", symbols)
        view = log if picked == "All" else log[log["symbol"] == picked]
        st.dataframe(
            view.tail(500).style.format(
                {
                    "weight_before": "{:.4f}", "weight_after": "{:.4f}",
                    "delta_weight": "{:+.4f}", "notional": "{:,.0f}",
                    "cost": "{:,.2f}", "nav_before": "{:,.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(f"Showing the last {min(len(view), 500):,} of {len(view):,} orders.")

    st.subheader("Downloads")
    tag = _tag(spec.start, spec.end)
    downloads: dict[str, tuple[str, pd.DataFrame | pd.Series]] = {
        "Trade log": (f"trade_log_{tag}.csv", log),
        "Daily returns": (
            f"returns_{tag}.csv",
            pd.DataFrame(
                {
                    "gross": result.execution.gross_returns,
                    "net": result.execution.net_returns,
                    "turnover": result.execution.turnover,
                    "cost": result.execution.costs,
                    "portfolio_value": result.execution.portfolio_value,
                }
            ),
        ),
        "Weights": (f"weights_{tag}.csv", result.execution.weights),
        "Metrics": (f"metrics_{tag}.csv", table),
    }
    cols = st.columns(len(downloads), gap="small")
    for col, (label, (filename, frame)) in zip(cols, downloads.items(), strict=True):
        col.download_button(
            label, _csv(frame), file_name=filename, mime="text/csv",
            width="stretch", key=f"dl_{label}",
        )


# ------------------------------------------------------------------- entrypoint


def main() -> None:
    st.set_page_config(
        page_title="Event Horizon", page_icon="◧", layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    spec = build_sidebar()
    st.markdown(
        '<div class="fl-title">'
        '<span class="fl-title-name">Event Horizon</span>'
        '<span class="fl-title-sep"></span>'
        '<span class="fl-title-sub">Factor Backtest Engine</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    if spec is None:
        _badges(_badge("Configure the run in the sidebar to begin", "", INFO_ICON))
        return

    try:
        with st.spinner("Fetching data and running the backtest..."):
            result = _run(spec)
    except Exception as exc:  # surface the real cause instead of a blank page
        _badges(_badge(f"Backtest failed: {exc}", "neg", WARN_ICON))
        st.exception(exc)
        return

    dropped = sorted(set(spec.symbols) - set(result.market.symbols))
    _badges(
        _badge(result.factor.name, "", INFO_ICON),
        _badge(f"{len(result.market.symbols)} symbols"),
        _badge(f"{spec.rebalance_freq} rebalance"),
        _badge(f"{spec.cost_bps + spec.slippage_bps:.0f} bps all-in"),
        _badge("long/short" if spec.allow_short else "long only"),
        _badge(
            f"{result.net_returns.index[0]:%Y-%m-%d} → "
            f"{result.net_returns.index[-1]:%Y-%m-%d}"
        ),
        *(
            [_badge(f"dropped: {', '.join(dropped)}", "warn", WARN_ICON)]
            if dropped
            else []
        ),
    )

    tab1, tab2, tab3 = st.tabs(["Performance", "Signals & Weights", "Risk & Trades"])
    with tab1:
        render_performance(result, spec)
    with tab2:
        render_signals(result)
    with tab3:
        render_risk(result, spec)


if __name__ == "__main__":
    main()

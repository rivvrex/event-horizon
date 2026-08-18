"""Plotly chart builders. Pure functions: data in, Figure out.

Separated from `app.py` so charts can be rendered in a notebook or exported to
static HTML without a Streamlit runtime.

Design system
-------------
A single dark surface (`SURFACE`), hairline chrome (`HAIRLINE`), and three ink
levels. Every axis ships with `showgrid=False`: on a near-black surface a grid
is the loudest thing on the canvas, and a thin axis line plus outside ticks
carries the same information for a fraction of the visual weight.

Colour is assigned by JOB, not by taste:

* **Entity** — one hue per series, fixed for the life of the dashboard, so the
  strategy is the same colour on every chart. Never cycled: a 9th series folds
  into "Other" rather than wrapping around to slot 0.
* **Status** — `POSITIVE` / `NEGATIVE` for gain/loss. Measured separation between
  those two under deuteranopia is ΔE 5.6, below the ΔE 6 floor, so status is
  *never* carried by hue alone here: signs, glyphs and axis position always say
  the same thing the colour does.
* **Polarity** — two hues with a neutral, surface-adjacent midpoint. Never a
  rainbow, never a hue at the midpoint.

The categorical ramp was validated (lightness band, chroma floor, adjacent-pair
CVD separation, normal-vision floor, contrast vs. surface) against `SURFACE`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------- surfaces
SURFACE = "#0E1117"        # chart paper + plot area
HAIRLINE = "#1E222B"       # axis lines, ticks, cell separators
NEUTRAL = "#252A35"        # diverging midpoint: reads as "no signal"
INK = "#F3F4F6"            # primary text
INK_MUTED = "#9CA3AF"      # axis titles, legends
INK_FAINT = "#6B7280"      # tick labels, annotations, "Other"

# ----------------------------------------------------------------------- status
POSITIVE = "#10B981"       # muted emerald — gains
NEGATIVE = "#F43F5E"       # soft rose — losses and drawdowns

# ------------------------------------------------------------------ categorical
# Fixed hue order. Assigned by position, never cycled past the last slot.
PALETTE: list[str] = [
    "#3987e5", "#d95926", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
]
OTHER_COLOR = INK_FAINT    # achromatic on purpose: "Other" is not an identity

# Entity colours. Pinned so a series never changes hue between charts.
STRATEGY_COLOR = POSITIVE
BENCH_COLOR = PALETTE[0]
EW_COLOR = PALETTE[3]
LOSS_COLOR = NEGATIVE

# ------------------------------------------------------------------- polarity
# Poles are dark enough that #F3F4F6 cell labels clear 3:1 on both ends.
CORR_SCALE: list[list[float | str]] = [
    [0.00, "#1F5FAE"], [0.25, "#22405F"], [0.50, NEUTRAL],
    [0.75, "#6B3520"], [1.00, "#9E3D18"],
]
PNL_SCALE: list[list[float | str]] = [
    [0.00, "#A81E38"], [0.25, "#5E2434"], [0.50, NEUTRAL],
    [0.75, "#125240"], [1.00, "#0B7A57"],
]

# ------------------------------------------------------------------- primitives
LINE_W = 1.5               # equity, benchmark, every headline series
HAIR_W = 1.2               # secondary / reference series
FILL_ALPHA = 0.14          # faint underwater + cost fills

_AXIS: dict[str, object] = {
    "showgrid": False,
    "zeroline": False,
    "showline": True,
    "linecolor": HAIRLINE,
    "linewidth": 1,
    "ticks": "outside",
    "ticklen": 4,
    "tickwidth": 1,
    "tickcolor": HAIRLINE,
    "tickfont": {"size": 10, "color": INK_FAINT},
    "title": {"font": {"size": 10, "color": INK_MUTED}},
    "automargin": True,
}

_LAYOUT: dict[str, object] = {
    "template": "plotly_dark",
    "paper_bgcolor": SURFACE,
    "plot_bgcolor": SURFACE,
    "colorway": PALETTE,
    "font": {"size": 11, "color": INK_MUTED},
    "hovermode": "x unified",
    "hoverlabel": {
        "bgcolor": "#12161F",
        "bordercolor": HAIRLINE,
        "font": {"size": 11, "color": INK},
    },
    # automargin expands these as labels need it, so they are a floor, not a box.
    "margin": {"l": 8, "r": 12, "t": 34, "b": 8},
    "legend": {
        "orientation": "h",
        "yanchor": "bottom", "y": 1.0,
        "xanchor": "right", "x": 1.0,
        "font": {"size": 10, "color": INK_MUTED},
        "bgcolor": "rgba(0,0,0,0)",
        "borderwidth": 0,
        "itemsizing": "constant",
    },
}


def _rgba(hex_color: str, alpha: float) -> str:
    """`#RRGGBB` -> `rgba(r,g,b,a)`. Plotly has no opacity-per-fill shorthand."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _apply(fig: go.Figure, title: str, height: int = 400) -> go.Figure:
    """Shared chrome: dark surface, gridless axes, left-aligned quiet title."""
    fig.update_layout(
        height=height,
        title={
            "text": title,
            "x": 0.0, "xanchor": "left",
            "y": 0.97, "yanchor": "top",
            "font": {"size": 12.5, "color": INK},
        }
        if title
        else None,
        **_LAYOUT,
    )
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)
    return fig


def _series_colors(names: Sequence[str]) -> list[str]:
    """Fixed-order assignment; anything past the ramp is folded, not wrapped."""
    return [
        PALETTE[i] if i < len(PALETTE) else OTHER_COLOR for i in range(len(names))
    ]


# ------------------------------------------------------------------------ charts


def equity_curve(
    curves: dict[str, pd.Series], *, log_scale: bool = False
) -> go.Figure:
    """Growth of the initial capital, rebased so every series starts at 100.

    Rebasing matters: an un-rebased chart comparing a $100k portfolio against
    SPY's share price is visually meaningless. Log scale is offered because on a
    linear axis equal PERCENTAGE moves look larger the higher the curve goes,
    which exaggerates late-period volatility.

    Every line is 1.5px and solid. Dash patterns were tried as a secondary
    encoding and removed: over ~1500 daily bars the dash period is shorter than
    the curve's own oscillation, so a dashed line shatters into a dot cloud. The
    entity trio (emerald / blue / amber) was instead validated to clear the CVD
    separation floor on its own, which is what makes hue sufficient here.
    """
    entity = [STRATEGY_COLOR, BENCH_COLOR, EW_COLOR]
    fig = go.Figure()
    for i, (name, series) in enumerate(curves.items()):
        if series.empty:
            continue
        rebased = 100.0 * series / float(series.iloc[0])
        color = entity[i] if i < len(entity) else PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=rebased.index,
                y=rebased.to_numpy(),
                name=name,
                mode="lines",
                line={"color": color, "width": LINE_W, "shape": "linear"},
                hovertemplate=f"<b>{name}</b>: %{{y:.1f}}<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="Growth of 100", type="log" if log_scale else "linear")
    return _apply(fig, "Cumulative performance", 430)


def drawdown_chart(drawdowns: dict[str, pd.Series]) -> go.Figure:
    """Underwater plot: a faint rose wash for the strategy, hairlines for the rest.

    This is the one chart where the strategy wears the status hue rather than its
    entity hue — every value on it is a loss by construction, so rose is the
    subject, not an identity. The fill sits at 14% alpha so overlapping regions
    stay readable instead of stacking into a solid block.
    """
    entity = [NEGATIVE, BENCH_COLOR, EW_COLOR]
    fig = go.Figure()
    for i, (name, dd) in enumerate(drawdowns.items()):
        color = entity[i] if i < len(entity) else PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=(dd * 100.0).to_numpy(),
                name=name,
                mode="lines",
                fill="tozeroy" if i == 0 else None,
                fillcolor=_rgba(color, FILL_ALPHA) if i == 0 else None,
                line={
                    "color": color,
                    "width": LINE_W if i == 0 else HAIR_W,
                    "shape": "spline",
                    "smoothing": 0.4,
                },
                hovertemplate=f"<b>{name}</b>: %{{y:.2f}}%<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="Drawdown", ticksuffix="%")
    return _apply(fig, "Underwater curve", 300)


def rolling_performance(
    returns: pd.Series, benchmark: pd.Series, window: int = 252
) -> go.Figure:
    """Rolling annualized Sharpe and volatility.

    A single headline Sharpe hides regime change entirely: a strategy that ran
    at 1.5 then collapsed to -0.5 reports the same average as one that was
    steadily 0.5. Rolling windows make that visible.

    Two measures, two panels on a shared x — never two y-scales on one panel.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=(
            f"Rolling {window}-bar Sharpe",
            f"Rolling {window}-bar volatility",
        ),
    )
    for name, series, color in (
        ("Strategy", returns, STRATEGY_COLOR),
        ("Benchmark", benchmark, BENCH_COLOR),
    ):
        roll = series.rolling(window)
        sharpe = roll.mean() / roll.std(ddof=1) * np.sqrt(252)
        vol = roll.std(ddof=1) * np.sqrt(252) * 100.0
        style = {"color": color, "width": LINE_W}
        fig.add_trace(
            go.Scatter(x=sharpe.index, y=sharpe.to_numpy(), name=name, line=style),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=vol.index, y=vol.to_numpy(), name=name, showlegend=False,
                       line=style),
            row=2, col=1,
        )
    fig.add_hline(y=0, line_color=HAIRLINE, line_width=1, row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", row=1, col=1)
    fig.update_yaxes(title_text="Vol", ticksuffix="%", row=2, col=1)
    fig = _apply(fig, "", 480)
    for ann in fig.layout.annotations:
        ann.update(font={"size": 11, "color": INK_MUTED}, x=0, xanchor="left")
    return fig


def correlation_heatmap(
    frame: pd.DataFrame, title: str, *, height: int | None = None
) -> go.Figure:
    """Correlation matrix on a DIVERGING scale centred at zero.

    `height` overrides the row-count default. A matrix that goes full-width needs
    to grow vertically too, or the cells stretch into flat ribbons.

    A sequential colormap would render -1 and +1 as merely "dark" and "light",
    hiding the sign — which is the only thing that matters when checking whether
    two factors are redundant or genuinely complementary. The midpoint is a
    surface-adjacent neutral, never a hue, so "no relationship" recedes.

    The diagonal is blanked. Every asset correlates 1.0 with itself, so those
    cells are a tautology painted in the most saturated colour on the plot —
    they pull the eye straight to the one place with no information.
    """
    corr = frame.corr()
    z = corr.to_numpy().astype("float64").copy()
    np.fill_diagonal(z, np.nan)
    labels = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.2f}")(z))
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale=CORR_SCALE,
            zmid=0.0, zmin=-1.0, zmax=1.0,
            xgap=2, ygap=2,  # 2px surface gap instead of cell borders
            text=labels,
            texttemplate="%{text}",
            textfont={"size": 10, "color": INK},
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
            colorbar={
                "title": {"text": "", "font": {"size": 10}},
                "thickness": 8, "len": 0.7, "outlinewidth": 0,
                "tickfont": {"size": 9, "color": INK_FAINT},
            },
        )
    )
    return _apply(fig, title, height or max(320, 60 + 32 * len(corr)))


def weight_area(weights: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Stacked exposure through time, long above zero and short below.

    Names beyond `top_n` (ranked by average absolute weight) collapse into a
    single achromatic "Other" band. Cycling the ramp back to slot 0 for a 9th
    name would silently give two different symbols the same colour.
    """
    if weights.empty:
        return _apply(go.Figure(), "Portfolio weights through time", 400)

    ranked = weights.abs().mean().sort_values(ascending=False)
    keep = list(ranked.index[:top_n])
    rest = [c for c in weights.columns if c not in keep]

    bands: dict[str, pd.Series] = {c: weights[c] for c in keep}
    if rest:
        bands[f"Other ({len(rest)})"] = weights[rest].sum(axis=1)
    colors = [*_series_colors(keep), *(([OTHER_COLOR]) if rest else [])]

    fig = go.Figure()
    for (name, series), color in zip(bands.items(), colors, strict=True):
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=(series * 100.0).to_numpy(),
                name=name, mode="lines", stackgroup="one",
                # A surface-coloured hairline is the 2px gap between stacked
                # segments. An edge in the band's own hue reads as extra data at
                # this density; the surface colour reads as a seam.
                line={"width": 0.7, "color": SURFACE},
                fillcolor=color,
                hovertemplate=f"<b>{name}</b>: %{{y:.2f}}%<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_color=HAIRLINE, line_width=1)
    fig.update_yaxes(title_text="Weight", ticksuffix="%")
    return _apply(fig, "Portfolio weights through time", 400)


def exposure_chart(weights: pd.DataFrame) -> go.Figure:
    """Gross (sum |w|) and net (sum w) exposure.

    Net exposure is the market bet; gross is the capital at risk. A book that
    drifts from dollar-neutral to net-long is quietly taking beta, and that is
    invisible on a weights chart alone.
    """
    gross = weights.abs().sum(axis=1) * 100.0
    net = weights.sum(axis=1) * 100.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=gross.index, y=gross.to_numpy(), name="Gross",
                   line={"color": PALETTE[0], "width": LINE_W},
                   hovertemplate="<b>Gross</b>: %{y:.1f}%<extra></extra>")
    )
    fig.add_trace(
        go.Scatter(x=net.index, y=net.to_numpy(), name="Net",
                   line={"color": PALETTE[1], "width": LINE_W, "dash": "dash"},
                   hovertemplate="<b>Net</b>: %{y:.1f}%<extra></extra>")
    )
    fig.add_hline(y=0, line_color=HAIRLINE, line_width=1)
    fig.update_yaxes(title_text="Exposure", ticksuffix="%")
    return _apply(fig, "Gross vs net exposure", 300)


def turnover_chart(turnover: pd.Series, costs: pd.Series) -> go.Figure:
    """Per-bar turnover above, cumulative cost drag below.

    These were once one panel with a secondary y-axis. Two y-scales on one plot
    let the author place the crossover wherever the story needs it, so the
    measures now get a panel each on a shared x — same comparison, no rescaling
    sleight of hand.

    Only bars with non-zero turnover are drawn. Off-rebalance bars are zero by
    construction, and plotting ~1500 of them into a 900px panel gives each one
    less than a pixel, which renders as grey antialiasing mush rather than data.
    """
    traded = turnover[turnover > 0.0]
    if traded.empty:
        traded = turnover
    # Bar width is in milliseconds on a date axis; without it Plotly derives one
    # from the tightest gap in the (irregular) rebalance schedule.
    if len(traded) > 1:
        gaps = np.diff(traded.index.to_numpy().astype("datetime64[ms]").astype("int64"))
        bar_width: float | None = float(np.median(gaps)) * 0.7
    else:
        bar_width = None

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
        subplot_titles=("Turnover per rebalance", "Cumulative cost drag"),
    )
    fig.add_trace(
        go.Bar(
            x=traded.index, y=traded.to_numpy(), name="Turnover", width=bar_width,
            marker={"color": PALETTE[0], "line": {"width": 0}},
            hovertemplate="Turnover: %{y:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=costs.index, y=(costs.cumsum() * 100.0).to_numpy(),
            name="Cumulative cost", showlegend=False,
            line={"color": NEGATIVE, "width": LINE_W},
            fill="tozeroy", fillcolor=_rgba(NEGATIVE, FILL_ALPHA),
            hovertemplate="Cost: %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.update_yaxes(title_text="Turnover", row=1, col=1)
    fig.update_yaxes(title_text="Cost", ticksuffix="%", row=2, col=1)
    fig = _apply(fig, "", 340)
    fig.update_layout(bargap=0.06, showlegend=False)
    for ann in fig.layout.annotations:
        ann.update(font={"size": 11, "color": INK_MUTED}, x=0, xanchor="left")
    return fig


def return_distribution(returns: pd.Series, var_95: float) -> go.Figure:
    """Return histogram, binned up front so each bar can be coloured by sign.

    Splitting the bars at zero puts the loss tail in the status hue, but the
    zero line and the x-axis say the same thing, so the meaning survives without
    the colour. The 5% VaR is marked because the tail, not the mode, is the part
    that ends a strategy.
    """
    pct = (returns * 100.0).to_numpy()
    counts, edges = np.histogram(pct[np.isfinite(pct)], bins=60)
    centers = (edges[:-1] + edges[1:]) / 2.0
    colors = np.where(centers < 0.0, NEGATIVE, POSITIVE)

    fig = go.Figure(
        go.Bar(
            x=centers, y=counts, name="Bars",
            width=float(edges[1] - edges[0]) * 0.88,  # 2px-equivalent surface gap
            marker={"color": colors, "line": {"width": 0}},
            hovertemplate="%{x:.2f}%: %{y} bars<extra></extra>",
        )
    )
    fig.add_vline(
        x=var_95 * 100.0, line_dash="dash", line_color=NEGATIVE, line_width=1,
        annotation_text=f"VaR 95% {var_95:.2%}", annotation_position="top left",
        annotation_font={"size": 10, "color": INK_MUTED},
    )
    fig.add_vline(x=0.0, line_color=HAIRLINE, line_width=1)
    fig.update_xaxes(title_text="Bar return", ticksuffix="%")
    fig.update_yaxes(title_text="Frequency")
    fig = _apply(fig, "Return distribution", 340)
    fig.update_layout(hovermode="closest", showlegend=False, bargap=0.0)
    return fig


def monthly_heatmap(returns: pd.Series) -> go.Figure:
    """Month-by-year returns grid. Surfaces seasonality and the worst months.

    Every cell carries its own signed number, which is what makes the emerald /
    rose pair legitimate here: the two hues are close under deuteranopia, so the
    printed sign — not the colour — is the load-bearing encoding.
    """
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    grid = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.to_numpy() * 100.0,
        }
    ).pivot(index="year", columns="month", values="ret")
    grid.columns = [
        pd.Timestamp(2000, int(m), 1).strftime("%b") for m in grid.columns
    ]
    limit = float(np.nanmax(np.abs(grid.to_numpy()))) if grid.size else 1.0
    # Heatmap rows render bottom-up, so reverse to read earliest year at the top.
    grid = grid.iloc[::-1]
    z = grid.to_numpy()
    # Blank the months before inception instead of printing "NaN" into them, and
    # sign every number: emerald and rose are a weak separation under CVD, so the
    # glyph is what actually carries win-or-lose.
    labels = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:+.1f}")(z))
    fig = go.Figure(
        go.Heatmap(
            z=z, x=list(grid.columns),
            y=[str(y) for y in grid.index],
            colorscale=PNL_SCALE,
            zmid=0.0, zmin=-limit, zmax=limit,
            xgap=2, ygap=2,
            text=labels, texttemplate="%{text}",
            textfont={"size": 10, "color": INK},
            hovertemplate="%{y} %{x}: %{z:+.2f}%<extra></extra>",
            colorbar={
                "title": {"text": "", "font": {"size": 10}},
                "thickness": 8, "len": 0.7, "outlinewidth": 0,
                "ticksuffix": "%",
                "tickfont": {"size": 9, "color": INK_FAINT},
            },
        )
    )
    return _apply(fig, "Monthly returns", max(280, 70 + 32 * len(grid)))


def factor_score_chart(scores: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Normalized factor scores through time for the most-traded names."""
    cols = list(scores.columns[:top_n])
    fig = go.Figure()
    for col, color in zip(cols, _series_colors(cols), strict=True):
        fig.add_trace(
            go.Scatter(
                x=scores.index, y=scores[col].to_numpy(), name=col,
                line={"color": color, "width": HAIR_W},
                hovertemplate=f"<b>{col}</b>: %{{y:.2f}}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_color=HAIRLINE, line_width=1)
    fig.update_yaxes(title_text="Normalized score (z)")
    return _apply(fig, "Factor scores through time", 340)

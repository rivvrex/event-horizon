# Event Horizon

**A cross-sectional factor backtester with a Streamlit dashboard.**

Event Horizon ranks a universe of equities on one or more quantitative factors,
goes long the top quantile and short the bottom, applies realistic transaction
costs and slippage, and reports the result against the S&P 500 — CAGR, Sharpe,
Sortino, max drawdown, alpha/beta, and a full trade log.

The engine is a typed, modular Python package (`factor_lab`). The dashboard is a
thin UI layer on top of it, so every backtest is equally runnable from a notebook,
a script, or a scheduled job without importing Streamlit.

---

## Table of contents

- [Screenshot walkthrough](#what-you-get)
- [Quickstart](#quickstart)
- [Using the dashboard](#using-the-dashboard)
- [Using the engine from Python](#using-the-engine-from-python)
- [The factors](#the-factors)
- [Project structure](#project-structure)
- [How it works — design decisions](#how-it-works--design-decisions)
- [Data sources & caching](#data-sources--caching)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [What to commit to Git](#what-to-commit-to-git)

---

## What you get

Three tabs, driven by one sidebar:

| Tab | Contents |
|---|---|
| **Performance** | Equity curve (linear + log) vs benchmark and equal-weight, underwater drawdown, rolling Sharpe & volatility, monthly returns heatmap |
| **Signals & Weights** | Factor-score correlation (pooled), asset return correlation, latest factor scores, portfolio weight area, gross/net/long/short exposure |
| **Risk & Trades** | Risk metric table, drawdown profile with recovery dates, return distribution, turnover & cost bars, filterable trade log, four CSV downloads |

Everything is computed vectorized — no per-bar Python loops anywhere in the
signal, execution, or metrics path.

---

## Quickstart

### Requirements

- **Python 3.12 or newer** (developed and verified on **3.14.6**). The package
  declares `requires-python = ">=3.11"` and the runtime works there, but
  `mypy` needs 3.12+ because numpy's bundled type stubs use `type` statements.
- An internet connection for the first run of any ticker/date combination.
  After that, prices are served from a local parquet cache.

### Install

```bash
git clone https://github.com/<your-username>/event-horizon.git
```

```bash
cd event-horizon
```

Create and activate a virtual environment.

**Windows (PowerShell):**

```bash
python -m venv .venv; .venv\Scripts\Activate.ps1
```

**Windows (Git Bash):**

```bash
python -m venv .venv && source .venv/Scripts/activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv && source .venv/bin/activate
```

Then install the package in editable mode with its dev extras:

```bash
pip install -e ".[dev]"
```

The editable install is what puts `factor_lab` on the import path — the app will
not start without it (or an equivalent `PYTHONPATH=src`).

### Run

```bash
streamlit run src/factor_lab/ui/app.py
```

Streamlit opens <http://localhost:8501>. The default universe is 12 large-cap US
names over the last six years, ranked on Momentum, rebalanced weekly. Press
**Run backtest** in the sidebar.

The first run downloads prices from Yahoo Finance and takes ~10–30 seconds
depending on the universe size. Subsequent runs with the same tickers and dates
read from `.cache/` and are effectively instant.

### Verify the install

```bash
pytest -q
```

18 tests should pass in roughly 20 seconds. They use synthetic data and make no
network calls, so this works offline.

---

## Using the dashboard

The sidebar has four collapsible sections. The two you change every run start
open; the two you set once start closed.

### Universe

| Input | Default | Notes |
|---|---|---|
| **Tickers** | `AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, XOM, JNJ, PG, WMT, KO` | Comma or whitespace separated. Duplicates are dropped, case is normalized. |
| **Start / End** | last 6 years → today | |
| **Benchmark** | `SPY` | Any Yahoo symbol. Falls back to `SPY` if blank. |

**A minimum of 4 tickers is enforced.** A cross-sectional rank needs a spread to
rank across; with three names a 30% quantile is a single stock and the "factor"
is noise. The sidebar shows a warning badge and refuses to run below that.

### Factors

Pick one or more of **Momentum**, **Mean Reversion**, **Volatility**, **Value**.
Each selection reveals its own lookback slider. Selecting two or more also
reveals a **Blend weight** slider per factor (0–3, default 1.0) — the factors are
z-scored before blending, so the weights are directly comparable.

### Portfolio

| Input | Range | Default |
|---|---|---|
| **Long quantile** | 0.05 – 0.50 | 0.30 |
| **Allow shorts** | on/off | on |
| **Short quantile** | 0.05 – 0.50 | 0.30 |
| **Rebalance** | Daily / Weekly / Monthly | Weekly |

With shorts off the book is long-only and the short quantile is forced to zero.

### Costs & capital

| Input | Range | Default |
|---|---|---|
| **Commission (bps)** | 0 – 200 | 10 |
| **Slippage (bps)** | 0 – 200 | 5 |
| **Initial capital** | 1,000 – 1e9 | 100,000 |
| **Risk-free rate (annual)** | 0.00 – 0.20 | 0.00 |

Costs are charged per unit of turnover, only on rebalance days. Set both to zero
to see the gross strategy; the Performance tab always reports gross and net side
by side so the cost drag is explicit.

**Clear data cache** at the bottom of the sidebar wipes the in-process Streamlit
cache. It does *not* delete `.cache/` — remove that directory manually to force a
fresh download from Yahoo.

### Downloads

The Risk & Trades tab exports four CSVs, all stamped with the date range:

- `trade_log_*.csv` — one row per executed order: `date, symbol, action,
  weight_before, weight_after, delta_weight, notional, cost, nav_before`
- `returns_*.csv` — daily `gross, net, turnover, cost, portfolio_value`
- `weights_*.csv` — the full daily weight matrix
- `metrics_*.csv` — the strategy/benchmark comparison table

---

## Using the engine from Python

The dashboard is optional. `run_backtest` is the whole pipeline in one call and
returns a frozen `BacktestResult`:

```python
from datetime import date

from factor_lab.ui.backtest import run_backtest

result = run_backtest(
    symbols=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM"],
    start=date(2019, 1, 1),
    end=date(2025, 1, 1),
    factor_names=["Momentum", "Volatility"],
    factor_params={"mom_lookback": 231, "mom_skip": 21, "vol_window": 63},
    factor_weights={"Momentum": 1.0, "Volatility": 0.5},
    long_quantile=0.30,
    short_quantile=0.30,
    allow_short=True,
    rebalance_freq="W",          # "D" | "W" | "M"
    cost_bps=10.0,
    slippage_bps=5.0,
    initial_capital=100_000.0,
    risk_free_rate=0.04,
    benchmark_symbol="SPY",
)

m = result.metrics
print(f"CAGR      {m.cagr:>8.2%}")
print(f"Ann. vol  {m.annual_volatility:>8.2%}")
print(f"Sharpe    {m.sharpe_ratio:>8.2f}")
print(f"Sortino   {m.sortino_ratio:>8.2f}")
print(f"Max DD    {m.max_drawdown:>8.2%}")
print(f"Alpha     {result.comparison.alpha:>8.2%}   beta {result.comparison.beta:.2f}")

result.execution.trade_log.to_csv("trades.csv", index=False)
```

`BacktestResult` carries everything the UI renders:

| Field | Type | What it is |
|---|---|---|
| `market` | `MarketData` | cleaned prices, log returns, simple returns |
| `factor` | `Factor` | the composed factor actually traded |
| `scores` | `DataFrame` | date × symbol factor scores |
| `signals` | `SignalFrame` | date × symbol in {-1, 0, +1} |
| `execution` | `ExecutionResult` | weights, turnover, costs, trade log, NAV |
| `net_returns` | `Series` | net returns over the live window |
| `benchmark`, `equal_weight` | `Series` | comparison streams |
| `metrics`, `gross_metrics`, `benchmark_metrics` | `PerformanceMetrics` | the analytics bundles |
| `comparison` | `BenchmarkComparison` | alpha, beta, IR, up/down capture |
| `drawdown` | `DrawdownInfo` | peak/trough/recovery dates, days underwater |
| `component_scores` | `dict[str, DataFrame]` | each factor's scores, at the params you passed |
| `equity` | `Series` (property) | NAV over the live window |

The lower-level modules are usable on their own:

```python
from factor_lab.data import CleaningConfig, DataPipeline, FetchRequest, YFinanceSource
from factor_lab.signals import Momentum, SignalConfig, SignalGenerator
from factor_lab.execution import ExecutionConfig, PortfolioEngine
from factor_lab.metrics import PerformanceAnalyzer

market = DataPipeline(YFinanceSource(), CleaningConfig()).run(
    FetchRequest.of(["AAPL", "MSFT", "NVDA", "AMZN"], date(2020, 1, 1), date(2024, 1, 1))
)
scores = Momentum(231, 21).compute(market)
signals = SignalGenerator(SignalConfig(long_quantile=0.25)).generate(scores)
execution = PortfolioEngine(ExecutionConfig(rebalance_freq="M")).run(
    signals, market.simple_returns
)
metrics = PerformanceAnalyzer(risk_free_rate=0.04).analyze(execution.net_returns)
```

---

## The factors

All four are price-only and fully vectorized. Higher score = more attractive, so
the sign conventions are chosen to make "rank descending, buy the top" always
correct.

| Factor | Formula | Param | Range | Default |
|---|---|---|---|---|
| **Momentum** | `sum(log_ret[t-skip-L+1 .. t-skip])` | lookback `L` | 21–504 | 231 |
| | | skip gap | 0–42 | 21 |
| **Mean Reversion** | `-(P_t - MA_w) / SD_w` | window `w` | 5–126 | 21 |
| **Volatility** | `-sqrt(252) * SD_w(log_ret)` | window `w` | 10–252 | 63 |
| **Value** | `-sum(log_ret[t-L+1 .. t])` | lookback `L` | 252–1260 | 756 |

**Momentum** is the classic 12-1: eleven months of return skipping the most
recent month. Because log returns are time-additive, cumulative momentum is a
plain rolling *sum* — no compounding loop, no price division.

**Mean Reversion** is negated on purpose: a price stretched *below* its own recent
mean scores high (buy the dip). Dividing by the rolling standard deviation makes
the score a z-distance, comparable across names of different volatility.

**Volatility** is the low-vol anomaly — realized volatility, negated, so calm
names rank highest.

**Value** is a long-horizon reversal used as a *price-only proxy* for value.
This is a deliberate limitation, documented in the code: real value needs
point-in-time fundamentals (B/P, E/P) that neither yfinance nor Alpha Vantage
supply cleanly, and joining *current* fundamentals onto historical prices is a
lookahead bug dressed up as a factor. A three-year reversal captures a similar
"beaten-down names outperform" effect without the survivorship and restatement
problems. Treat it as a proxy, not as value.

**Blending** goes through `CompositeFactor`, which cross-sectionally z-scores each
component before applying weights. Without that step a factor measured in "sum of
log returns" (~0.4) and one measured in sigmas (~2.0) would combine on wildly
different scales and the weights would mean nothing.

---

## Project structure

```
event-horizon/
├── README.md
├── pyproject.toml              # deps, ruff, mypy, pytest config
├── .gitignore
├── .streamlit/
│   └── config.toml             # dark theme — REQUIRED, see note below
├── src/
│   └── factor_lab/
│       ├── __init__.py
│       ├── types.py            # shared type aliases (PriceFrame, SignalFrame, ...)
│       ├── data/               # MODULE 1 — ingestion & cleaning
│       │   ├── sources.py      #   YFinanceSource, AlphaVantageSource, DataSource ABC
│       │   ├── cache.py        #   CachedDataSource -> parquet
│       │   └── pipeline.py     #   DataPipeline, CleaningConfig, MarketData
│       ├── signals/            # MODULE 2 — factors & signal generation
│       │   ├── base.py         #   Factor ABC, Normalizer, ZScore, RankNormalizer
│       │   ├── factors.py      #   Momentum, MeanReversion, Volatility, Value, Composite
│       │   └── generator.py    #   SignalGenerator, SignalConfig
│       ├── execution/          # MODULE 3 — portfolio & costs
│       │   ├── schedule.py     #   rebalance_mask, normalize_freq
│       │   └── engine.py       #   PortfolioEngine, ExecutionConfig, ExecutionResult
│       ├── metrics/            # MODULE 4 — analytics
│       │   ├── periodicity.py  #   Periodicity, resample_returns
│       │   ├── analytics.py    #   PerformanceAnalyzer, PerformanceMetrics, DrawdownInfo
│       │   └── benchmark.py    #   BenchmarkLoader, SP500_PROXY
│       └── ui/                 # MODULE 5 — dashboard
│           ├── backtest.py     #   run_backtest, BacktestResult (no Streamlit import)
│           ├── charts.py       #   10 Plotly builders, pure functions
│           └── app.py          #   Streamlit layout, sidebar, CSS
└── tests/
    ├── test_data.py
    ├── test_signals.py
    ├── test_execution.py
    ├── test_metrics.py
    └── test_ui.py
```

The distributed package is named `factor-lab` in `pyproject.toml` (the import
path is `factor_lab`); **Event Horizon** is the product name shown in the UI.

> **`.streamlit/config.toml` is not optional.** Streamlit's BaseWeb widget
> internals read their colours from that file, not from injected CSS. Delete it
> and the sidebar inputs revert to the default light theme while the charts stay
> dark — the app looks broken. Commit it.

---

## How it works — design decisions

These are the choices that matter for whether the numbers are trustworthy.

### 1. A one-bar shift kills lookahead bias

Factor scores are computed on data through day *t*, then shifted forward one bar
before they can become a position. A score built from day *t*'s close cannot be
traded at day *t*'s close. This single shift is the difference between a Sharpe
of 3.0 and a Sharpe of 0.4, and it is enforced inside `SignalGenerator` rather
than left to the caller.

### 2. Log returns for factors, simple returns for P&L

Log returns are time-additive, which makes cumulative momentum a rolling sum
instead of a compounding loop. But log returns are *not* portfolio-additive — you
cannot weight-average them across positions. So factors consume
`market.log_returns` and the execution engine consumes `market.simple_returns`.
Both are produced by the pipeline; using the wrong one is a quiet 1–2% annual
error.

### 3. Weight drift, and turnover measured against it

Between rebalances, positions drift with prices — a winner becomes a larger share
of the book on its own. The engine tracks the drifted weights `w_drift` and
computes turnover as `|w_target − w_drift|`, not `|w_target − w_previous_target|`.
The naive version charges you for drift you never traded and materially
overstates costs at monthly cadence.

Costs are `(cost_bps + slippage_bps) × turnover`, applied on rebalance days only
and deducted from gross returns to give net.

### 4. Every metric derives from one equity curve

`E_t = Π(1 + r_net,t)`. CAGR, drawdown, Calmar, and total return all read from
that single series. Computing CAGR from a mean return while computing drawdown
from a separately accumulated curve is how a dashboard ends up reporting a max
drawdown that doesn't appear on the chart it just drew. Deriving everything from
`E_t` makes that inconsistency structurally impossible.

### 5. Cadence-aware annualization

Sharpe and volatility scale by `sqrt(periods_per_year)` — `sqrt(252)` daily,
`sqrt(52)` weekly, `sqrt(12)` monthly — inferred from the index rather than
assumed. CAGR uses the **calendar span**, not `n / periods_per_year`, because
holidays and halts mean the observation count lies. The calendar span is
frequency-invariant, so a daily and a monthly view of the same track record
report the same CAGR.

### 6. Sortino divides by *n*, not by the loss count

Downside deviation is `sqrt(mean(min(r − MAR, 0)²))` with the mean taken over all
*n* periods. Dividing by the count of negative periods instead would reward a
strategy for having few but catastrophic losses, and makes the statistic
non-comparable across strategies with different loss frequencies.

### 7. Max drawdown is vectorized

`dd_t = E_t / cummax(E_u) − 1`, in `[-1, 0]`. Sign convention is
negative-is-bad, matching how drawdown is plotted and quoted everywhere else.
Time-underwater is computed by `cumsum` on the at-peak flag to form episodes —
still no loop over bars.

### 8. The factor warm-up is trimmed

A 231-bar momentum lookback means the book is structurally flat for the first
year. Including that stretch drags CAGR and volatility toward zero for reasons
that have nothing to do with the strategy. `run_backtest` trims `net_returns` to
the first date any signal is non-zero, and the benchmark is aligned to that same
window — so the comparison never credits the strategy for a period it didn't
trade.

### 9. Missing data is handled explicitly, not silently

`CleaningConfig` defaults:

| Setting | Default | Why |
|---|---|---|
| `fill_method` | `"ffill"` | a halted stock holds its last price |
| `max_ffill_days` | `5` | past a week, forward-filling invents data |
| `min_history_frac` | `0.80` | symbols with <80% coverage are dropped, not filled |
| `max_abs_daily_return` | `0.50` | flags likely bad ticks |
| `winsorize_bad_ticks` | `True` | clip rather than drop, so the index stays intact |

Prices are adjusted for splits and dividends at the source (`auto_adjust`), so a
split doesn't register as a −50% return.

### 10. Frozen dataclasses throughout

Every config and result object is `@dataclass(frozen=True, slots=True)`. A cached
`BacktestResult` cannot be mutated by the UI, and `RunSpec` — the sidebar's
frozen input bundle — doubles as the `@st.cache_data` key. One object defines both
the run and its cache identity, so they cannot drift apart.

### 11. Charts follow a design system, not taste

`charts.py` holds ten pure functions returning Plotly figures. Rules that are
enforced rather than suggested: no dual-axis charts ever (two measures of
different scale get two stacked subplots); sequential scales use one hue,
diverging scales use two hues plus a neutral grey midpoint; categorical colours
are assigned in fixed order and never cycled — a ninth series folds into an
achromatic "Other"; and every positive/negative cue carries a sign or glyph as
well as a colour, because emerald and rose are not reliably separable under
deuteranopia. The three-entity palette was validated for colour-vision
deficiency rather than eyeballed.

---

## Data sources & caching

### Yahoo Finance (default)

`YFinanceSource` needs no API key and no configuration. It is wrapped in
`CachedDataSource`, which writes one parquet file per request to `.cache/`:

```python
from factor_lab.ui.backtest import default_source
source = default_source(".cache")   # CachedDataSource(YFinanceSource(), ".cache")
```

The cache key covers symbols, dates, and interval, so changing the universe
fetches only what's new. Delete `.cache/` to force a full refresh.

### Alpha Vantage (optional)

`AlphaVantageSource` takes the key as an **explicit constructor argument**. The
codebase reads no environment variables anywhere — if you want to source the key
from the environment, that's your call to make:

```python
import os
from factor_lab.data import AlphaVantageSource, CachedDataSource

source = CachedDataSource(
    AlphaVantageSource(os.environ["ALPHAVANTAGE_KEY"], timeout=30),
    ".cache",
)
```

Pass it through `run_backtest(..., source=source)`. Note the free tier is capped
at 25 requests/day, so a 12-name universe uses half your daily quota in one run —
the parquet cache matters much more here than with Yahoo.

### Bring your own source

Subclass `DataSource` and implement `fetch(request) -> PriceFrame`. Anything that
returns a date-indexed OHLCV frame works — a CSV directory, a database, an
internal API. Pass it to `run_backtest(source=...)` and nothing downstream
changes. This is also how the test suite runs entirely offline.

---

## Development

```bash
pytest -q
```

```bash
ruff check src tests
```

```bash
mypy src
```

All three are green on the committed tree: 18 tests pass, ruff reports no
findings, and mypy runs in `strict` mode across 21 source files with no errors.

Configuration lives in `pyproject.toml` — ruff at line-length 90 with
`E, F, I, N, UP, B, SIM, ANN` enabled, mypy strict with
`ignore_missing_imports`. There is deliberately **no `python_version` pin** under
`[tool.mypy]`: pinning it below 3.12 makes numpy's own stubs fail to parse.

The UI is tested headlessly with `streamlit.testing.v1.AppTest` against a
synthetic `DataSource`, so `test_ui.py` verifies the app renders without
exceptions and without touching the network.

`.streamlit/config.toml` sets `runOnSave = true`, so edits to any file under
`src/` reload the running app automatically.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'factor_lab'`**
The editable install didn't happen or you're in the wrong environment. Re-run
`pip install -e ".[dev]"` with the venv activated. As a one-off workaround,
`PYTHONPATH=src streamlit run src/factor_lab/ui/app.py` also works.

**Sidebar widgets are light-themed while the charts are dark**
`.streamlit/config.toml` is missing. See the note in
[Project structure](#project-structure).

**`Needs 4+ tickers` and the run button won't fire**
By design — a cross-sectional rank needs at least four names to be meaningful.

**Empty charts, or "not enough history"**
Your date range is shorter than the factor lookback. Momentum at its 231-bar
default needs about 14 months of data before it produces a single signal; Value
at 756 bars needs over three years. Either widen the range or shorten the
lookback.

**yfinance returns nothing / rate-limits**
Yahoo throttles bursts. Wait a minute and retry; cached tickers are unaffected.
Also check the symbol — Yahoo wants `BRK-B`, not `BRK.B`.

**Only one correlation heatmap in the Signals tab**
Expected with a single factor selected. A *factor-score* correlation needs two
factors to correlate, so the asset-return matrix takes the full width instead of
leaving half the row empty. Add a second factor to see both.

**mypy fails inside `numpy/__init__.pyi`**
You're on Python 3.11, or something pinned `python_version` below 3.12. numpy's
stubs use `type` statements, which mypy only parses under 3.12+. The runtime is
fine; only type-checking is affected.

---

## What to commit to Git

The repo is not yet initialized. Here's exactly what belongs in it.

### ✅ Commit these

```
README.md
pyproject.toml
.gitignore
.streamlit/config.toml
src/factor_lab/**          (21 .py files — the entire package)
tests/**                   (6 .py files including __init__.py)
```

That's it. Roughly 30 files. Everything needed for a stranger to clone, install,
and run — and nothing else.

### ❌ Never commit these

| Path | Why |
|---|---|
| `.venv/` | hundreds of MB, machine- and OS-specific |
| `.cache/` | downloaded parquet price data; regenerates on demand, and republishing vendor data is a licensing question you don't want |
| `__pycache__/`, `*.pyc` | build artifacts |
| `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/` | tool caches |
| `*.egg-info/` | packaging artifact from the editable install |
| `.env` | secrets — an Alpha Vantage key here must never be pushed |
| `.claude/` | local AI-assistant session state |

The included `.gitignore` covers all of these.

### First push

```bash
git init && git add . && git status
```

Read that `git status` output before committing — it is the last cheap chance to
catch a `.venv` or a stray `.env` that slipped past the ignore rules. Then:

```bash
git commit -m "Event Horizon: modular factor backtesting engine and dashboard"
```

```bash
git branch -M main && git remote add origin https://github.com/<your-username>/event-horizon.git && git push -u origin main
```

### Worth adding before you push

- **`LICENSE`** — without one, nobody can legally reuse the code. MIT is the
  conventional choice for a portfolio project.
- **A screenshot** in the README. Put a PNG at `docs/screenshot.png`, commit it,
  and reference it near the top. A dashboard project is judged on the dashboard,
  and most people won't clone it to look.
- **`requirements.txt`** only if you want a byte-exact reproduction of your
  environment (`pip freeze > requirements.txt`). It's redundant with
  `pyproject.toml` for normal installs — the pinned versions are the point, not
  the dependency list.

---

## Disclaimer

This is a research and educational tool. Backtested results are not indicative of
future performance, the universe you type in is subject to survivorship bias, and
the `Value` factor is a price-only proxy rather than a fundamental measure.
Nothing here is investment advice.

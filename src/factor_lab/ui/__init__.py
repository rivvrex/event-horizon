"""Module 5 public surface.

`app` is deliberately NOT re-exported: importing it pulls in Streamlit, which
should stay optional for anyone driving the engine from a notebook or a job.
Import `factor_lab.ui.app` explicitly if you need the dashboard.
"""

from factor_lab.ui.backtest import (
    FACTOR_BUILDERS,
    BacktestResult,
    build_factor,
    default_source,
    run_backtest,
)

__all__ = [
    "FACTOR_BUILDERS",
    "BacktestResult",
    "build_factor",
    "default_source",
    "run_backtest",
]

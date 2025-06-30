"""Thin wrapper for technical indicators supporting TA-Lib or pandas-ta."""

from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import talib  # type: ignore
    _USE_TALIB = True
except Exception:  # pragma: no cover - fallback
    import pandas_ta as ta  # type: ignore
    _USE_TALIB = False

import pandas as pd


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    if _USE_TALIB:
        return pd.Series(talib.RSI(close, timeperiod=length), index=close.index)
    return ta.rsi(close, length=length)


def bbands(close: pd.Series, length: int = 20):
    if _USE_TALIB:
        upper, middle, lower = talib.BBANDS(close, timeperiod=length)
        return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})
    bands = ta.bbands(close, length=length)
    return bands.rename(columns={
        bands.columns[0]: "lower",
        bands.columns[1]: "middle",
        bands.columns[2]: "upper",
    })[["upper", "middle", "lower"]]


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    if _USE_TALIB:
        return pd.Series(talib.ATR(high, low, close, timeperiod=length), index=close.index)
    return ta.atr(high, low, close, length=length)

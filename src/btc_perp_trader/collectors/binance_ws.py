"""
Binance WebSocket – gera velas de 1m em tempo real (mainnet ou testnet).

Uso:
    from btc_perp_trader.collectors.binance_ws import BinanceWebsocket
    feed = BinanceWebsocket("BTCUSDT", testnet=True)
    async for candle in feed.stream():
        ...
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

import websockets
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    timestamp: int          # ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    atr: Optional[float] = None  # True Range médio (calculado on-the-fly)

    def to_features(self) -> pd.Series:
        """Converte para Série com indicadores básicos (RSI 14 + ATR 14)."""
        df = pd.DataFrame(
            {
                "open": [self.open],
                "high": [self.high],
                "low": [self.low],
                "close": [self.close],
                "volume": [self.volume],
            }
        )
        df["rsi14"] = ta.rsi(df["close"], length=14)
        df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        return df.iloc[-1]


class BinanceWebsocket:
    """Async generator que faz streaming de velas 1m."""

    def __init__(self, symbol: str = "BTCUSDT", *, testnet: bool = False) -> None:
        self.symbol = symbol.lower()
        base = "wss://stream.binance.com:9443" if not testnet else "wss://stream.binancefuture.com"
        self.url = f"{base}/ws/{self.symbol}@kline_1m"
        self._prev_candles = []  # p/ ATR incremental

    async def _parse(self, msg: str) -> Optional[Candle]:
        data = json.loads(msg)
        k = data.get("k")  # kline payload
        if not k or not k["x"]:  # somente quando a vela fecha
            return None

        candle = Candle(
            timestamp=k["t"],
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
        )

        # ATR incremental: manter as últimas 14 TRs
        tr = max(
            candle.high - candle.low,
            abs(candle.high - (self._prev_candles[-1].close if self._prev_candles else candle.close)),
            abs(candle.low - (self._prev_candles[-1].close if self._prev_candles else candle.close)),
        )
        self._prev_candles.append(candle)
        if len(self._prev_candles) > 14:
            self._prev_candles.pop(0)

        if len(self._prev_candles) == 14:
            candle.atr = sum(
                max(
                    c.high - c.low,
                    abs(c.high - (self._prev_candles[i - 1].close if i else c.close)),
                    abs(c.low - (self._prev_candles[i - 1].close if i else c.close)),
                )
                for i, c in enumerate(self._prev_candles)
            ) / 14

        return candle

    async def stream(self) -> AsyncGenerator[Candle, None]:
        async with websockets.connect(self.url, ping_interval=20) as ws:
            logger.info("WebSocket conectado: %s", self.url)
            async for raw in ws:
                candle = await self._parse(raw)
                if candle:
                    yield candle


import asyncio
import json
import importlib

import pandas as pd
import pytest

try:
    import pandas_ta  # noqa: F401
except Exception:
    pytest.skip("pandas_ta not installed", allow_module_level=True)

from btc_perp_trader.collectors.binance_ws import BinanceWebsocket, Candle


class DummyWS:
    def __init__(self, messages):
        self.messages = messages

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.messages:
            raise StopAsyncIteration
        return self.messages.pop(0)


@pytest.mark.asyncio
async def test_stream_parses_candle(monkeypatch):
    msg_open = json.dumps({"k": {"x": False}})
    msg_close = json.dumps(
        {
            "k": {
                "t": 1,
                "o": "1",
                "h": "2",
                "l": "0.5",
                "c": "1.5",
                "v": "10",
                "x": True,
            }
        }
    )
    dummy = DummyWS([msg_open, msg_close])
    monkeypatch.setattr(
        "btc_perp_trader.collectors.binance_ws.websockets.connect", lambda *a, **k: dummy
    )
    feed = BinanceWebsocket("BTCUSDT")
    candles = []
    async for c in feed.stream():
        candles.append(c)
    assert len(candles) == 1
    c = candles[0]
    assert isinstance(c, Candle)
    assert c.open == 1.0
    assert c.close == 1.5


def test_ohlcv_shape_and_imputation():
    ts = pd.date_range("2024-01-01", periods=5, freq="1min")
    df = pd.DataFrame(
        {"timestamp": ts, "symbol": "BTC", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
    )
    df = df.drop(2)  # create gap
    filled = df.set_index("timestamp").asfreq("1min").ffill()
    assert filled.shape == (5, 6)
    assert not filled.isna().any().any()


import requests, pandas as pd, numpy as np, datetime as dt, pathlib, json, gzip
from pandas import Timestamp

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "binance_train.parquet"
CACHE.parent.mkdir(exist_ok=True)

def _fetch_candles(symbol="BTCUSDT", interval="1m", limit=1500) -> pd.DataFrame:
    url = f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{dt.date.today()}.zip"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(gzip.decompress(r.content))
    except Exception:
        # fallback via REST (limit 1500)
        resp = requests.get("https://fapi.binance.com/fapi/v1/klines",
                            params=dict(symbol=symbol, interval=interval, limit=limit),
                            timeout=10).json()
        df = pd.DataFrame(resp, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","trades","taker_buy_base",
            "taker_buy_quote","ignore"
        ])
    df = df.astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

def _engineering(df: pd.DataFrame) -> pd.DataFrame:
    import pandas_ta as ta
    df["rsi14"]   = ta.rsi(df["close"], length=14)
    df["atr14"]   = ta.atr(df["high"], df["low"], df["close"], length=14)
    # Bollinger e outras features são ignoradas por enquanto para manter
    # compatibilidade com o streaming em tempo real (que só calcula RSI/ATR).
    df["label"]   = (df["close"].shift(-3) > df["close"]).astype(int)  # 1=queda→short
    df = df.dropna().reset_index(drop=True)
    # Mantém somente as colunas que o robô vai fornecer na predição
    keep = [
        "open", "high", "low", "close", "volume",
        "rsi14", "atr14", "label",
    ]
    return df[keep]

def load_dataset() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    df = _fetch_candles(limit=1500)
    df = _engineering(df)
    df.to_parquet(CACHE)
    return df

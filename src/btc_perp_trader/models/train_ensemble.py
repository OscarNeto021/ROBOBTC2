import requests, pandas as pd, numpy as np, datetime as dt, pathlib, json, gzip
from pandas import Timestamp

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "binance_train.parquet"
LOCAL_CSV = ROOT / "src" / "data" / "btc_1m_2025-06-30.csv"
CACHE.parent.mkdir(exist_ok=True)

def _fetch_candles(symbol="BTCUSDT", interval="1m", limit=None) -> pd.DataFrame:
    url = f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{dt.date.today()}.zip"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(gzip.decompress(r.content))
    except Exception:
        # fallback via REST (limit 1500)
        params = dict(symbol=symbol, interval=interval)
        if limit:
            params["limit"] = limit
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params=params,
            timeout=10,
        ).json()
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

    df["rsi14"] = ta.rsi(df["close"], length=14)
    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    bb = ta.bbands(df["close"])
    if bb is not None:
        df["bb_lower"] = bb["BBL_20_2.0"]
        df["bb_middle"] = bb["BBM_20_2.0"]
        df["bb_upper"] = bb["BBU_20_2.0"]

    macd = ta.macd(df["close"])
    if macd is not None:
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]

    df["obv"] = ta.obv(df["close"], df["volume"])
    df["volatility"] = df["close"].rolling(window=20).std()
    df["volume_ma20"] = df["volume"].rolling(window=20).mean()

    if "timestamp" in df.columns:
        df["dow"] = df["timestamp"].dt.dayofweek
        df["hour"] = df["timestamp"].dt.hour
        df["month"] = df["timestamp"].dt.month

    df["label"] = (df["close"].shift(-3) > df["close"]).astype(int)
    df = df.dropna().reset_index(drop=True)

    keep = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rsi14",
        "atr14",
        "bb_lower",
        "bb_middle",
        "bb_upper",
        "macd",
        "macd_signal",
        "obv",
        "volatility",
        "volume_ma20",
        "dow",
        "hour",
        "month",
        "label",
    ]
    return df[keep]

def load_dataset() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)

    if LOCAL_CSV.exists():
        df = pd.read_csv(LOCAL_CSV)
        if "ts" in df.columns:
            df.rename(columns={"ts": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df = _fetch_candles()

    df = _engineering(df)
    df.to_parquet(CACHE)
    return df

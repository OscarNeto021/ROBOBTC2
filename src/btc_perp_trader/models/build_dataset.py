import argparse
import pathlib
import pickle
from typing import Optional

import numpy as np
import orjson
import pandas as pd
import pandas_ta as ta


# ─────────────────────────────────────────────────────────────────────
# util - garante existência da coluna ts em DF de candles
# aceita open_time/close_time em ms, s ou string ISO
# --------------------------------------------------------------------
def _ensure_ts(df: pd.DataFrame, interval="1min") -> pd.DataFrame:
    if "ts" in df.columns:
        return df
    if "open_time" in df.columns:
        # Binance CSV: open_time geralmente em milissegundos
        ts = pd.to_datetime(df["open_time"], unit="ms", errors="coerce", utc=True)
    elif "timestamp" in df.columns:
        # fallback genérico
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    else:
        # último recurso: tenta index
        ts = pd.to_datetime(df.index, errors="coerce", utc=True)
    df.insert(0, "ts", ts)
    df["ts"] = df["ts"].dt.floor(interval)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df["ts"] = pd.to_datetime(df["ts"])
    df.set_index("ts", inplace=True)
    df.ta.bbands(length=20, append=True)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["rsi"] = ta.rsi(df["close"])
    df.dropna(inplace=True)
    return df


def add_headlines(df: pd.DataFrame, headlines_json: Optional[str]) -> pd.DataFrame:
    """Adiciona manchetes ao dataframe de candles garantindo timestamps"""

    if not headlines_json:
        return df

    # 1. Garante coluna ts e remove ambiguidade índice/coluna
    df = _ensure_ts(df, interval="5min")
    #   Se o índice também se chama 'ts', move-o para coluna única
    if df.index.name == "ts":
        df = df.reset_index(drop=True)
    #   Se o índice for DatetimeIndex sem nome, também reseta
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index(drop=True)

    # 2. Carrega manchetes
    with open(headlines_json, "rb") as f:
        h = pd.DataFrame(orjson.loads(f.read()))

    # 3. Normaliza timestamp das manchetes
    h["ts"] = (
        pd.to_datetime(h["timestamp"], errors="coerce", utc=True)
        .dt.tz_convert("UTC")
        .dt.floor("5min")
    )

    # 4. Se coluna não existir
    if "headline" not in h.columns:
        h["headline"] = np.nan

    # 5. Merge
    df = df.merge(h[["ts", "headline"]], on="ts", how="left")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--headlines-json")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = engineer(df)
    df = add_headlines(df, args.headlines_json)
    if "headline" not in df.columns:
        df["headline"] = ""
    df["headline"] = df["headline"].astype(str).fillna("")
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(df, open(args.out, "wb"))
    print("Dataset salvo em", args.out, "com", len(df), "linhas")

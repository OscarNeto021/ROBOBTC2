import argparse
import ccxt
import csv
import datetime
import pathlib
import time
import tqdm


def fetch(symbol: str, tf: str, start: str, out: str) -> None:
    ex = ccxt.binance({"enableRateLimit": True})
    since = int(ex.parse8601(f"{start}T00:00:00Z"))
    tf_ms = ex.parse_timeframe(tf) * 1000
    rows = []
    pbar = tqdm.tqdm(desc="Download", unit="candle")
    while True:
        ohlcvs = ex.fetch_ohlcv(symbol, tf, since, 1000)
        if not ohlcvs:
            break
        rows.extend(ohlcvs)
        pbar.update(len(ohlcvs))
        since = ohlcvs[-1][0] + tf_ms
        if since >= ex.milliseconds():
            break
        time.sleep(ex.rateLimit / 1000)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ts", "open", "high", "low", "close", "volume"])
        for ts, o, h, l, c, v in rows:
            iso = datetime.datetime.utcfromtimestamp(ts / 1000).isoformat()
            writer.writerow([iso, o, h, l, c, v])
    pbar.close()
    print("Salvo", len(rows), "linhas em", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fetch(args.symbol, args.interval, args.start, args.out)

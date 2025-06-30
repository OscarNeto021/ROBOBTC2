import requests, datetime, logging, json

URL = "https://api.alternative.me/fng/?limit=1&format=json"


def fetch_feargreed():
    try:
        resp = requests.get(URL, timeout=10)
        resp.raise_for_status()
        js = resp.json()
        rec = js["data"][0]
        score = float(rec["value"]) / 100.0  # 0-1
        ts = datetime.datetime.utcfromtimestamp(int(rec["timestamp"]))
    except Exception as exc:  # Timeout, JSON errors, etc.
        logging.warning("Fear&Greed indisponível (%s) – usando valor neutro", exc)
        ts, score = datetime.datetime.utcnow(), 0.0
    return {
        "timestamp": ts.isoformat(),
        "source": "FearGreedIndex",
        "sentiment_score": score,
        "news_volume": 1,
    }

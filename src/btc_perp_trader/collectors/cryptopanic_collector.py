import requests

from btc_perp_trader.config import CRYPTOPANIC_TOKEN

TOKEN = CRYPTOPANIC_TOKEN
URL = (
    "https://cryptopanic.com/api/v1/posts/?auth_token="
    + TOKEN
    + "&kind=news&filter=rising"
)


def fetch_cryptopanic():
    js = requests.get(URL, timeout=10).json()
    return [p["title"] for p in js.get("results", [])]

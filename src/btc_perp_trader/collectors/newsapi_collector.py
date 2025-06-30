import logging

import aiohttp

from btc_perp_trader.config import NEWSAPI_KEY

API_KEY = NEWSAPI_KEY
URL = (
    "https://newsapi.org/v2/everything?"
    "q=bitcoin&language=en&sortBy=publishedAt&pageSize=100&from={start}&to={end}&apiKey="
)
if API_KEY:
    URL += API_KEY
if not API_KEY:
    logging.warning(
        "NEWSAPI_KEY não definido – coletor NewsAPI inativo "
        "(define set NEWSAPI_KEY=... p/ ativar)."
    )


async def fetch_news(start: str, end: str):
    if not API_KEY:
        return []
    async with aiohttp.ClientSession() as sess:
        async with sess.get(URL.format(start=start, end=end)) as r:
            js = await r.json()
    return [a["title"] for a in js.get("articles", [])]

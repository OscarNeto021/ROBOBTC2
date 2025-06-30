import asyncio
import datetime
import logging

import aiohttp

from btc_perp_trader.config import CRYPTOPANIC_TOKEN, NEWSAPI_KEY

NEWS_QUEUE = asyncio.Queue()
logger = logging.getLogger("NewsCollector")

NEWS_FEEDS = [
    f"https://newsapi.org/v2/top-headlines?category=business&q=bitcoin&apiKey={NEWSAPI_KEY}",
    f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_TOKEN}&kind=news",
]


async def fetch(session, url):
    async with session.get(url) as r:
        return await r.json()


async def news_loop():
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                results = await asyncio.gather(*(fetch(session, u) for u in NEWS_FEEDS))
                headlines = []
                for data in results:
                    if "articles" in data:  # NewsAPI
                        headlines.extend(a["title"] for a in data["articles"])
                    elif "results" in data:  # CryptoPanic
                        headlines.extend(p["title"] for p in data["results"])
                joined = ". ".join(headlines[:20])
                ts = datetime.datetime.utcnow().isoformat()
                await NEWS_QUEUE.put({"timestamp": ts, "headline": joined})
            except Exception as e:
                logger.warning("News fetch error: %s", e)
            await asyncio.sleep(300)

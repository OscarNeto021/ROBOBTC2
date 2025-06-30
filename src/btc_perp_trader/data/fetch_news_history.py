import asyncio
import datetime
import json
import pathlib
import random

import orjson
import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..collectors import (
    cryptopanic_collector as cp,
)
from ..collectors import (
    feargreed_collector as fg,
)
from ..collectors import (
    newsapi_collector as napi,
)
from ..collectors import (
    reddit_collector as rd,
)

CACHE_DIR = pathlib.Path("data/news_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_path(date_iso: str) -> pathlib.Path:
    return CACHE_DIR / f"{date_iso}.json"


analyser = SentimentIntensityAnalyzer()


async def gather_day(day_iso: str, out_list: list, use_real: bool = True):
    start = day_iso + "T00:00:00Z"
    end = day_iso + "T23:59:00Z"
    # ---------- COLETA ----------
    titles: list[str] = []
    if use_real:
        titles += await napi.fetch_news(start, end)
        titles += cp.fetch_cryptopanic()
        titles += rd.fetch_reddit(start, end)
    else:  # modo sintético: usa cache ou cria dummy headline
        if cache_path(day_iso).exists():
            titles = json.loads(cache_path(day_iso).read_text())
        else:
            titles = [f"dummy headline {i}" for i in range(random.randint(3, 8))]
            cache_path(day_iso).write_text(json.dumps(titles))
    for t in titles:
        ss = analyser.polarity_scores(t)["compound"]
        out_list.append(
            {
                "timestamp": start,
                "source": "news_mix",
                "headline": t,
                "sentiment_score": ss,
                "news_volume": 1,
            }
        )
    out_list.append(fg.fetch_feargreed())


async def main(start: str, out: str, use_real_api: bool):
    s = datetime.date.fromisoformat(start)
    e = datetime.date.today()
    out_list = []
    for d in tqdm.tqdm(
        [s + datetime.timedelta(days=i) for i in range((e - s).days + 1)]
    ):
        real = use_real_api and d == datetime.date.today()
        await gather_day(d.isoformat(), out_list, use_real=real)
    pathlib.Path(out).write_bytes(orjson.dumps(out_list))
    print(
        "Salvo",
        len(out_list),
        "registros →",
        out,
        "(modo real)" if use_real_api else "(modo cache/dummy)",
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--mode",
        choices=["real", "cache"],
        default="real",
        help="'real' = usa APIs HOJE, 'cache' = histórico offline",
    )
    args = ap.parse_args()
    asyncio.run(main(args.start, args.out, use_real_api=args.mode == "real"))

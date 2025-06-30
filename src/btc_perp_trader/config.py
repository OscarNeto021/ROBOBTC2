import os

from dotenv import load_dotenv

load_dotenv()

BINANCE_API_KEY = os.getenv(
    "BINANCE_API_KEY",
    "a8b5e8012ee9c199afff56e98d755b0356c50f15e84052ad07c7fff59abdc5d8",
)
BINANCE_API_SECRET = os.getenv(
    "BINANCE_API_SECRET",
    "e4db4831e81baca85e3a7df5d8d8170c11b58e6a4d6c47ceef81374b6e366e81",
)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "716824a957c9486abe4b1bbccc401e6e")
CRYPTOPANIC_TOKEN = os.getenv("CRYPTOPANIC_TOKEN", "716824a957c9486abe4b1bbccc401e6e")

missing = [
    k
    for k, v in {
        "BINANCE_API_KEY": BINANCE_API_KEY,
        "BINANCE_API_SECRET": BINANCE_API_SECRET,
        "NEWSAPI_KEY": NEWSAPI_KEY,
        "CRYPTOPANIC_TOKEN": CRYPTOPANIC_TOKEN,
    }.items()
    if not v
]
if missing:
    raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

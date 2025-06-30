import asyncio
from btc_perp_trader.collectors import news_collector


async def trader_loop():
    while True:
        await asyncio.sleep(1)


async def main():
    queue = asyncio.Queue()
    ws = None  # placeholder for market data websocket
    user_ws = None  # placeholder for user stream
    await asyncio.gather(
        *(t for t in [
            news_collector.news_loop(),
            trader_loop()
        ])
    )


if __name__ == "__main__":
    asyncio.run(main())

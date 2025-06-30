import asyncio
import json
import logging
import aiohttp

LOGGER = logging.getLogger("BinanceUserWS")
BASE_WSS = "wss://stream.binancefuture.com/ws/"


class BinanceUserStream:
    """Listen to Binance user-data stream and trigger callback on realized PnL."""

    KEEPALIVE_SEC = 30 * 60  # 30 min

    def __init__(self, ccxt_ex, on_realized_pnl):
        self.ex = ccxt_ex
        self.on_realized_pnl = on_realized_pnl
        self.listen_key = None
        self.ws = None

    async def _keepalive(self) -> None:
        while True:
            await asyncio.sleep(self.KEEPALIVE_SEC)
            try:
                self.ex.fapiPrivatePutListenKey({"listenKey": self.listen_key})
                LOGGER.debug("listenKey keep-alive ok")
            except Exception as exc:  # pragma: no cover - depends on ccxt network
                LOGGER.warning("keep-alive failed: %s", exc)

    async def _connect(self) -> None:
        self.listen_key = self.ex.fapiPrivatePostListenKey()["listenKey"]
        url = BASE_WSS + self.listen_key
        session = aiohttp.ClientSession()
        self.ws = await session.ws_connect(url)
        LOGGER.info("User-data WS conectado.")
        asyncio.create_task(self._keepalive())

    async def loop(self) -> None:
        if not self.ws:
            await self._connect()
        async for msg in self.ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            if data.get("e") == "ACCOUNT_UPDATE":
                for p in data["a"].get("P", []):
                    rp = float(p["rp"])
                    pa = float(p["pa"])
                    sym = p["s"]
                    if rp != 0 and pa == 0:
                        LOGGER.info("Realized PnL %s %s", sym, rp)
                        self.on_realized_pnl(sym, rp)

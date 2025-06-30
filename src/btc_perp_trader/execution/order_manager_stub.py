"""Simple order manager using ccxt testnet when USE_TESTNET is true."""

import os

import ccxt

from btc_perp_trader.config import BINANCE_API_KEY, BINANCE_API_SECRET


class OrderManagerStub:
    def __init__(self, exchange: str = "binance"):
        self.use_testnet = os.getenv("USE_TESTNET", "true").lower() == "true"
        self.exchange_name = exchange
        self.client = self._create_client()

    def _create_client(self):
        if self.exchange_name == "binance":
            params = {
                "apiKey": BINANCE_API_KEY,
                "secret": BINANCE_API_SECRET,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
            client = ccxt.binance(params)
            if self.use_testnet:
                client.set_sandbox_mode(True)
            return client
        raise NotImplementedError(f"Exchange {self.exchange_name} not supported")

    def create_order(self, symbol: str, side: str, qty: float):
        """Place a market order (stub)."""
        typ = "buy" if side.lower() == "buy" else "sell"
        return {
            "id": "stub",
            "symbol": symbol,
            "side": typ,
            "quantity": qty,
            "testnet": self.use_testnet,
        }

import importlib

import pytest
from dotenv import load_dotenv

if importlib.util.find_spec("ccxt") is None:
    pytest.skip("ccxt not installed", allow_module_level=True)

import ccxt

from btc_perp_trader.config import BINANCE_API_KEY, BINANCE_API_SECRET

load_dotenv()


def main():
    ex = ccxt.binance(
        {
            "apiKey": BINANCE_API_KEY,
            "secret": BINANCE_API_SECRET,
            "options": {"defaultType": "future"},
        }
    )
    ex.set_sandbox_mode(True)
    print(ex.fapiPublic_get_ping())
    print("Ping OK – testnet conectado!")


if __name__ == "__main__":
    main()

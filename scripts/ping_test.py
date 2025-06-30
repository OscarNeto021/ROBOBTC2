from dotenv import load_dotenv
import importlib
import pytest

if importlib.util.find_spec("ccxt") is None:
    pytest.skip("ccxt not installed", allow_module_level=True)

load_dotenv()
import os
import ccxt

def main():
    ex = ccxt.binance({
        "apiKey": os.getenv("BINANCE_API_KEY"),
        "secret": os.getenv("BINANCE_API_SECRET"),
        "options": {"defaultType": "future"},
    })
    ex.set_sandbox_mode(True)
    print(ex.fapiPublic_get_ping())
    print("Ping OK – testnet conectado!")

if __name__ == "__main__":
    main()

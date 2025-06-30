from pathlib import Path
from dotenv import dotenv_values, set_key

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def get_keys() -> dict:
    return dotenv_values(ENV_PATH)


def save_keys(key: str, secret: str, demo: bool = True) -> None:
    if demo:
        set_key(ENV_PATH, "BINANCE_API_KEY_DEMO", key)
        set_key(ENV_PATH, "BINANCE_SECRET_KEY_DEMO", secret)
    else:
        set_key(ENV_PATH, "BINANCE_API_KEY", key)
        set_key(ENV_PATH, "BINANCE_SECRET_KEY", secret)

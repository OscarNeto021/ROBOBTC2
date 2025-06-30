from __future__ import annotations
import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "trades.csv"

@dataclass
class Trade:
    ts: str; symbol: str; side: str; qty: float; price: float; pnl: float
    def row(self): return asdict(self)

if not LOG_FILE.exists():
    with LOG_FILE.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=Trade("", "", "", 0, 0, 0).row().keys()).writeheader()

def _append(trade: Trade) -> None:
    with LOG_FILE.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=trade.row().keys()).writerow(trade.row())

def log_open(symbol: str, side: str, qty: float, price: float):
    _append(Trade(datetime.utcnow().isoformat(), symbol, side, qty, price, 0.0))

def log_close(symbol: str, qty: float, entry: float, exit: float):
    pnl = (exit - entry) * qty * (1 if qty > 0 else -1)
    _append(Trade(datetime.utcnow().isoformat(), symbol, "close", qty, exit, pnl))


def is_recorded(ts: str, symbol: str, side: str) -> bool:
    """Check if a trade is already recorded in the log."""
    if not LOG_FILE.exists():
        return False
    with LOG_FILE.open() as fh:
        return any(ts in line and symbol in line and side in line for line in fh.readlines())


def log_realized(symbol: str, pnl: float) -> None:
    """Record realized PnL coming from the user-data stream."""
    now = datetime.utcnow().isoformat()
    _append(Trade(now, symbol, "realized", 0.0, 0.0, pnl))

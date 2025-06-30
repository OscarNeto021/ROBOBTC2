from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Signal:
    side: str | None  # "long", "short" ou None
    confidence: float


def entry_signal(long_p: float, short_p: float, thr: float = 0.55) -> Signal:
    if long_p > thr and long_p > short_p:
        return Signal("long", long_p)
    if short_p > thr and short_p > long_p:
        return Signal("short", short_p)
    return Signal(None, 0.0)

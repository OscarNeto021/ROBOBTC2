"""
Funções de sizing de posição.

atr_position_size(balance, atr, risk_pct)
-----------------------------------------
Arrisca `risk_pct` do saldo por trade, usando ATR como stop implícito.
"""

from __future__ import annotations
from typing import Optional
import os

# -------- saldo virtual somente para TESTNET ------------------------------------------
VIRTUAL_BALANCE_CAP = float(
    os.getenv("VIRTUAL_BALANCE_CAP", 100.0)
)  # default: 100 USDT


def atr_position_size(
    balance: float,
    atr: Optional[float],
    *,
    price: Optional[float],
    risk_pct: float = 0.005,  # default; pode ser sobrescrito
    leverage: int = 20,
    contract_value: float = 1.0,
) -> float:
    """Calcula o tamanho da posição considerando risco e alavancagem."""

    # aplica o cap virtual
    balance = min(balance, VIRTUAL_BALANCE_CAP)

    if atr is None or atr <= 0 or price is None or price <= 0:
        return 0.0

    risk_amount = balance * risk_pct
    qty_by_atr = risk_amount / (atr * contract_value) * 0.9
    qty_by_leverage = (balance * leverage) / (price * contract_value) * 0.9

    qty = min(round(qty_by_atr, 3), qty_by_leverage)
    return max(qty, 0.0)

import numpy as np

from btc_perp_trader.risk.risk_manager import RiskMetrics
from btc_perp_trader.risk.position_sizing import atr_position_size, VIRTUAL_BALANCE_CAP


def test_var_cvar_reference():
    returns = np.array([-0.05, -0.02, 0.01, 0.03, -0.01])
    var = RiskMetrics.calculate_var(returns, 0.05)
    cvar = RiskMetrics.calculate_cvar(returns, 0.05)
    assert np.isclose(var, np.percentile(returns, 5))
    assert cvar <= var


def test_atr_position_size():
    qty = atr_position_size(1000, 100, price=50000)
    expected = round((min(1000, VIRTUAL_BALANCE_CAP) * 0.005) / (100 * 1.0) * 0.9, 3)
    assert qty == expected
    assert qty >= 0


def test_atr_position_size_invalid_atr():
    assert atr_position_size(1000, 0, price=50000) == 0.0
    assert atr_position_size(1000, -10, price=50000) == 0.0
    assert atr_position_size(1000, None, price=50000) == 0.0


def test_atr_position_size_leverage_cap():
    qty = atr_position_size(1000, 1, price=20000, leverage=20)
    expected = (min(1000, VIRTUAL_BALANCE_CAP) * 20) / (20000 * 1.0) * 0.9
    assert qty == expected

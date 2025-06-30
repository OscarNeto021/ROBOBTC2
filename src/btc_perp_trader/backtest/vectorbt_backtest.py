import numpy as np
import pandas as pd
import vectorbt as vbt


def run_backtest():
    """Run a simple vectorbt backtest returning a Portfolio."""
    # Gerar dados de preço mais realistas
    np.random.seed(42)
    price = pd.Series(100 + np.cumsum(np.random.randn(1000)), name="BTC")

    # Gerar sinais de entrada e saída mais complexos
    fast_ma = price.ewm(span=10).mean()
    slow_ma = price.ewm(span=50).mean()

    entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

    pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=100_000)
    return pf



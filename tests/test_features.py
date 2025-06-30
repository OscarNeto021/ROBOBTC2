import numpy as np
import pandas as pd
import importlib
import pytest

try:
    import pandas_ta  # noqa: F401
except Exception:
    pytest.skip("pandas_ta not installed", allow_module_level=True)

from btc_perp_trader.features.feature_engineering import FeatureEngineer


def test_feature_generation_no_nan():
    ts = pd.date_range('2024-01-01', periods=10, freq='1min')
    data = pd.DataFrame({
        'timestamp': ts,
        'open': np.arange(10),
        'high': np.arange(10)+1,
        'low': np.arange(10)-1,
        'close': np.arange(10),
        'volume': np.random.randint(1, 5, size=10)
    })
    fe = FeatureEngineer()
    out = fe.process_full_pipeline(data)
    assert out.shape[0] == 10
    assert out.shape[1] > data.shape[1]
    assert not out.isna().any().any()

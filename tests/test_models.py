import numpy as np
import pandas as pd
import importlib
import pytest


def _sample_df(n=50):
    ts = pd.date_range('2024-01-01', periods=n, freq='1min')
    data = pd.DataFrame({
        'timestamp': ts,
        'open': np.random.rand(n),
        'high': np.random.rand(n),
        'low': np.random.rand(n),
        'close': np.random.rand(n),
        'volume': np.random.rand(n)
    })
    return data


@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="xgboost not installed")
def test_xgboost_prediction_shape_dtype():
    from btc_perp_trader.models.xgboost_model import XGBoostModel
    df = _sample_df()
    fe = XGBoostModel({'n_estimators': 5})
    X_train, X_val, X_test, y_train, y_val, y_test, cols = fe.prepare_data(
        df.assign(target_return_1=np.random.rand(len(df))), 'target_return_1'
    )
    fe.train(X_train, y_train, X_val, y_val)
    preds = fe.predict(X_test)
    assert preds.shape[0] == X_test.shape[0]
    assert preds.dtype == np.float64


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_lstm_output_shape_dtype():
    from btc_perp_trader.models.lstm_model import LSTMModel
    df = _sample_df(80)
    model = LSTMModel(
        {
            'epochs': 1,
            'sequence_length': 5,
            'hidden_size': 4,
            'num_layers': 1,
        }
    )
    X_train, X_val, X_test, y_train, y_val, y_test, cols = model.prepare_data(
        df.assign(target_return_1=np.random.rand(len(df))), 'target_return_1'
    )
    model.train(X_train, y_train, X_val, y_val)
    out = model.predict(X_test[:10])
    assert out.shape[0] == 10
    assert out.dtype == np.float32 or out.dtype == np.float64

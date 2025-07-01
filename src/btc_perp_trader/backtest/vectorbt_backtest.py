import pandas as pd
import vectorbt as vbt

from btc_perp_trader.models.train_ensemble import load_dataset
from btc_perp_trader.models.ensemble_model import EnsembleModel


def run_backtest() -> vbt.Portfolio:
    """Run a backtest using ML predictions as trading signals."""

    df = load_dataset()
    model = EnsembleModel.load_or_train()

    probs = df.apply(
        lambda row: model.predict_proba(
            row.drop(labels=["label"], errors="ignore")
        ),
        axis=1,
    )
    df[["p_long", "p_short"]] = pd.DataFrame(list(probs), index=df.index)

    price = df["close"]

    mean_long = df["p_long"].mean()
    std_long = df["p_long"].std()
    mean_short = df["p_short"].mean()
    std_short = df["p_short"].std()

    long_entries = df["p_long"] > mean_long + std_long
    long_exits = df["p_long"] < mean_long - std_long
    short_entries = df["p_short"] > mean_short + std_short
    short_exits = df["p_short"] < mean_short - std_short

    pf = vbt.Portfolio.from_signals(
        price,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        sl_stop=0.01,
        tp_stop=0.02,
        init_cash=100_000,
    )
    return pf



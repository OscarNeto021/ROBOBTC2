"""pipeline/offline_train.py

Atualizado: 2025-07-02.

* Permite definir data inicial via --start (default = 2023-01-01)
* Gera dataset e mantém ponteiro data/latest_feats.pkl
* Treina ensemble offline e pré-treina modelo on-line (River)
"""

import argparse
import datetime as dt
import pathlib
import subprocess
import shutil
import pickle
import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def sh(cmd: str) -> None:
    """Executa um comando shell exibindo-o antes."""
    print("\u25b6", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--start",
        default="2023-01-01",
        help="Data inicial YYYY-MM-DD para baixar candles/headlines",
    )
    return p.parse_args()


def main():
    args = parse_args()
    today = dt.date.today().isoformat()

    # 1) Candles ----------------------------------------------------------------
    ohlcv_csv = DATA / f"btc_1m_{args.start}_{today}.csv"
    if not ohlcv_csv.exists():
        sh(
            f"python -m btc_perp_trader.data.fetch_binance "
            f"--symbol BTCUSDT --interval 1m --start {args.start} --out {ohlcv_csv}"
        )

    # 2) Manchetes --------------------------------------------------------------
    headlines_json = DATA / "headlines.json"
    sh(
        f"python -m btc_perp_trader.data.fetch_news_history "
        f"--start {args.start} --out {headlines_json} --mode real"
    )

    # 3) Features ---------------------------------------------------------------
    feats_pkl = DATA / f"btc_feats_until_{today}.pkl"
    sh(
        f"python -m btc_perp_trader.models.build_dataset "
        f"--input {ohlcv_csv} --out {feats_pkl} --headlines-json {headlines_json}"
    )

    # ponteiro 'latest' (link simbólico quando possível) -----------------------
    latest = DATA / "latest_feats.pkl"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(feats_pkl)
    except (OSError, NotImplementedError):
        shutil.copyfile(feats_pkl, latest)

    # 4) Ensemble offline -------------------------------------------------------
    sh(
        f"python -m btc_perp_trader.models.train_ensemble "
        f"--dataset {feats_pkl} --out {MODELS / 'default.pkl'}"
    )

    # 5) Pré-treino on-line (River) --------------------------------------------
    from btc_perp_trader.models.online_model import ONLINE_MODEL  # noqa: WPS433
    import pandas as pd  # noqa: WPS433

    df: pd.DataFrame = pickle.load(open(feats_pkl, "rb"))
    labels = (df["close"].shift(-1) > df["close"]).astype(int)

    for x, y in tqdm.tqdm(
        zip(df.drop(columns=["close"]).to_dict("records"), labels),
        total=len(df),
        desc="Pre-train River",
    ):
        ONLINE_MODEL.learn(x, int(y))

    print("ROC pré-treino:", ONLINE_MODEL.roc())


if __name__ == "__main__":
    main()

"""Offline pipeline for daily dataset download and model training."""

import datetime
import pathlib
import pickle
import subprocess

import tqdm

from btc_perp_trader.models.online_model import ONLINE_MODEL

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
MODELS = ROOT / "models"


def sh(cmd: str) -> None:
    print(">>", cmd)
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    today = datetime.date.today().isoformat()
    ohlcv_csv = DATA / f"btc_1m_{today}.csv"
    dataset_pkl = DATA / f"btc_feats_{today}.pkl"
    # 1a. baixar/atualizar OHLCV
    # if not ohlcv_csv.exists():
    #     sh(
    #         f"python -m btc_perp_trader.data.fetch_binance --symbol BTCUSDT "
    #         f"--interval 1m --start 2023-01-01 --out {ohlcv_csv}"
    #     )

    # 1b. baixar/atualizar manchetes (usa 'real' só no dia atual)
    # head_json = DATA / "headlines.json"
    # mode = "real" if today == datetime.date.today().isoformat() else "cache"
    # sh(
    #     f"python -m btc_perp_trader.data.fetch_news_history --start 2023-01-01 "
    #     f"--out {head_json} --mode {mode}"
    # )

    # Usando dados dummy para OHLCV e manchetes para contornar problemas de API
    ohlcv_csv = DATA / "btc_1m_2025-06-30.csv"
    head_json = DATA / "headlines.json"    # 2. build dataset (agora com headlines.json)
    sh(
        f"python -m btc_perp_trader.models.build_dataset --input {ohlcv_csv} "
        f"--out {dataset_pkl} --headlines-json {head_json}"
    )

    sh(
        f"python -m btc_perp_trader.models.train_ensemble "
        f"--dataset {dataset_pkl} --out {MODELS / 'default.pkl'}"
    )

    df = pickle.load(open(dataset_pkl, "rb"))
    for x, y in tqdm.tqdm(
        zip(
            df.drop(["close"], axis=1).to_dict("records"),
            (df["close"].shift(-1) > df["close"]).astype(int),
        ),
        total=len(df),
        desc="Pre-train River",
    ):
        ONLINE_MODEL.learn(x, int(y))
    print("ROC pré-treino:", ONLINE_MODEL.roc())


if __name__ == "__main__":
    main()

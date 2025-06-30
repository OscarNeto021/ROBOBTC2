import datetime
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def sh(cmd: str):
    print("\u25b6", cmd)
    sys.stdout.flush()
    subprocess.run(cmd, shell=True, check=True)


def main():
    today = datetime.date.today().isoformat()

    # 1. Baixar candles reais da Binance
    ohlcv_csv = DATA / f"btc_1m_until_{today}.csv"
    if not ohlcv_csv.exists():
        sh(
            f"python -m btc_perp_trader.data.fetch_binance "
            f"--symbol BTCUSDT --interval 1m --start 2024-06-30 "
            f"--out {ohlcv_csv}"
        )

    # 2. Baixar ou atualizar as manchetes
    headlines_json = DATA / "headlines.json"
    sh(
        f"python -m btc_perp_trader.data.fetch_news_history "
        f"--start 2024-06-30 --out {headlines_json} --mode real"
    )

    # 3. Gerar features
    feats_pkl = DATA / f"btc_feats_{today}.pkl"
    sh(
        f"python -m btc_perp_trader.models.build_dataset "
        f"--input {ohlcv_csv} --out {feats_pkl} --headlines-json {headlines_json}"
    )

    # 4. Treinar modelo ensemble
    sh(
        f"python -m btc_perp_trader.models.train_ensemble "
        f"--dataset {feats_pkl} --out {MODELS / 'default.pkl'}"
    )

    # 5. Treinar modelo online River
    import pickle

    import tqdm

    from btc_perp_trader.models.online_model import ONLINE_MODEL

    df = pickle.load(open(feats_pkl, "rb"))
    for x, y in tqdm.tqdm(
        zip(
            df.drop(columns=["close"]).to_dict("records"),
            (df["close"].shift(-1) > df["close"]).astype(int),
        ),
        total=len(df),
        desc="Pre-train River",
    ):
        ONLINE_MODEL.learn(x, int(y))
    print("ROC pré-treino:", ONLINE_MODEL.roc())


if __name__ == "__main__":
    main()

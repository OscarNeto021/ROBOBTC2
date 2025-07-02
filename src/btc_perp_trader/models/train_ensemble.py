"""Treino / utilitários para o EnsembleModel.

• Procura sempre o dataset mais recente (`data/latest_feats.pkl` ou
  o .pkl mais novo) e oferece uma CLI:

    poetry run python -m btc_perp_trader.models.train_ensemble
    poetry run python -m btc_perp_trader.models.train_ensemble --dataset data/btc_feats_until_2025-07-01.pkl --out models/meu_modelo.pkl
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import pickle
from typing import Optional

import pandas as pd

from btc_perp_trader.models.ensemble_model import EnsembleModel

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _latest_feats_path() -> Optional[pathlib.Path]:
    latest = DATA_DIR / "latest_feats.pkl"
    if latest.exists():
        return latest
    pkls = sorted(
        DATA_DIR.glob("btc_feats_*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return pkls[0] if pkls else None


def load_dataset(path: pathlib.Path | None = None) -> pd.DataFrame:
    """Carrega o DataFrame de features para treinar/avaliar modelos."""
    if path:
        return pickle.load(path.open("rb"))
    p = _latest_feats_path()
    if not p:
        raise FileNotFoundError(
            "Nenhum dataset encontrado em 'data/'. Execute offline_train.py primeiro."
        )
    _logger.info("Carregando dataset %s", p.name)
    return pickle.load(p.open("rb"))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=pathlib.Path,
        default=None,
        help="Caminho opcional para .pkl de features",
    )
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=ROOT / "models/default.pkl",
        help="Arquivo onde o modelo será salvo",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    df = load_dataset(args.dataset)

    _logger.info("Treinando EnsembleModel com %d amostras", len(df))
    # Usa API interna do EnsembleModel — se existir _train_full_from_df, preferimos
    mdl = (
        EnsembleModel._train_full_from_df(df)  # type: ignore[attr-defined]
        if hasattr(EnsembleModel, "_train_full_from_df")
        else EnsembleModel._train_full(name="default")
    )

    with args.out.open("wb") as fh:
        pickle.dump(mdl, fh)
    print(f"✅ Modelo salvo em {args.out}")  # noqa: WPS421


if __name__ == "__main__":  # pragma: no cover
    main()

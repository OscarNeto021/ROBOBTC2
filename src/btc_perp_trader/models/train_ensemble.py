"""Treino / utilitários para o EnsembleModel (versão 2025-07-02).

• Converte colunas datetime64 → Unix-time antes de treinar  
• `--dataset` agora força o EnsembleModel a usar exatamente esse arquivo
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

_FORCED_PATH: Optional[pathlib.Path] = None  # definido pelo CLI


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


def _coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma colunas datetime64[ns] em segundos Unix (float)."""
    dt_cols = df.select_dtypes(include="datetime").columns
    if len(dt_cols):
        df = df.copy()
        for col in dt_cols:
            df[col] = df[col].view("int64") / 1_000_000_000  # ns → s
    return df


def load_dataset(path: pathlib.Path | None = None) -> pd.DataFrame:
    global _FORCED_PATH  # pragma: no cover
    path = path or _FORCED_PATH or _latest_feats_path()
    if not path:
        raise FileNotFoundError(
            "Nenhum dataset encontrado em 'data/'. Execute offline_train.py primeiro."
        )
    df = pickle.load(path.open("rb"))
    return _coerce_datetimes(df)


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
    global _FORCED_PATH  # noqa: WPS420
    args = _parse_args()
    _FORCED_PATH = args.dataset  # garante que EnsembleModel use este dataset

    # EnsembleModel internamente chamará train_ensemble.load_dataset()
    _logger.info("Treinando EnsembleModel …")
    model = EnsembleModel._train_full(name="default")

    with args.out.open("wb") as fh:
        pickle.dump(model, fh)
    print(f"✅ Modelo salvo em {args.out}")  # noqa: WPS421


if __name__ == "__main__":  # pragma: no cover
    main()

"""Treino / utilitários para o EnsembleModel (versão 2025-07-02).

• Converte colunas datetime64 → Unix-time antes de treinar  
• `--dataset` agora força o EnsembleModel a usar exatamente esse arquivo
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import pickle
from typing import List, Optional

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
    """Sanitiza o DataFrame para conter **apenas valores numéricos** (float32).

    Regras:
      • ÍNDICE datetime   -> reset/drop
      • Colunas ['date', 'datetime', 'timestamp', ...] -> drop
      • Demais datetime  -> int64 (segundos Unix)
      • Colunas object   -> tenta converter via `pd.to_numeric`
    """
    import numpy as np
    import pandas as pd
    from pandas.api.types import is_datetime64_any_dtype

    df = df.copy()

    # 0) Remover DatetimeIndex -------------------------------------------------
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index(drop=True)

    # 1) Regras de drop/conversão ---------------------------------------------
    DROP_KEYWORDS: List[str] = ["date", "datetime", "timestamp"]

    for col in list(df.columns):
        # a) DROP se nome indica coluna de data pura --------------------------
        if any(key in col.lower() for key in DROP_KEYWORDS):
            df = df.drop(columns=[col])
            continue

        # b) datetime64 direto -------------------------------------------------
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype("int64") // 1_000_000_000
            continue

        # c) object possivelmente datetime ------------------------------------
        if df[col].dtype == "object":
            try:
                dt_series = pd.to_datetime(df[col], errors="raise")
                if is_datetime64_any_dtype(dt_series):
                    df[col] = dt_series.astype("int64") // 1_000_000_000
                    continue
            except Exception:
                pass  # não é datetime, tenta numérico abaixo

            # d) tenta converter para numérico (strings de número) ------------
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2) Garante float32 -------------------------------------------------------
    df = df.apply(pd.to_numeric, errors="ignore")
    non_numeric = df.select_dtypes(exclude=["number"]).columns
    # Nunca descartamos a coluna 'label'
    drop_cols = [c for c in non_numeric if c != "label"]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 3) Cast final ------------------------------------------------------------
    feat_cols = [c for c in df.columns if c != "label"]
    df[feat_cols] = df[feat_cols].astype(np.float32)
    if "label" in df.columns and df["label"].dtype != "int32":
        df["label"] = (
            pd.to_numeric(df["label"], errors="coerce")
            .fillna(0)
            .astype("int32")
        )

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

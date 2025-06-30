from __future__ import annotations
import logging, pickle, pathlib, json, joblib
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)
ROOT   = pathlib.Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ---------- 1. TFT-LSTM MINIMAL (PyTorch) ------------------------------------
class _TFT(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, lstm_layers: int = 2):
        super().__init__()
        self.lstm  = nn.LSTM(in_dim, hidden, lstm_layers, batch_first=True)
        self.fc    = nn.Linear(hidden, 1)

    def forward(self, x):                 # x: (B,T,F)
        out, _ = self.lstm(x)
        out    = out[:, -1]               # último passo
        return torch.sigmoid(self.fc(out))

# ---------- 2. EnsembleModel --------------------------------------------------
@dataclass
class _SubModels:
    xgb:       XGBClassifier
    tft:       _TFT
    stack:     LogisticRegression

class EnsembleModel:
    """
    • load_or_train(name)   – carrega, senão treina full-stack.
    • predict_proba(SERIES) – devolve (p_long, p_short) ∈ [0,1].
    """

    def __init__(self, subs: _SubModels, feats_order: List[str]) -> None:
        self.subs = subs
        self.order = feats_order

    # ----------------------------- API pública --------------------------------
    @classmethod
    def load_or_train(cls, name: str = "default") -> "EnsembleModel":
        path = MODEL_DIR / f"{name}.pkl"
        if path.exists():
            with path.open("rb") as fh:
                logger.info("🔹 Modelo %s carregado", path.name)
                return pickle.load(fh)

        logger.info("🔸 Modelo %s não encontrado – iniciando treino…", name)
        model = cls._train_full(name)
        with path.open("wb") as fh:
            pickle.dump(model, fh)
        logger.info("✅ Treino concluído e salvo em %s", path)
        return model

    def predict_proba(self, features: pd.Series | dict) -> Tuple[float, float]:
        if isinstance(features, dict):
            features = pd.Series(features)
        # Garante alinhamento: se faltar alguma feature, preenche com 0.0
        x = np.array([features.get(col, 0.0) for col in self.order],
                     dtype=np.float32).reshape(1, -1)

        # ------------------- XGB + TFT ---------------------------------------
        p_xgb = self.subs.xgb.predict_proba(x)[0, 1]       # prob de CLASSE 1 = short
        # TFT: precisa shape (B,T,F) → usamos T=1
        with torch.no_grad():
            p_tft = self.subs.tft(torch.from_numpy(x).unsqueeze(1)).item()

        # ---------- garante valores finitos -------------------------------
        if not np.isfinite(p_xgb):
            p_xgb = 0.5
        if not np.isfinite(p_tft):
            p_tft = 0.5

        # ------------------- empilhamento ------------------------------------
        stack_input = np.array([[p_xgb, p_tft]], dtype=np.float32)
        p_short = self.subs.stack.predict_proba(stack_input)[0, 1]
        p_long  = 1.0 - p_short
        return p_long, p_short

    # ----------------------------- treino full --------------------------------
    @classmethod
    def _train_full(cls, name: str) -> "EnsembleModel":
        from btc_perp_trader.models.train_ensemble import load_dataset
        df = load_dataset()                                # pandas DataFrame

        feats = [c for c in df.columns if c not in ("label", "timestamp")]
        X = df[feats].values.astype(np.float32)
        y = df["label"].values.astype(int)

        # ---- 1. XGB hiper-tune simplificado (Optuna, 30 trials) -------------
        import optuna
        def objective(trial):
            params = dict(
                max_depth      = trial.suggest_int("max_depth", 3, 8),
                learning_rate  = trial.suggest_float("lr", 0.03, 0.3, log=True),
                n_estimators   = trial.suggest_int("n_est", 200, 800),
                subsample      = trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree = trial.suggest_float("colsample", 0.5, 1.0),
            )
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in skf.split(X, y):
                clf = XGBClassifier(**params, objective="binary:logistic", verbosity=0)
                clf.fit(X[train_idx], y[train_idx])
                scores.append(clf.score(X[val_idx], y[val_idx]))
            return 1 - np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        xgb_best = XGBClassifier(objective="binary:logistic", **study.best_params)
        xgb_best.fit(X, y)

        # ---- 2. TFT-LSTM (PyTorch CPU/GPU) ----------------------------------
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tft = _TFT(in_dim=len(feats)).to(device)
        loss_fn = nn.BCELoss()
        opt = optim.Adam(tft.parameters(), lr=1e-3)
        X_t = torch.from_numpy(X).unsqueeze(1).to(device)      # (N,1,F)
        y_t = torch.from_numpy(y).float().unsqueeze(1).to(device)
        for epoch in range(10):                                # epochs rápidos
            pred = tft(X_t)
            loss = loss_fn(pred, y_t)
            opt.zero_grad(); loss.backward(); opt.step()

        # ---- 3. Stack logistic ---------------------------------------------
        p_xgb  = xgb_best.predict_proba(X)[:, 1]
        with torch.no_grad():
            p_tft = tft(X_t).cpu().numpy().flatten()
        stack = LogisticRegression()
        stack.fit(np.column_stack([p_xgb, p_tft]), y)

        subs = _SubModels(xgb_best, tft.cpu(), stack)
        return EnsembleModel(subs, feats)

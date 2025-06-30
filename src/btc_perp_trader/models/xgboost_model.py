"""
Modelo XGBoost para trading de BTC-PERP.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import xgboost as xgb

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """Modelo XGBoost para previsão de preços/retornos."""

    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": 50,
            "eval_metric": "rmse",
        }

        if config:
            default_config.update(config)

        super().__init__("XGBoost", default_config)

    def build_model(self, input_shape: Tuple[int, ...]) -> xgb.XGBRegressor:
        """Constrói o modelo XGBoost."""
        logger.info(f"Construindo modelo XGBoost com input_shape: {input_shape}")

        self.model = xgb.XGBRegressor(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            learning_rate=self.config["learning_rate"],
            subsample=self.config["subsample"],
            colsample_bytree=self.config["colsample_bytree"],
            random_state=self.config["random_state"],
            eval_metric=self.config["eval_metric"],
            early_stopping_rounds=self.config["early_stopping_rounds"],
        )

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """Treina o modelo XGBoost."""
        logger.info(f"Treinando {self.model_name}")

        if self.model is None:
            self.build_model((X_train.shape[1],))

        # Preparar dados de validação
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        else:
            eval_set = [(X_train, y_train)]

        # Treinar modelo
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        self.is_trained = True

        # Calcular feature importance
        self.feature_importance = {
            "importance_gain": self.model.feature_importances_,
            "importance_weight": self.model.get_booster().get_score(
                importance_type="weight"
            ),
            "importance_cover": self.model.get_booster().get_score(
                importance_type="cover"
            ),
        }

        # Histórico de treinamento
        self.training_history = {
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score,
            "validation_scores": self.model.evals_result(),
        }

        logger.info(
            f"Treinamento concluído. Best iteration: {self.model.best_iteration}"
        )

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz previsões."""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda")

        return self.model.predict(X).astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Faz previsões de probabilidade (para classificação)."""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda")

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Modelo não suporta predict_proba")


class XGBoostClassifier(BaseModel):
    """Modelo XGBoost para classificação de direção."""

    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": 50,
            "eval_metric": "logloss",
        }

        if config:
            default_config.update(config)

        super().__init__("XGBoost_Classifier", default_config)

    def build_model(self, input_shape: Tuple[int, ...]) -> xgb.XGBClassifier:
        """Constrói o modelo XGBoost Classifier."""
        logger.info(
            f"Construindo modelo XGBoost Classifier com input_shape: {input_shape}"
        )

        self.model = xgb.XGBClassifier(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            learning_rate=self.config["learning_rate"],
            subsample=self.config["subsample"],
            colsample_bytree=self.config["colsample_bytree"],
            random_state=self.config["random_state"],
            eval_metric=self.config["eval_metric"],
            early_stopping_rounds=self.config["early_stopping_rounds"],
        )

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """Treina o modelo XGBoost Classifier."""
        logger.info(f"Treinando {self.model_name}")

        if self.model is None:
            self.build_model((X_train.shape[1],))

        # Preparar dados de validação
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        else:
            eval_set = [(X_train, y_train)]

        # Treinar modelo
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        self.is_trained = True

        # Calcular feature importance
        self.feature_importance = {
            "importance_gain": self.model.feature_importances_,
            "importance_weight": self.model.get_booster().get_score(
                importance_type="weight"
            ),
            "importance_cover": self.model.get_booster().get_score(
                importance_type="cover"
            ),
        }

        # Histórico de treinamento
        self.training_history = {
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score,
            "validation_scores": self.model.evals_result(),
        }

        logger.info(
            f"Treinamento concluído. Best iteration: {self.model.best_iteration}"
        )

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz previsões de classe."""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda")

        return self.model.predict(X).astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Faz previsões de probabilidade."""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda")

        return self.model.predict_proba(X)


def main():
    """Função principal para teste."""
    import sys

    sys.path.append("..")

    import pandas as pd

    from btc_perp_trader.features.feature_engineering import FeatureEngineer

    # Criar dados de teste
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 50000 + np.cumsum(np.random.randn(n) * 10),
            "high": 50000 + np.cumsum(np.random.randn(n) * 10) + 50,
            "low": 50000 + np.cumsum(np.random.randn(n) * 10) - 50,
            "close": 50000 + np.cumsum(np.random.randn(n) * 10),
            "volume": np.random.randint(100, 1000, n),
        }
    )

    # Criar features
    engineer = FeatureEngineer()
    features_df = engineer.process_full_pipeline(df)

    # Testar modelo de regressão
    model_reg = XGBoostModel()
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = (
        model_reg.prepare_data(features_df, "target_return_1")
    )

    # Treinar modelo
    model_reg.train(X_train, y_train, X_val, y_val)

    # Avaliar modelo
    metrics = model_reg.evaluate(X_test, y_test, "regression")
    print(f"Métricas de regressão: {metrics}")

    # Testar modelo de classificação
    model_clf = XGBoostClassifier()
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = (
        model_clf.prepare_data(features_df, "target_direction_1")
    )

    # Treinar modelo
    model_clf.train(X_train, y_train, X_val, y_val)

    # Avaliar modelo
    metrics = model_clf.evaluate(X_test, y_test, "classification")
    print(f"Métricas de classificação: {metrics}")


if __name__ == "__main__":
    main()

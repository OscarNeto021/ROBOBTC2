"""
Classe base para todos os modelos de machine learning.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Classe base abstrata para todos os modelos."""

    def __init__(self, model_name: str, config: Optional[Dict] = None):
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.feature_importance = None
        self.training_history = {}

    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...]) -> Any:
        """Constrói o modelo."""
        pass

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """Treina o modelo."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz previsões."""
        pass

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Tuple:
        """Prepara os dados para treinamento."""
        logger.info(f"Preparando dados para {self.model_name}")

        # Remover linhas com NaN no target
        df_clean = df.dropna(subset=[target_col])

        # Separar features e target
        feature_cols = [
            col
            for col in df_clean.columns
            if not col.startswith("target_") and col != "timestamp"
        ]
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values

        # Split inicial train/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )

        # Split train/validation
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=val_size_adjusted,
                random_state=42,
                shuffle=False,
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, None, y_temp, None

        msg = (
            f"Dados preparados - Train: {X_train.shape}, "
            f"Val: {X_val.shape if X_val is not None else None}, "
            f"Test: {X_test.shape}"
        )
        logger.info(msg)

        return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, task_type: str = "regression"
    ) -> Dict:
        """Avalia o modelo."""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda")

        predictions = self.predict(X_test)

        if task_type == "regression":
            metrics = {
                "mse": mean_squared_error(y_test, predictions),
                "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                "r2": r2_score(y_test, predictions),
                "mae": np.mean(np.abs(y_test - predictions)),
            }
        else:  # classification
            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "classification_report": classification_report(
                    y_test, predictions, output_dict=True
                ),
            }

        logger.info(f"Avaliação do {self.model_name}: {metrics}")
        return metrics

    def save_model(self, filepath: str):
        """Salva o modelo."""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda")

        model_data = {
            "model": self.model,
            "model_name": self.model_name,
            "config": self.config,
            "feature_importance": self.feature_importance,
            "training_history": self.training_history,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Modelo salvo em {filepath}")

    def load_model(self, filepath: str):
        """Carrega o modelo."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.model_name = model_data["model_name"]
        self.config = model_data["config"]
        self.feature_importance = model_data.get("feature_importance")
        self.training_history = model_data.get("training_history", {})
        self.is_trained = True

        logger.info(f"Modelo carregado de {filepath}")

    def get_feature_importance(self) -> Optional[Dict]:
        """Retorna a importância das features."""
        return self.feature_importance


class ModelEnsemble:
    """Classe para ensemble de modelos."""

    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.is_trained = False

        if len(self.weights) != len(self.models):
            raise ValueError("Número de pesos deve ser igual ao número de modelos")

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """Treina todos os modelos do ensemble."""
        logger.info(f"Treinando ensemble com {len(self.models)} modelos")

        training_results = {}

        for i, model in enumerate(self.models):
            logger.info(
                f"Treinando modelo {i+1}/{len(self.models)}: {model.model_name}"
            )

            # Construir modelo se necessário
            if model.model is None:
                input_shape = (X_train.shape[1],)
                model.build_model(input_shape)

            # Treinar modelo
            result = model.train(X_train, y_train, X_val, y_val)
            training_results[model.model_name] = result

        self.is_trained = True
        logger.info("Ensemble treinado com sucesso")

        return training_results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz previsões usando ensemble."""
        if not self.is_trained:
            raise ValueError("Ensemble não foi treinado ainda")

        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # Média ponderada das previsões
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)

        return ensemble_pred

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, task_type: str = "regression"
    ) -> Dict:
        """Avalia o ensemble."""
        predictions = self.predict(X_test)

        if task_type == "regression":
            metrics = {
                "ensemble_mse": mean_squared_error(y_test, predictions),
                "ensemble_rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                "ensemble_r2": r2_score(y_test, predictions),
                "ensemble_mae": np.mean(np.abs(y_test - predictions)),
            }
        else:  # classification
            metrics = {
                "ensemble_accuracy": accuracy_score(y_test, predictions),
                "ensemble_classification_report": classification_report(
                    y_test, predictions, output_dict=True
                ),
            }

        # Avaliar modelos individuais
        individual_metrics = {}
        for model in self.models:
            individual_metrics[model.model_name] = model.evaluate(
                X_test, y_test, task_type
            )

        metrics["individual_models"] = individual_metrics

        logger.info(f"Avaliação do ensemble: {metrics}")
        return metrics


def main():
    """Função principal para teste."""
    # Teste será implementado nas classes filhas
    pass


if __name__ == "__main__":
    main()

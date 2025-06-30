"""
Modelo LSTM para trading de BTC-PERP usando PyTorch.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class LSTMNet(nn.Module):
    """Rede neural LSTM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super(LSTMNet, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Camadas LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Camadas densas
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # Inicializar estados ocultos
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass através do LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Pegar apenas a última saída da sequência
        last_output = lstm_out[:, -1, :]

        # Camadas densas
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class LSTMModel(BaseModel):
    """Modelo LSTM para previsão de séries temporais."""

    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            "sequence_length": 60,  # Janela de tempo
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "patience": 10,
            "device": "cpu",
        }

        if config:
            default_config.update(config)

        super().__init__("LSTM", default_config)
        self.scaler = StandardScaler()
        self.device = torch.device(self.config["device"])

    def build_model(self, input_shape: Tuple[int, ...]) -> LSTMNet:
        """Constrói o modelo LSTM."""
        logger.info(f"Construindo modelo LSTM com input_shape: {input_shape}")

        input_size = input_shape[-1]  # Número de features

        self.model = LSTMNet(
            input_size=input_size,
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            output_size=1,
        ).to(self.device)

        return self.model

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cria sequências para o LSTM."""
        seq_length = self.config["sequence_length"]

        X_seq, y_seq = [], []

        for i in range(seq_length, len(X)):
            X_seq.append(X[i - seq_length : i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """Treina o modelo LSTM."""
        logger.info(f"Treinando {self.model_name}")

        # Normalizar dados
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)

        # Criar sequências
        X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train)

        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val)
        else:
            X_val_seq, y_val_seq = None, None

        # Construir modelo se necessário
        if self.model is None:
            input_shape = (self.config["sequence_length"], X_train.shape[1])
            self.build_model(input_shape)

        # Converter para tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).reshape(-1, 1).to(self.device)

        # DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["batch_size"], shuffle=True
        )

        # Validação
        if X_val_seq is not None:
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_seq).reshape(-1, 1).to(self.device)

        # Otimizador e loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        criterion = nn.MSELoss()

        # Treinamento
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config["epochs"]):
            # Treinamento
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validação
            if X_val_seq is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    val_losses.append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Salvar melhor modelo
                    torch.save(self.model.state_dict(), "best_lstm_model.pth")
                else:
                    patience_counter += 1

                if patience_counter >= self.config["patience"]:
                    logger.info(f"Early stopping na época {epoch}")
                    # Carregar melhor modelo
                    self.model.load_state_dict(torch.load("best_lstm_model.pth"))
                    break

            if epoch % 10 == 0:
                if X_val_seq is not None:
                    msg = (
                        f"Época {epoch}: Train Loss: {train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}"
                    )
                    logger.info(msg)
                else:
                    logger.info(f"Época {epoch}: Train Loss: {train_loss:.6f}")

        self.is_trained = True

        # Histórico de treinamento
        self.training_history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
        }

        logger.info("Treinamento LSTM concluído")

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz previsões."""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda")

        # Normalizar dados
        X_scaled = self.scaler.transform(X)

        # Criar sequências
        seq_length = self.config["sequence_length"]
        if len(X_scaled) < seq_length:
            raise ValueError(
                f"Dados insuficientes. Necessário pelo menos {seq_length} amostras"
            )

        X_seq = []
        for i in range(seq_length, len(X_scaled) + 1):
            X_seq.append(X_scaled[i - seq_length : i])

        X_seq = np.array(X_seq)

        # Converter para tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        # Previsão
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy().flatten()

        # pad to match input length
        if len(predictions) < len(X):
            pad_len = len(X) - len(predictions)
            pad = np.repeat(predictions[0], pad_len)
            predictions = np.concatenate([pad, predictions])

        return predictions


class LSTMClassifier(LSTMModel):
    """Modelo LSTM para classificação."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.model_name = "LSTM_Classifier"

    def build_model(self, input_shape: Tuple[int, ...]) -> LSTMNet:
        """Constrói o modelo LSTM para classificação."""
        logger.info(
            f"Construindo modelo LSTM Classifier com input_shape: {input_shape}"
        )

        input_size = input_shape[-1]

        # Para classificação binária, usar sigmoid na saída
        class LSTMClassifierNet(LSTMNet):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                out = super().forward(x)
                return self.sigmoid(out)

        self.model = LSTMClassifierNet(
            input_size=input_size,
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            output_size=1,
        ).to(self.device)

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """Treina o modelo LSTM para classificação."""
        # Usar BCELoss para classificação binária

        # Substituir temporariamente o critério
        def train_with_bce():
            # Normalizar dados
            X_train_scaled = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)

            # Criar sequências
            X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train)

            if X_val is not None and y_val is not None:
                X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val)
            else:
                X_val_seq, y_val_seq = None, None

            # Construir modelo se necessário
            if self.model is None:
                input_shape = (self.config["sequence_length"], X_train.shape[1])
                self.build_model(input_shape)

            # Converter para tensors
            X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
            y_train_tensor = (
                torch.FloatTensor(y_train_seq).reshape(-1, 1).to(self.device)
            )

            # DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(
                train_dataset, batch_size=self.config["batch_size"], shuffle=True
            )

            # Validação
            if X_val_seq is not None:
                X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
                y_val_tensor = (
                    torch.FloatTensor(y_val_seq).reshape(-1, 1).to(self.device)
                )

            # Otimizador e loss (BCE para classificação)
            optimizer = optim.Adam(
                self.model.parameters(), lr=self.config["learning_rate"]
            )
            criterion = nn.BCELoss()

            # Treinamento
            train_losses = []
            val_losses = []
            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(self.config["epochs"]):
                # Treinamento
                self.model.train()
                train_loss = 0.0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)
                train_losses.append(train_loss)

                # Validação
                if X_val_seq is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_outputs = self.model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor).item()
                        val_losses.append(val_loss)

                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save(self.model.state_dict(), "best_lstm_classifier.pth")
                    else:
                        patience_counter += 1

                    if patience_counter >= self.config["patience"]:
                        logger.info(f"Early stopping na época {epoch}")
                        self.model.load_state_dict(
                            torch.load("best_lstm_classifier.pth")
                        )
                        break

                if epoch % 10 == 0:
                    if X_val_seq is not None:
                        msg = (
                            f"Época {epoch}: Train Loss: {train_loss:.6f}, "
                            f"Val Loss: {val_loss:.6f}"
                        )
                        logger.info(msg)
                    else:
                        logger.info(f"Época {epoch}: Train Loss: {train_loss:.6f}")

            self.is_trained = True

            # Histórico de treinamento
            self.training_history = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
            }

            logger.info("Treinamento LSTM Classifier concluído")

            return self.training_history

        return train_with_bce()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz previsões de classe (0 ou 1)."""
        predictions = super().predict(X)
        return (predictions > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Faz previsões de probabilidade."""
        return super().predict(X)


def main():
    """Função principal para teste."""
    import sys

    sys.path.append("..")

    import pandas as pd

    from btc_perp_trader.features.feature_engineering import FeatureEngineer

    # Criar dados de teste
    np.random.seed(42)
    n = 500  # Menor para teste mais rápido
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

    # Testar modelo LSTM de regressão
    config = {"epochs": 20, "sequence_length": 30}  # Configuração rápida para teste
    model_reg = LSTMModel(config)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = (
        model_reg.prepare_data(features_df, "target_return_1")
    )

    # Treinar modelo
    model_reg.train(X_train, y_train, X_val, y_val)

    # Avaliar modelo
    metrics = model_reg.evaluate(X_test, y_test, "regression")
    print(f"Métricas LSTM regressão: {metrics}")


if __name__ == "__main__":
    main()

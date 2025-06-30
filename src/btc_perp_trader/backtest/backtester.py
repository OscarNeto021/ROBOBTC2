"""
Sistema de backtesting para estratégias de trading.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Position:
    """Representa uma posição de trading."""

    def __init__(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        entry_time: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        self.symbol = symbol
        self.side = side  # 'long' ou 'short'
        self.size = size
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.exit_price = None
        self.exit_time = None
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.pnl = 0.0
        self.is_open = True

    def close_position(self, exit_price: float, exit_time: datetime):
        """Fecha a posição."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.is_open = False

        # Calcular PnL
        if self.side == "long":
            self.pnl = (exit_price - self.entry_price) * self.size
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.size

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calcula PnL não realizado."""
        if not self.is_open:
            return self.pnl

        if self.side == "long":
            return (current_price - self.entry_price) * self.size
        else:  # short
            return (self.entry_price - current_price) * self.size


class TradingStrategy:
    """Classe base para estratégias de trading."""

    def __init__(self, name: str):
        self.name = name

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Gera sinais de trading. Deve ser implementado pelas subclasses."""
        raise NotImplementedError

    def should_close_position(
        self, position: Position, current_data: pd.Series
    ) -> bool:
        """Determina se uma posição deve ser fechada."""
        current_price = current_data["close"]

        # Verificar stop loss
        if position.stop_loss:
            if position.side == "long" and current_price <= position.stop_loss:
                return True
            elif position.side == "short" and current_price >= position.stop_loss:
                return True

        # Verificar take profit
        if position.take_profit:
            if position.side == "long" and current_price >= position.take_profit:
                return True
            elif position.side == "short" and current_price <= position.take_profit:
                return True

        return False


class MLTradingStrategy(TradingStrategy):
    """Estratégia baseada em modelo de machine learning."""

    def __init__(self, model, threshold: float = 0.001):
        super().__init__("ML_Strategy")
        self.model = model
        self.threshold = threshold  # Threshold para gerar sinais

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Gera sinais baseados nas previsões do modelo."""
        # Preparar features (remover colunas target e timestamp)
        feature_cols = [
            col
            for col in data.columns
            if not col.startswith("target_") and col != "timestamp"
        ]
        X = data[feature_cols].values

        # Fazer previsões
        predictions = self.model.predict(X)

        # Converter previsões em sinais
        signals = pd.Series(0, index=data.index)  # 0 = hold, 1 = buy, -1 = sell

        # Gerar sinais baseados no threshold
        signals[predictions > self.threshold] = 1  # Buy signal
        signals[predictions < -self.threshold] = -1  # Sell signal

        return signals


class Backtester:
    """Sistema de backtesting."""

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0001,
        max_position_size: float = 0.95,
    ):
        self.initial_capital = initial_capital
        self.commission = commission  # Taxa de comissão
        self.slippage = slippage  # Slippage
        self.max_position_size = max_position_size  # Máximo % do capital por posição

        # Estado do backtesting
        self.capital = initial_capital
        self.positions = []
        self.closed_positions = []
        self.equity_curve = []
        self.trades_log = []

    def reset(self):
        """Reseta o estado do backtester."""
        self.capital = self.initial_capital
        self.positions = []
        self.closed_positions = []
        self.equity_curve = []
        self.trades_log = []

    def run_backtest(self, data: pd.DataFrame, strategy: TradingStrategy) -> Dict:
        """Executa o backtesting."""
        logger.info(f"Iniciando backtesting da estratégia: {strategy.name}")

        self.reset()

        # Gerar sinais
        signals = strategy.generate_signals(data)

        # Iterar através dos dados
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row["close"]
            current_signal = signals.iloc[i] if i < len(signals) else 0

            # Verificar posições abertas para fechamento
            self._check_position_exits(row, strategy)

            # Processar novo sinal
            if current_signal != 0:
                self._process_signal(current_signal, row, timestamp)

            # Calcular equity atual
            current_equity = self._calculate_current_equity(current_price)
            self.equity_curve.append(
                {
                    "timestamp": timestamp,
                    "equity": current_equity,
                    "price": current_price,
                }
            )

        # Fechar todas as posições abertas no final
        final_price = data["close"].iloc[-1]
        final_timestamp = data.index[-1]
        for position in self.positions:
            position.close_position(final_price, final_timestamp)
            self.closed_positions.append(position)
        self.positions = []

        # Calcular métricas
        results = self._calculate_metrics()

        logger.info(
            f"Backtesting concluído. Total de trades: {len(self.closed_positions)}"
        )

        return results

    def _check_position_exits(self, current_data: pd.Series, strategy: TradingStrategy):
        """Verifica se alguma posição deve ser fechada."""
        positions_to_close = []

        for position in self.positions:
            if strategy.should_close_position(position, current_data):
                positions_to_close.append(position)

        # Fechar posições
        for position in positions_to_close:
            exit_price = current_data["close"] * (
                1 + self.slippage if position.side == "short" else 1 - self.slippage
            )
            position.close_position(exit_price, current_data.name)
            self.closed_positions.append(position)
            self.positions.remove(position)

            # Atualizar capital
            self.capital += position.pnl - abs(position.pnl * self.commission)

            # Log do trade
            self.trades_log.append(
                {
                    "entry_time": position.entry_time,
                    "exit_time": position.exit_time,
                    "side": position.side,
                    "entry_price": position.entry_price,
                    "exit_price": position.exit_price,
                    "size": position.size,
                    "pnl": position.pnl,
                    "commission": abs(position.pnl * self.commission),
                }
            )

    def _process_signal(self, signal: int, current_data: pd.Series, timestamp):
        """Processa um sinal de trading."""
        current_price = current_data["close"]

        # Determinar lado da posição
        side = "long" if signal > 0 else "short"

        # Calcular tamanho da posição
        available_capital = self.capital * self.max_position_size
        position_size = available_capital / current_price

        # Ajustar preço com slippage
        entry_price = current_price * (
            1 + self.slippage if side == "long" else 1 - self.slippage
        )

        # Criar posição
        position = Position(
            symbol="BTC-USDT",
            side=side,
            size=position_size,
            entry_price=entry_price,
            entry_time=timestamp,
        )

        self.positions.append(position)

        # Atualizar capital (reservar para a posição)
        self.capital -= available_capital

    def _calculate_current_equity(self, current_price: float) -> float:
        """Calcula o equity atual."""
        equity = self.capital

        # Adicionar PnL não realizado das posições abertas
        for position in self.positions:
            unrealized_pnl = position.get_unrealized_pnl(current_price)
            equity += unrealized_pnl

        return equity

    def _calculate_metrics(self) -> Dict:
        """Calcula métricas de performance."""
        if not self.equity_curve:
            return {}

        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index("timestamp", inplace=True)

        # Métricas básicas
        final_equity = equity_df["equity"].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # Calcular retornos diários
        daily_returns = equity_df["equity"].pct_change().dropna()

        # Sharpe ratio (assumindo 252 períodos de trading por ano)
        sharpe_ratio = (
            np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            if daily_returns.std() > 0
            else 0
        )

        # Maximum drawdown
        rolling_max = equity_df["equity"].expanding().max()
        drawdown = (equity_df["equity"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Métricas de trades
        if self.closed_positions:
            pnls = [pos.pnl for pos in self.closed_positions]
            winning_trades = [pnl for pnl in pnls if pnl > 0]
            losing_trades = [pnl for pnl in pnls if pnl < 0]

            win_rate = len(winning_trades) / len(pnls) if pnls else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            profit_factor = (
                abs(sum(winning_trades) / sum(losing_trades))
                if losing_trades
                else float("inf")
            )
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        metrics = {
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "total_trades": len(self.closed_positions),
            "winning_trades": len(
                [pos for pos in self.closed_positions if pos.pnl > 0]
            ),
            "losing_trades": len([pos for pos in self.closed_positions if pos.pnl < 0]),
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "equity_curve": equity_df,
            "trades_log": (
                pd.DataFrame(self.trades_log) if self.trades_log else pd.DataFrame()
            ),
        }

        return metrics

    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Plota os resultados do backtesting."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Equity curve
        equity_df = results["equity_curve"]
        axes[0, 0].plot(equity_df.index, equity_df["equity"], label="Equity")
        axes[0, 0].plot(
            equity_df.index,
            equity_df["price"]
            * (results["initial_capital"] / equity_df["price"].iloc[0]),
            label="Buy & Hold",
            alpha=0.7,
        )
        axes[0, 0].set_title("Equity Curve")
        axes[0, 0].set_ylabel("Equity ($)")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Drawdown
        rolling_max = equity_df["equity"].expanding().max()
        drawdown = (equity_df["equity"] - rolling_max) / rolling_max * 100
        axes[0, 1].fill_between(equity_df.index, drawdown, 0, alpha=0.3, color="red")
        axes[0, 1].set_title("Drawdown")
        axes[0, 1].set_ylabel("Drawdown (%)")
        axes[0, 1].grid(True)

        # Distribuição de retornos
        if not equity_df.empty:
            daily_returns = equity_df["equity"].pct_change().dropna() * 100
            axes[1, 0].hist(daily_returns, bins=50, alpha=0.7, edgecolor="black")
            axes[1, 0].set_title("Distribution of Returns")
            axes[1, 0].set_xlabel("Daily Return (%)")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].grid(True)

        # Métricas resumo
        metrics_text = f"""
        Total Return: {results['total_return_pct']:.2f}%
        Sharpe Ratio: {results['sharpe_ratio']:.2f}
        Max Drawdown: {results['max_drawdown_pct']:.2f}%
        Total Trades: {results['total_trades']}
        Win Rate: {results['win_rate_pct']:.2f}%
        Profit Factor: {results['profit_factor']:.2f}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment="center")
        axes[1, 1].set_title("Performance Metrics")
        axes[1, 1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Gráficos salvos em {save_path}")

        plt.show()


def main():
    """Função principal para teste."""
    import sys

    sys.path.append("..")

    from btc_perp_trader.features.feature_engineering import FeatureEngineer
    from btc_perp_trader.models.xgboost_model import XGBoostModel

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

    # Treinar modelo
    model = XGBoostModel({"n_estimators": 100})
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = model.prepare_data(
        features_df, "target_return_1"
    )
    model.train(X_train, y_train, X_val, y_val)

    # Criar estratégia
    strategy = MLTradingStrategy(model, threshold=0.001)

    # Executar backtesting
    backtester = Backtester(initial_capital=100000)
    results = backtester.run_backtest(features_df, strategy)

    # Mostrar resultados
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {results['win_rate_pct']:.2f}%")


if __name__ == "__main__":
    main()

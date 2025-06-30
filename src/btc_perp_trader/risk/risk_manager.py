"""
Sistema de gestão de risco para trading algorítmico.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskMetrics:
    """Classe para calcular métricas de risco."""

    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calcula Value at Risk (VaR)."""
        if len(returns) == 0:
            return 0.0

        return np.percentile(returns, confidence_level * 100)

    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calcula Conditional Value at Risk (CVaR)."""
        if len(returns) == 0:
            return 0.0

        var = RiskMetrics.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()

    @staticmethod
    def calculate_sharpe_ratio(
        returns: np.ndarray, risk_free_rate: float = 0.0
    ) -> float:
        """Calcula Sharpe Ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns)

    @staticmethod
    def calculate_sortino_ratio(
        returns: np.ndarray, risk_free_rate: float = 0.0
    ) -> float:
        """Calcula Sortino Ratio."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float("inf") if np.mean(excess_returns) > 0 else 0.0

        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0

        return np.mean(excess_returns) / downside_deviation

    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
        """Calcula Maximum Drawdown."""
        if len(equity_curve) == 0:
            return 0.0, 0, 0

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak

        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)

        # Encontrar o pico antes do máximo drawdown
        peak_idx = np.argmax(peak[: max_dd_idx + 1])

        return max_dd, peak_idx, max_dd_idx

    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray) -> float:
        """Calcula Calmar Ratio."""
        if len(returns) == 0 or len(equity_curve) == 0:
            return 0.0

        annual_return = np.mean(returns) * 252  # Assumindo retornos diários
        max_dd, _, _ = RiskMetrics.calculate_max_drawdown(equity_curve)

        if max_dd == 0:
            return float("inf") if annual_return > 0 else 0.0

        return annual_return / abs(max_dd)


class PositionSizer:
    """Classe para dimensionamento de posições."""

    def __init__(self, method: str = "fixed_fraction", **kwargs):
        self.method = method
        self.params = kwargs

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        volatility: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> float:
        """Calcula o tamanho da posição."""

        if self.method == "fixed_fraction":
            fraction = self.params.get("fraction", 0.02)  # 2% do capital por padrão
            return (capital * fraction) / price

        elif self.method == "kelly_criterion":
            if confidence is None:
                raise ValueError("Confidence é necessário para Kelly Criterion")

            win_rate = self.params.get("win_rate", 0.55)
            avg_win = self.params.get("avg_win", 0.02)
            avg_loss = self.params.get("avg_loss", 0.01)

            if avg_loss == 0:
                return 0.0

            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Limitar a 25%

            return (capital * kelly_fraction) / price

        elif self.method == "volatility_targeting":
            if volatility is None:
                raise ValueError("Volatility é necessário para Volatility Targeting")

            target_vol = self.params.get(
                "target_volatility", 0.15
            )  # 15% de volatilidade alvo

            if volatility == 0:
                return 0.0

            vol_scalar = target_vol / volatility
            base_fraction = self.params.get("base_fraction", 0.1)

            position_fraction = base_fraction * vol_scalar
            position_fraction = max(
                0.01, min(position_fraction, 0.5)
            )  # Limitar entre 1% e 50%

            return (capital * position_fraction) / price

        else:
            raise ValueError(f"Método de dimensionamento desconhecido: {self.method}")


class RiskManager:
    """Gerenciador de risco principal."""

    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            "max_position_size": 0.1,  # 10% do capital máximo por posição
            "max_portfolio_risk": 0.2,  # 20% do capital máximo em risco
            "max_daily_loss": 0.05,  # 5% perda máxima diária
            "max_drawdown": 0.15,  # 15% drawdown máximo
            "var_confidence": 0.05,  # 5% VaR
            "position_sizing_method": "fixed_fraction",
            "stop_loss_pct": 0.02,  # 2% stop loss
            "take_profit_pct": 0.04,  # 4% take profit
            "risk_free_rate": 0.02,  # 2% taxa livre de risco anual
        }

        self.config = {**default_config, **(config or {})}

        # Inicializar position sizer
        self.position_sizer = PositionSizer(
            method=self.config["position_sizing_method"],
            fraction=self.config["max_position_size"],
        )

        # Histórico de métricas
        self.risk_history = []

    def check_risk_limits(
        self,
        current_capital: float,
        initial_capital: float,
        open_positions: List,
        daily_pnl: float,
    ) -> Dict[str, bool]:
        """Verifica se os limites de risco estão sendo respeitados."""

        checks = {
            "max_daily_loss": True,
            "max_drawdown": True,
            "max_portfolio_risk": True,
            "position_limits": True,
        }

        # Verificar perda diária máxima
        daily_loss_pct = daily_pnl / initial_capital
        if daily_loss_pct < -self.config["max_daily_loss"]:
            checks["max_daily_loss"] = False
            logger.warning(f"Limite de perda diária excedido: {daily_loss_pct:.2%}")

        # Verificar drawdown máximo
        current_drawdown = (initial_capital - current_capital) / initial_capital
        if current_drawdown > self.config["max_drawdown"]:
            checks["max_drawdown"] = False
            logger.warning(f"Drawdown máximo excedido: {current_drawdown:.2%}")

        # Verificar risco total do portfólio
        total_risk = sum(
            [abs(pos.get("size", 0) * pos.get("price", 0)) for pos in open_positions]
        )
        portfolio_risk_pct = total_risk / current_capital
        if portfolio_risk_pct > self.config["max_portfolio_risk"]:
            checks["max_portfolio_risk"] = False
            logger.warning(f"Risco do portfólio excedido: {portfolio_risk_pct:.2%}")

        # Verificar limites de posição individual
        for pos in open_positions:
            position_size_pct = (
                abs(pos.get("size", 0) * pos.get("price", 0)) / current_capital
            )
            if position_size_pct > self.config["max_position_size"]:
                checks["position_limits"] = False
                logger.warning(f"Tamanho de posição excedido: {position_size_pct:.2%}")
                break

        return checks

    def calculate_position_size(
        self,
        signal_strength: float,
        current_capital: float,
        current_price: float,
        volatility: Optional[float] = None,
    ) -> float:
        """Calcula o tamanho da posição baseado no sinal e risco."""

        # Ajustar tamanho baseado na força do sinal
        base_size = self.position_sizer.calculate_position_size(
            current_capital, current_price, volatility
        )

        # Escalar pelo strength do sinal (0-1)
        adjusted_size = base_size * abs(signal_strength)

        # Aplicar limites máximos
        max_size = (current_capital * self.config["max_position_size"]) / current_price
        adjusted_size = min(adjusted_size, max_size)

        return adjusted_size

    def calculate_stop_loss_take_profit(
        self, entry_price: float, side: str
    ) -> Tuple[float, float]:
        """Calcula níveis de stop loss e take profit."""

        stop_loss_pct = self.config["stop_loss_pct"]
        take_profit_pct = self.config["take_profit_pct"]

        if side == "long":
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:  # short
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)

        return stop_loss, take_profit

    def calculate_portfolio_metrics(
        self, equity_curve: List[float], returns: List[float]
    ) -> Dict:
        """Calcula métricas de risco do portfólio."""

        if not equity_curve or not returns:
            return {}

        equity_array = np.array(equity_curve)
        returns_array = np.array(returns)

        metrics = {
            "var_5pct": RiskMetrics.calculate_var(returns_array, 0.05),
            "cvar_5pct": RiskMetrics.calculate_cvar(returns_array, 0.05),
            "sharpe_ratio": RiskMetrics.calculate_sharpe_ratio(
                returns_array, self.config["risk_free_rate"] / 252
            ),
            "sortino_ratio": RiskMetrics.calculate_sortino_ratio(
                returns_array, self.config["risk_free_rate"] / 252
            ),
            "max_drawdown": RiskMetrics.calculate_max_drawdown(equity_array)[0],
            "calmar_ratio": RiskMetrics.calculate_calmar_ratio(
                returns_array, equity_array
            ),
            "volatility": np.std(returns_array)
            * np.sqrt(252),  # Volatilidade anualizada
            "skewness": self._calculate_skewness(returns_array),
            "kurtosis": self._calculate_kurtosis(returns_array),
        }

        # Adicionar ao histórico
        self.risk_history.append({"timestamp": pd.Timestamp.now(), **metrics})

        return metrics

    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calcula skewness dos retornos."""
        if len(returns) < 3:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
        return skewness

    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calcula kurtosis dos retornos."""
        if len(returns) < 4:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
        return kurtosis

    def should_halt_trading(
        self,
        current_capital: float,
        initial_capital: float,
        daily_pnl: float,
        open_positions: List,
    ) -> bool:
        """Determina se o trading deve ser interrompido."""

        risk_checks = self.check_risk_limits(
            current_capital, initial_capital, open_positions, daily_pnl
        )

        # Parar trading se qualquer limite crítico for violado
        critical_violations = [
            not risk_checks["max_daily_loss"],
            not risk_checks["max_drawdown"],
        ]

        if any(critical_violations):
            logger.critical(
                "Trading interrompido devido a violação de limites de risco críticos"
            )
            return True

        return False

    def get_risk_report(self) -> Dict:
        """Gera relatório de risco."""

        if not self.risk_history:
            return {"message": "Nenhum dado de risco disponível"}

        latest_metrics = self.risk_history[-1]

        # Calcular tendências
        if len(self.risk_history) > 1:
            prev_metrics = self.risk_history[-2]
            trends = {
                "sharpe_trend": latest_metrics["sharpe_ratio"]
                - prev_metrics["sharpe_ratio"],
                "volatility_trend": latest_metrics["volatility"]
                - prev_metrics["volatility"],
                "drawdown_trend": latest_metrics["max_drawdown"]
                - prev_metrics["max_drawdown"],
            }
        else:
            trends = {"sharpe_trend": 0, "volatility_trend": 0, "drawdown_trend": 0}

        report = {
            "current_metrics": latest_metrics,
            "trends": trends,
            "risk_limits": self.config,
            "recommendations": self._generate_risk_recommendations(latest_metrics),
        }

        return report

    def _generate_risk_recommendations(self, metrics: Dict) -> List[str]:
        """Gera recomendações baseadas nas métricas de risco."""

        recommendations = []

        # Sharpe ratio
        if metrics["sharpe_ratio"] < 0.5:
            recommendations.append(
                "Sharpe ratio baixo - considere revisar a estratégia"
            )

        # Volatilidade
        if metrics["volatility"] > 0.3:
            recommendations.append(
                "Alta volatilidade - considere reduzir tamanho das posições"
            )

        # Drawdown
        if abs(metrics["max_drawdown"]) > 0.1:
            recommendations.append(
                "Drawdown elevado - considere implementar stop loss mais rigoroso"
            )

        # Skewness
        if metrics["skewness"] < -1:
            recommendations.append(
                "Assimetria negativa alta - cuidado com risco de cauda"
            )

        # Kurtosis
        if metrics["kurtosis"] > 3:
            recommendations.append("Curtose alta - eventos extremos mais prováveis")

        if not recommendations:
            recommendations.append("Métricas de risco dentro dos parâmetros aceitáveis")

        return recommendations


def main():
    """Função principal para teste."""
    # Criar dados de teste
    np.random.seed(42)

    # Simular equity curve
    returns = np.random.normal(0.001, 0.02, 252)  # Retornos diários
    equity_curve = np.cumprod(1 + returns) * 100000  # Começar com $100k

    # Criar risk manager
    risk_manager = RiskManager()

    # Calcular métricas
    metrics = risk_manager.calculate_portfolio_metrics(
        equity_curve.tolist(), returns.tolist()
    )

    print("Métricas de Risco:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Gerar relatório
    report = risk_manager.get_risk_report()
    print("\nRecomendações:")
    for rec in report["recommendations"]:
        print(f"- {rec}")


if __name__ == "__main__":
    main()

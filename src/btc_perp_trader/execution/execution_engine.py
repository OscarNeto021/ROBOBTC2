"""
Motor de execução stub para trading algorítmico.
Simula execução de ordens sem conectar a exchanges reais.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import ccxt
import numpy as np
import pandas as pd
import yaml

from btc_perp_trader.config import BINANCE_API_KEY, BINANCE_API_SECRET

cfg = yaml.safe_load(open("config/config.yaml"))
api_key = cfg["exchange"].get("api_key") or BINANCE_API_KEY
api_secret = cfg["exchange"].get("api_secret") or BINANCE_API_SECRET
testnet = cfg["exchange"].get("testnet", False)

exchange = ccxt.binance(
    {
        "apiKey": api_key,
        "secret": api_secret,
        "options": {"defaultType": "future"},
        "enableRateLimit": True,
    }
)
exchange.set_sandbox_mode(testnet)

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Tipos de ordem."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Lado da ordem."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Status da ordem."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Representa uma ordem de trading."""

    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    timestamp: datetime = None
    fill_timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Trade:
    """Representa uma execução de trade."""

    id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0


@dataclass
class Position:
    """Representa uma posição atual."""

    symbol: str
    quantity: float  # Positivo = long, Negativo = short
    average_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class ExecutionEngine:
    """Motor de execução stub."""

    def __init__(self, initial_balance: float = 100000, commission_rate: float = 0.001):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission_rate = commission_rate

        # Estado do sistema
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self.order_history: List[Order] = []

        # Simulação de mercado
        self.current_prices: Dict[str, float] = {}
        self.market_data: Dict[str, pd.DataFrame] = {}

        # Métricas
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        logger.info(
            f"ExecutionEngine inicializado com balance: ${initial_balance:,.2f}"
        )

    def set_market_data(self, symbol: str, data: pd.DataFrame):
        """Define dados de mercado para simulação."""
        self.market_data[symbol] = data
        if not data.empty:
            self.current_prices[symbol] = data["close"].iloc[-1]
        logger.info(f"Dados de mercado definidos para {symbol}")

    def update_market_price(self, symbol: str, price: float):
        """Atualiza preço atual do mercado."""
        self.current_prices[symbol] = price
        self._check_pending_orders(symbol)
        self._update_positions_pnl(symbol)

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> str:
        """Coloca uma ordem."""

        # Validações básicas
        if quantity <= 0:
            raise ValueError("Quantidade deve ser positiva")

        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
            raise ValueError("Preço é obrigatório para ordens limit")

        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            raise ValueError("Stop price é obrigatório para ordens stop")

        # Verificar se há saldo suficiente
        if not self._check_sufficient_balance(symbol, side, quantity, price):
            raise ValueError("Saldo insuficiente")

        # Criar ordem
        order_id = str(uuid.uuid4())
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
        )

        self.orders[order_id] = order

        # Tentar executar imediatamente se for market order
        if order_type == OrderType.MARKET:
            self._execute_market_order(order)

        logger.info(
            f"Ordem colocada: {order_id} - {side.value} {quantity} {symbol} @ {price}"
        )

        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancela uma ordem."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False

        order.status = OrderStatus.CANCELLED
        self.order_history.append(order)
        del self.orders[order_id]

        logger.info(f"Ordem cancelada: {order_id}")
        return True

    def get_position(self, symbol: str) -> Optional[Position]:
        """Retorna posição atual para um símbolo."""
        return self.positions.get(symbol)

    def get_balance(self) -> float:
        """Retorna saldo atual."""
        return self.balance

    def get_total_equity(self) -> float:
        """Retorna equity total (saldo + PnL não realizado)."""
        total_unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )
        return self.balance + total_unrealized_pnl

    def get_open_orders(self) -> List[Order]:
        """Retorna ordens abertas."""
        return [
            order
            for order in self.orders.values()
            if order.status == OrderStatus.PENDING
        ]

    def get_order_history(self) -> List[Order]:
        """Retorna histórico de ordens."""
        return self.order_history + list(self.orders.values())

    def get_trade_history(self) -> List[Trade]:
        """Retorna histórico de trades."""
        return self.trades

    def get_performance_metrics(self) -> Dict:
        """Retorna métricas de performance."""
        if self.total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return": 0.0,
            }

        total_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        total_return = (
            self.get_total_equity() - self.initial_balance
        ) / self.initial_balance

        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": (
                self.winning_trades / self.total_trades
                if self.total_trades > 0
                else 0.0
            ),
            "total_pnl": total_pnl,
            "total_return": total_return,
            "current_balance": self.balance,
            "total_equity": self.get_total_equity(),
        }

    def _check_sufficient_balance(
        self, symbol: str, side: OrderSide, quantity: float, price: Optional[float]
    ) -> bool:
        """Verifica se há saldo suficiente para a ordem."""

        if side == OrderSide.BUY:
            # Para compra, precisa de saldo em dinheiro
            required_amount = quantity * (price or self.current_prices.get(symbol, 0))
            return self.balance >= required_amount
        else:
            # Para venda, precisa ter a posição
            position = self.positions.get(symbol)
            if position is None:
                return False
            return position.quantity >= quantity

    def _execute_market_order(self, order: Order):
        """Executa uma ordem de mercado."""
        current_price = self.current_prices.get(order.symbol)

        if current_price is None:
            order.status = OrderStatus.REJECTED
            logger.warning(
                f"Ordem rejeitada - preço não disponível para {order.symbol}"
            )
            return

        # Simular slippage
        slippage = np.random.normal(0, 0.0001)  # 0.01% de slippage médio
        execution_price = current_price * (1 + slippage)

        self._fill_order(order, execution_price, order.quantity)

    def _check_pending_orders(self, symbol: str):
        """Verifica ordens pendentes que podem ser executadas."""
        current_price = self.current_prices.get(symbol)
        if current_price is None:
            return

        orders_to_execute = []

        for order in self.orders.values():
            if order.symbol != symbol or order.status != OrderStatus.PENDING:
                continue

            should_execute = False

            if order.type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and current_price <= order.price:
                    should_execute = True
                elif order.side == OrderSide.SELL and current_price >= order.price:
                    should_execute = True

            elif order.type == OrderType.STOP:
                if order.side == OrderSide.BUY and current_price >= order.stop_price:
                    should_execute = True
                elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                    should_execute = True

            if should_execute:
                orders_to_execute.append(order)

        # Executar ordens
        for order in orders_to_execute:
            execution_price = (
                order.price if order.type == OrderType.LIMIT else current_price
            )
            self._fill_order(order, execution_price, order.quantity)

    def _fill_order(self, order: Order, price: float, quantity: float):
        """Executa o preenchimento de uma ordem."""

        # Calcular comissão
        commission = quantity * price * self.commission_rate

        # Criar trade
        trade = Trade(
            id=str(uuid.uuid4()),
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
            commission=commission,
        )

        self.trades.append(trade)

        # Atualizar ordem
        order.filled_quantity += quantity
        order.average_fill_price = price
        order.status = OrderStatus.FILLED
        order.fill_timestamp = datetime.now()

        # Atualizar posição
        self._update_position(order.symbol, order.side, quantity, price)

        # Atualizar saldo
        if order.side == OrderSide.BUY:
            self.balance -= quantity * price + commission
        else:
            self.balance += quantity * price - commission

        # Mover ordem para histórico
        self.order_history.append(order)
        del self.orders[order.id]

        # Atualizar estatísticas
        self.total_trades += 1

        logger.info(f"Ordem executada: {order.id} - {quantity} @ {price}")

    def _update_position(
        self, symbol: str, side: OrderSide, quantity: float, price: float
    ):
        """Atualiza posição."""

        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol, quantity=0, average_price=0
            )

        position = self.positions[symbol]

        if side == OrderSide.BUY:
            # Aumentar posição long ou reduzir posição short
            if position.quantity >= 0:
                # Aumentar long
                total_cost = (
                    position.quantity * position.average_price + quantity * price
                )
                position.quantity += quantity
                position.average_price = (
                    total_cost / position.quantity if position.quantity > 0 else 0
                )
            else:
                # Reduzir short
                if quantity >= abs(position.quantity):
                    # Fechar short e abrir long
                    realized_pnl = abs(position.quantity) * (
                        position.average_price - price
                    )
                    position.realized_pnl += realized_pnl
                    remaining_quantity = quantity - abs(position.quantity)
                    position.quantity = remaining_quantity
                    position.average_price = price if remaining_quantity > 0 else 0

                    if realized_pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                else:
                    # Apenas reduzir short
                    realized_pnl = quantity * (position.average_price - price)
                    position.realized_pnl += realized_pnl
                    position.quantity += (
                        quantity  # quantity é positivo, position.quantity é negativo
                    )

        else:  # SELL
            # Reduzir posição long ou aumentar posição short
            if position.quantity > 0:
                # Reduzir long
                if quantity >= position.quantity:
                    # Fechar long e abrir short
                    realized_pnl = position.quantity * (price - position.average_price)
                    position.realized_pnl += realized_pnl
                    remaining_quantity = quantity - position.quantity
                    position.quantity = -remaining_quantity
                    position.average_price = price if remaining_quantity > 0 else 0

                    if realized_pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                else:
                    # Apenas reduzir long
                    realized_pnl = quantity * (price - position.average_price)
                    position.realized_pnl += realized_pnl
                    position.quantity -= quantity
            else:
                # Aumentar short
                total_cost = (
                    abs(position.quantity) * position.average_price + quantity * price
                )
                position.quantity -= quantity
                position.average_price = (
                    total_cost / abs(position.quantity) if position.quantity != 0 else 0
                )

        # Remover posição se quantidade for zero
        if abs(position.quantity) < 1e-8:
            del self.positions[symbol]

    def _update_positions_pnl(self, symbol: str):
        """Atualiza PnL não realizado das posições."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        current_price = self.current_prices[symbol]

        if position.quantity > 0:
            # Posição long
            position.unrealized_pnl = position.quantity * (
                current_price - position.average_price
            )
        elif position.quantity < 0:
            # Posição short
            position.unrealized_pnl = abs(position.quantity) * (
                position.average_price - current_price
            )
        else:
            position.unrealized_pnl = 0


class TradingBot:
    """Bot de trading que usa o ExecutionEngine."""

    def __init__(self, execution_engine: ExecutionEngine, strategy, risk_manager):
        self.execution_engine = execution_engine
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.is_running = False

    def start(self):
        """Inicia o bot de trading."""
        self.is_running = True
        logger.info("Trading bot iniciado")

    def stop(self):
        """Para o bot de trading."""
        self.is_running = False
        logger.info("Trading bot parado")

    def process_market_data(self, symbol: str, data: pd.Series):
        """Processa novos dados de mercado."""
        if not self.is_running:
            return

        # Atualizar preço no execution engine
        current_price = data["close"]
        self.execution_engine.update_market_price(symbol, current_price)

        # Gerar sinal da estratégia
        signal = self._generate_signal(data)

        if signal != 0:
            # Verificar limites de risco
            current_equity = self.execution_engine.get_total_equity()
            if not self.risk_manager.should_halt_trading(
                current_equity, self.execution_engine.initial_balance, 0, []
            ):
                self._execute_signal(symbol, signal, current_price)

    def _generate_signal(self, data: pd.Series) -> float:
        """Gera sinal de trading (-1 a 1)."""
        # Implementação simplificada - usar estratégia
        if hasattr(self.strategy, "predict"):
            # Preparar dados para o modelo
            feature_cols = [
                col
                for col in data.index
                if not col.startswith("target_") and col != "timestamp"
            ]
            X = data[feature_cols].values.reshape(1, -1)
            prediction = self.strategy.predict(X)[0]

            # Converter previsão em sinal
            if prediction > 0.001:
                return 1.0
            elif prediction < -0.001:
                return -1.0
            else:
                return 0.0

        return 0.0

    def _execute_signal(self, symbol: str, signal: float, current_price: float):
        """Executa um sinal de trading."""

        # Calcular tamanho da posição
        current_equity = self.execution_engine.get_total_equity()
        position_size = self.risk_manager.calculate_position_size(
            abs(signal), current_equity, current_price
        )

        if position_size > 0:
            side = OrderSide.BUY if signal > 0 else OrderSide.SELL

            try:
                self.execution_engine.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=position_size,
                )
                logger.info(f"Sinal executado: {side.value} {position_size} {symbol}")
            except Exception as e:
                logger.error(f"Erro ao executar sinal: {e}")


def main():
    """Função principal para teste."""
    import sys

    sys.path.append("..")

    from btc_perp_trader.features.feature_engineering import FeatureEngineer
    from btc_perp_trader.models.xgboost_model import XGBoostModel
    from btc_perp_trader.risk.risk_manager import RiskManager

    # Criar dados de teste
    np.random.seed(42)
    n = 100
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

    # Treinar modelo simples
    model = XGBoostModel({"n_estimators": 50})
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = model.prepare_data(
        features_df, "target_return_1"
    )
    model.train(X_train, y_train, X_val, y_val)

    # Criar execution engine
    engine = ExecutionEngine(initial_balance=100000)
    engine.set_market_data("BTC-USDT", features_df)

    # Criar risk manager
    risk_manager = RiskManager()

    # Criar bot
    bot = TradingBot(engine, model, risk_manager)
    bot.start()

    # Simular alguns trades
    for i in range(10):
        if i < len(features_df):
            bot.process_market_data("BTC-USDT", features_df.iloc[i])

    # Mostrar resultados
    metrics = engine.get_performance_metrics()
    print(f"Performance: {metrics}")


if __name__ == "__main__":
    main()

"""
Rotas da API para o sistema de trading.
"""

import logging
import os
import sys
from datetime import datetime

from flask import Blueprint, jsonify, request
from flask_cors import cross_origin

# Adicionar o diretório pai ao path para importar módulos do projeto principal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    from src.execution.execution_engine import (
        ExecutionEngine,
        OrderSide,
        OrderType,
        TradingBot,
    )
    from src.features.feature_engineering import FeatureEngineer
    from src.risk.risk_manager import RiskManager
except ImportError as e:
    logging.error(f"Erro ao importar módulos: {e}")
    # Fallback para desenvolvimento
    ExecutionEngine = None
    TradingBot = None
    RiskManager = None

logger = logging.getLogger(__name__)

trading_bp = Blueprint("trading", __name__)

# Estado global da aplicação
app_state = {
    "execution_engine": None,
    "trading_bot": None,
    "risk_manager": None,
    "model": None,
    "feature_engineer": None,
    "is_initialized": False,
}


@trading_bp.route("/status", methods=["GET"])
@cross_origin()
def get_status():
    """Retorna o status do sistema de trading."""
    try:
        if not app_state["is_initialized"]:
            return (
                jsonify(
                    {"status": "not_initialized", "message": "Sistema não inicializado"}
                ),
                200,
            )

        engine = app_state["execution_engine"]
        bot = app_state["trading_bot"]

        status = {
            "status": "running" if bot and bot.is_running else "stopped",
            "balance": engine.get_balance() if engine else 0,
            "total_equity": engine.get_total_equity() if engine else 0,
            "open_orders": len(engine.get_open_orders()) if engine else 0,
            "total_trades": engine.total_trades if engine else 0,
            "positions": len(engine.positions) if engine else 0,
            "timestamp": datetime.now().isoformat(),
        }

        return jsonify(status), 200

    except Exception as e:
        logger.error(f"Erro ao obter status: {e}")
        return jsonify({"error": str(e)}), 500


@trading_bp.route("/initialize", methods=["POST"])
@cross_origin()
def initialize_system():
    """Inicializa o sistema de trading."""
    try:
        data = request.get_json() or {}

        # Configurações padrão
        initial_balance = data.get("initial_balance", 100000)
        commission_rate = data.get("commission_rate", 0.001)

        # Inicializar componentes
        if ExecutionEngine is None:
            return jsonify({"error": "Módulos não disponíveis"}), 500

        app_state["execution_engine"] = ExecutionEngine(
            initial_balance=initial_balance, commission_rate=commission_rate
        )

        app_state["risk_manager"] = RiskManager()
        app_state["feature_engineer"] = FeatureEngineer()

        # Modelo será carregado quando necessário
        app_state["model"] = None

        app_state["is_initialized"] = True

        logger.info("Sistema de trading inicializado")

        return (
            jsonify(
                {
                    "message": "Sistema inicializado com sucesso",
                    "initial_balance": initial_balance,
                    "commission_rate": commission_rate,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Erro ao inicializar sistema: {e}")
        return jsonify({"error": str(e)}), 500


@trading_bp.route("/start", methods=["POST"])
@cross_origin()
def start_trading():
    """Inicia o bot de trading."""
    try:
        if not app_state["is_initialized"]:
            return jsonify({"error": "Sistema não inicializado"}), 400

        if app_state["trading_bot"] and app_state["trading_bot"].is_running:
            return jsonify({"message": "Bot já está rodando"}), 200

        # Criar bot se não existir
        if app_state["trading_bot"] is None:
            # Usar modelo mock se não houver modelo treinado
            mock_model = type(
                "MockModel", (), {"predict": lambda self, X: [0.0] * len(X)}
            )()

            app_state["trading_bot"] = TradingBot(
                execution_engine=app_state["execution_engine"],
                strategy=mock_model,
                risk_manager=app_state["risk_manager"],
            )

        app_state["trading_bot"].start()

        logger.info("Bot de trading iniciado")

        return jsonify({"message": "Bot de trading iniciado"}), 200

    except Exception as e:
        logger.error(f"Erro ao iniciar trading: {e}")
        return jsonify({"error": str(e)}), 500


@trading_bp.route("/stop", methods=["POST"])
@cross_origin()
def stop_trading():
    """Para o bot de trading."""
    try:
        if app_state["trading_bot"]:
            app_state["trading_bot"].stop()
            logger.info("Bot de trading parado")
            return jsonify({"message": "Bot de trading parado"}), 200
        else:
            return jsonify({"message": "Bot não estava rodando"}), 200

    except Exception as e:
        logger.error(f"Erro ao parar trading: {e}")
        return jsonify({"error": str(e)}), 500


@trading_bp.route("/orders", methods=["GET"])
@cross_origin()
def get_orders():
    """Retorna ordens abertas e histórico."""
    try:
        if not app_state["is_initialized"]:
            return jsonify({"error": "Sistema não inicializado"}), 400

        engine = app_state["execution_engine"]

        open_orders = []
        for order in engine.get_open_orders():
            open_orders.append(
                {
                    "id": order.id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "type": order.type.value,
                    "quantity": order.quantity,
                    "price": order.price,
                    "status": order.status.value,
                    "timestamp": order.timestamp.isoformat(),
                }
            )

        order_history = []
        for order in engine.get_order_history()[-10:]:  # Últimas 10 ordens
            order_history.append(
                {
                    "id": order.id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "type": order.type.value,
                    "quantity": order.quantity,
                    "price": order.price,
                    "status": order.status.value,
                    "timestamp": order.timestamp.isoformat(),
                    "fill_timestamp": (
                        order.fill_timestamp.isoformat()
                        if order.fill_timestamp
                        else None
                    ),
                }
            )

        return (
            jsonify({"open_orders": open_orders, "order_history": order_history}),
            200,
        )

    except Exception as e:
        logger.error(f"Erro ao obter ordens: {e}")
        return jsonify({"error": str(e)}), 500


@trading_bp.route("/positions", methods=["GET"])
@cross_origin()
def get_positions():
    """Retorna posições atuais."""
    try:
        if not app_state["is_initialized"]:
            return jsonify({"error": "Sistema não inicializado"}), 400

        engine = app_state["execution_engine"]

        positions = []
        for symbol, position in engine.positions.items():
            positions.append(
                {
                    "symbol": symbol,
                    "quantity": position.quantity,
                    "average_price": position.average_price,
                    "unrealized_pnl": position.unrealized_pnl,
                    "realized_pnl": position.realized_pnl,
                    "side": "long" if position.quantity > 0 else "short",
                }
            )

        return jsonify({"positions": positions}), 200

    except Exception as e:
        logger.error(f"Erro ao obter posições: {e}")
        return jsonify({"error": str(e)}), 500


@trading_bp.route("/performance", methods=["GET"])
@cross_origin()
def get_performance():
    """Retorna métricas de performance."""
    try:
        if not app_state["is_initialized"]:
            return jsonify({"error": "Sistema não inicializado"}), 400

        engine = app_state["execution_engine"]
        metrics = engine.get_performance_metrics()

        return jsonify(metrics), 200

    except Exception as e:
        logger.error(f"Erro ao obter performance: {e}")
        return jsonify({"error": str(e)}), 500


@trading_bp.route("/place_order", methods=["POST"])
@cross_origin()
def place_order():
    """Coloca uma nova ordem."""
    try:
        if not app_state["is_initialized"]:
            return jsonify({"error": "Sistema não inicializado"}), 400

        data = request.get_json()
        if not data:
            return jsonify({"error": "Dados não fornecidos"}), 400

        # Validar dados obrigatórios
        required_fields = ["symbol", "side", "type", "quantity"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Campo obrigatório: {field}"}), 400

        engine = app_state["execution_engine"]

        # Converter strings para enums
        side = OrderSide.BUY if data["side"].lower() == "buy" else OrderSide.SELL
        order_type = OrderType(data["type"].lower())

        order_id = engine.place_order(
            symbol=data["symbol"],
            side=side,
            order_type=order_type,
            quantity=float(data["quantity"]),
            price=float(data["price"]) if data.get("price") else None,
            stop_price=float(data["stop_price"]) if data.get("stop_price") else None,
        )

        return (
            jsonify({"message": "Ordem colocada com sucesso", "order_id": order_id}),
            200,
        )

    except Exception as e:
        logger.error(f"Erro ao colocar ordem: {e}")
        return jsonify({"error": str(e)}), 500


@trading_bp.route("/cancel_order/<order_id>", methods=["DELETE"])
@cross_origin()
def cancel_order(order_id):
    """Cancela uma ordem."""
    try:
        if not app_state["is_initialized"]:
            return jsonify({"error": "Sistema não inicializado"}), 400

        engine = app_state["execution_engine"]
        success = engine.cancel_order(order_id)

        if success:
            return jsonify({"message": "Ordem cancelada com sucesso"}), 200
        else:
            return (
                jsonify({"error": "Ordem não encontrada ou não pode ser cancelada"}),
                404,
            )

    except Exception as e:
        logger.error(f"Erro ao cancelar ordem: {e}")
        return jsonify({"error": str(e)}), 500


@trading_bp.route("/market_data", methods=["POST"])
@cross_origin()
def update_market_data():
    """Atualiza dados de mercado."""
    try:
        if not app_state["is_initialized"]:
            return jsonify({"error": "Sistema não inicializado"}), 400

        data = request.get_json()
        if not data:
            return jsonify({"error": "Dados não fornecidos"}), 400

        symbol = data.get("symbol")
        price = data.get("price")

        if not symbol or not price:
            return jsonify({"error": "Symbol e price são obrigatórios"}), 400

        engine = app_state["execution_engine"]
        engine.update_market_price(symbol, float(price))

        return jsonify({"message": "Preço atualizado com sucesso"}), 200

    except Exception as e:
        logger.error(f"Erro ao atualizar dados de mercado: {e}")
        return jsonify({"error": str(e)}), 500


@trading_bp.route("/risk_metrics", methods=["GET"])
@cross_origin()
def get_risk_metrics():
    """Retorna métricas de risco."""
    try:
        if not app_state["is_initialized"]:
            return jsonify({"error": "Sistema não inicializado"}), 400

        risk_manager = app_state["risk_manager"]

        # Gerar relatório de risco
        report = risk_manager.get_risk_report()

        return jsonify(report), 200

    except Exception as e:
        logger.error(f"Erro ao obter métricas de risco: {e}")
        return jsonify({"error": str(e)}), 500


@trading_bp.route("/health", methods=["GET"])
@cross_origin()
def health_check():
    """Health check da API."""
    return (
        jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
            }
        ),
        200,
    )


@trading_bp.route("/signal", methods=["GET"])
@cross_origin()
def get_signal():
    """Endpoint simples que retorna sinal mockado."""
    return jsonify({"signal": "hold", "timestamp": datetime.now().isoformat()}), 200

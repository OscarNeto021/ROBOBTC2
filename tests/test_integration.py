"""
Testes de integração para o sistema de trading.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from btc_perp_trader.backtest.backtester import Backtester, MLTradingStrategy
    from btc_perp_trader.execution.execution_engine import (
        ExecutionEngine,
        OrderSide,
        OrderType,
        TradingBot,
    )
    from btc_perp_trader.features.feature_engineering import FeatureEngineer
    from btc_perp_trader.models.ensemble_model import EnsembleModel
    from btc_perp_trader.models.xgboost_model import XGBoostModel
    from btc_perp_trader.risk.risk_manager import RiskManager
except ImportError as e:
    import pytest
    pytest.skip(f"Erro ao importar módulos: {e}", allow_module_level=True)


class TestIntegration(unittest.TestCase):
    """Testes de integração do sistema completo."""

    def setUp(self):
        """Configurar dados de teste."""
        np.random.seed(42)

        # Criar dados de mercado sintéticos
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="5min")

        # Simular movimento de preços com tendência e volatilidade
        price_changes = np.random.normal(0.0001, 0.01, n)  # Pequenos movimentos
        prices = 50000 * np.cumprod(1 + price_changes)

        self.market_data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices * (1 + np.random.normal(0, 0.0005, n)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.001, n))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.001, n))),
                "close": prices,
                "volume": np.random.randint(100, 1000, n),
            }
        )

        # Criar feature engineer
        self.feature_engineer = FeatureEngineer()

    def test_feature_engineering_pipeline(self):
        """Testar pipeline de engenharia de features."""
        print("Testando pipeline de features...")

        # Processar features
        features_df = self.feature_engineer.process_full_pipeline(self.market_data)

        # Verificações
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertGreater(features_df.shape[1], self.market_data.shape[1])
        self.assertFalse(features_df.empty)

        # Verificar se targets foram criados
        target_cols = [col for col in features_df.columns if col.startswith("target_")]
        self.assertGreater(len(target_cols), 0)

        print(f"✓ Features criadas: {features_df.shape[1]} colunas")

    def test_model_training_and_prediction(self):
        """Testar treinamento e previsão de modelos."""
        print("Testando treinamento de modelos...")

        # Criar features
        features_df = self.feature_engineer.process_full_pipeline(self.market_data)

        # Testar XGBoost
        model = XGBoostModel({"n_estimators": 50, "max_depth": 3})

        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = (
            model.prepare_data(features_df, "target_return_1")
        )

        # Treinar modelo
        training_result = model.train(X_train, y_train, X_val, y_val)

        # Verificações
        self.assertTrue(model.is_trained)
        self.assertIsNotNone(training_result)

        # Testar previsões
        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))

        # Avaliar modelo
        metrics = model.evaluate(X_test, y_test, "regression")
        self.assertIn("mse", metrics)
        self.assertIn("r2", metrics)

        print(f"✓ Modelo treinado - R²: {metrics['r2']:.4f}")

    def test_ensemble_model(self):
        """Testar ensemble de modelos."""
        print("Testando ensemble...")

        # Criar features
        features_df = self.feature_engineer.process_full_pipeline(self.market_data)

        # Carregar ou treinar ensemble simplificado
        ensemble = EnsembleModel.load_or_train("test")

        # Usar primeira linha de features para prever
        p_long, p_short = ensemble.predict_proba(features_df.iloc[0])
        self.assertAlmostEqual(p_long + p_short, 1.0, places=5)

        print("✓ Ensemble carregado e previsão gerada")

    def test_backtesting_system(self):
        """Testar sistema de backtesting."""
        print("Testando backtesting...")

        # Criar features
        features_df = self.feature_engineer.process_full_pipeline(self.market_data)

        # Treinar modelo simples
        model = XGBoostModel({"n_estimators": 30})
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = (
            model.prepare_data(features_df, "target_return_1")
        )
        model.train(X_train, y_train, X_val, y_val)

        # Criar estratégia
        strategy = MLTradingStrategy(model, threshold=0.001)

        # Executar backtesting
        backtester = Backtester(initial_capital=100000, commission=0.001)
        results = backtester.run_backtest(features_df, strategy)

        # Verificações
        self.assertIsInstance(results, dict)
        self.assertIn("total_return", results)
        self.assertIn("sharpe_ratio", results)
        self.assertIn("max_drawdown", results)
        self.assertIn("total_trades", results)

        print(f"✓ Backtesting concluído - Return: {results['total_return_pct']:.2f}%")

    def test_risk_management(self):
        """Testar sistema de gestão de risco."""
        print("Testando gestão de risco...")

        # Criar risk manager
        risk_manager = RiskManager()

        # Simular equity curve e retornos
        returns = np.random.normal(0.001, 0.02, 100)
        equity_curve = np.cumprod(1 + returns) * 100000

        # Calcular métricas
        metrics = risk_manager.calculate_portfolio_metrics(
            equity_curve.tolist(), returns.tolist()
        )

        # Verificações
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("var_5pct", metrics)
        self.assertIn("volatility", metrics)

        # Testar verificação de limites
        risk_checks = risk_manager.check_risk_limits(
            current_capital=95000,
            initial_capital=100000,
            open_positions=[],
            daily_pnl=-2000,
        )

        self.assertIsInstance(risk_checks, dict)
        self.assertIn("max_daily_loss", risk_checks)

        print("✓ Sistema de risco funcionando")

    def test_execution_engine(self):
        """Testar motor de execução."""
        print("Testando motor de execução...")

        # Criar execution engine
        engine = ExecutionEngine(initial_balance=100000, commission_rate=0.001)

        # Definir dados de mercado
        engine.set_market_data("BTC-USDT", self.market_data)

        # Atualizar preço
        engine.update_market_price("BTC-USDT", 50000.0)

        # Colocar ordem
        order_id = engine.place_order(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        # Verificações
        self.assertIsNotNone(order_id)
        self.assertEqual(len(engine.order_history), 1)
        self.assertEqual(len(engine.positions), 1)

        # Verificar posição
        position = engine.get_position("BTC-USDT")
        self.assertIsNotNone(position)
        self.assertGreater(position.quantity, 0)

        # Verificar métricas
        metrics = engine.get_performance_metrics()
        self.assertIn("total_trades", metrics)
        self.assertEqual(metrics["total_trades"], 1)

        print("✓ Motor de execução funcionando")

    def test_trading_bot_integration(self):
        """Testar integração completa do trading bot."""
        print("Testando integração completa...")

        # Criar features
        features_df = self.feature_engineer.process_full_pipeline(self.market_data)

        # Treinar modelo
        model = XGBoostModel({"n_estimators": 20})
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = (
            model.prepare_data(features_df, "target_return_1")
        )
        model.train(X_train, y_train, X_val, y_val)

        # Criar componentes
        execution_engine = ExecutionEngine(initial_balance=100000)
        risk_manager = RiskManager()

        # Criar bot
        bot = TradingBot(execution_engine, model, risk_manager)

        # Definir dados de mercado
        execution_engine.set_market_data("BTC-USDT", features_df)

        # Iniciar bot
        bot.start()
        self.assertTrue(bot.is_running)

        # Simular processamento de dados
        for i in range(min(10, len(features_df))):
            bot.process_market_data("BTC-USDT", features_df.iloc[i])

        # Parar bot
        bot.stop()
        self.assertFalse(bot.is_running)

        # Verificar resultados
        metrics = execution_engine.get_performance_metrics()
        self.assertIsInstance(metrics, dict)

        print("✓ Integração completa funcionando")

    def test_end_to_end_workflow(self):
        """Testar workflow completo end-to-end."""
        print("Testando workflow end-to-end...")

        # 1. Engenharia de features
        features_df = self.feature_engineer.process_full_pipeline(self.market_data)

        # 2. Treinamento de modelo
        model = XGBoostModel({"n_estimators": 30})
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = (
            model.prepare_data(features_df, "target_return_1")
        )
        model.train(X_train, y_train, X_val, y_val)

        # 3. Backtesting
        strategy = MLTradingStrategy(model, threshold=0.001)
        backtester = Backtester(initial_capital=100000)
        backtest_results = backtester.run_backtest(features_df, strategy)

        # 4. Análise de risco
        risk_manager = RiskManager()

        # 5. Execução simulada
        execution_engine = ExecutionEngine(initial_balance=100000)
        bot = TradingBot(execution_engine, model, risk_manager)

        execution_engine.set_market_data("BTC-USDT", features_df)
        bot.start()

        # Processar alguns dados
        for i in range(min(5, len(features_df))):
            bot.process_market_data("BTC-USDT", features_df.iloc[i])

        bot.stop()

        # Verificações finais
        self.assertIsInstance(backtest_results, dict)
        self.assertIn("total_return", backtest_results)

        execution_metrics = execution_engine.get_performance_metrics()
        self.assertIsInstance(execution_metrics, dict)

        print("✓ Workflow end-to-end concluído com sucesso")
        print(f"  - Backtest Return: {backtest_results['total_return_pct']:.2f}%")
        print(f"  - Execution Trades: {execution_metrics['total_trades']}")


def run_integration_tests():
    """Executar todos os testes de integração."""
    print("=" * 60)
    print("EXECUTANDO TESTES DE INTEGRAÇÃO")
    print("=" * 60)

    # Configurar unittest
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestIntegration)

    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Resumo
    print("\n" + "=" * 60)
    print("RESUMO DOS TESTES")
    print("=" * 60)
    print(f"Testes executados: {result.testsRun}")
    print(f"Falhas: {len(result.failures)}")
    print(f"Erros: {len(result.errors)}")

    if result.failures:
        print("\nFALHAS:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nERROS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nResultado: {'✓ SUCESSO' if success else '✗ FALHA'}")

    return success


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)

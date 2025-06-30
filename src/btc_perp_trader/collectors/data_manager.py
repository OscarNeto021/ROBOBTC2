"""
Gerenciador de dados que coordena todos os coletores.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd
import yaml

from .binance_ws import BinanceDataCollector
from .onchain import OnChainDataCollector
from .orderbook_ws import OrderBookCollector

logger = logging.getLogger(__name__)


class DataManager:
    """Gerenciador central para coordenar a coleta de dados."""

    def __init__(self, config_path: str = "config/config.yaml", data_dir: str = "data"):
        self.config_path = config_path
        self.data_dir = data_dir
        self.config = self._load_config()

        # Inicializar coletores
        self.binance_collector = BinanceDataCollector(
            symbol=self.config.get("data", {}).get("symbol", "BTC-USDT"),
            data_dir=data_dir,
        )

        self.orderbook_collector = OrderBookCollector(
            exchange=self.config.get("exchange", {}).get("name", "binance"),
            symbol=self.config.get("data", {}).get("symbol", "BTC-USDT"),
            depth=self.config.get("data", {}).get("orderbook_depth", 50),
            data_dir=data_dir,
        )

        self.onchain_collector = OnChainDataCollector(data_dir=data_dir)

    def _load_config(self) -> Dict:
        """Carrega a configuração do arquivo YAML."""
        try:
            with open(self.config_path, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            msg = (
                f"Arquivo de configuração {self.config_path} não encontrado. "
                "Usando configuração padrão."
            )
            logger.warning(msg)
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Retorna configuração padrão."""
        return {
            "exchange": {"name": "binance"},
            "data": {
                "symbol": "BTC-USDT",
                "candle_interval": "5m",
                "candle_limit": 5000,
                "orderbook_depth": 50,
            },
        }

    async def collect_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Coleta dados históricos de todas as fontes."""
        logger.info("Iniciando coleta de dados históricos...")

        results = {}

        # Coletar velas históricas
        try:
            candle_limit = self.config.get("data", {}).get("candle_limit", 5000)
            candle_interval = self.config.get("data", {}).get("candle_interval", "5m")

            historical_candles = (
                await self.binance_collector.collect_historical_candles(
                    limit=candle_limit, interval=candle_interval
                )
            )

            if not historical_candles.empty:
                results["historical_candles"] = historical_candles
                logger.info(f"Coletadas {len(historical_candles)} velas históricas")

        except Exception as e:
            logger.error(f"Erro ao coletar velas históricas: {e}")

        # Coletar dados on-chain
        try:
            onchain_data = await self.onchain_collector.collect_and_save()
            if onchain_data:
                results["onchain_data"] = pd.DataFrame(onchain_data)
                logger.info(f"Coletados {len(onchain_data)} registros on-chain")
        except Exception as e:
            logger.error(f"Erro ao coletar dados on-chain: {e}")

        return results

    async def collect_realtime_data(
        self, duration_seconds: int = 300
    ) -> Dict[str, List]:
        """Coleta dados em tempo real de todas as fontes."""
        msg = (
            "Iniciando coleta de dados em tempo real por "
            f"{duration_seconds} segundos..."
        )
        logger.info(msg)

        # Executar coletores em paralelo
        tasks = [
            self.binance_collector.start_realtime_collection(duration_seconds),
            self.orderbook_collector.start_collection(duration_seconds),
        ]

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Erro durante coleta em tempo real: {e}")

        # Retornar dados coletados
        return {
            "candles": self.binance_collector.candles_data,
            "orderbook": self.orderbook_collector.orderbook_snapshots,
        }

    def save_all_data(self):
        """Salva todos os dados coletados."""
        logger.info("Salvando todos os dados coletados...")

        # Salvar dados dos coletores individuais
        self.binance_collector.save_data()
        self.orderbook_collector.save_data()

        # Criar um resumo dos dados coletados
        self._create_data_summary()

    def _create_data_summary(self):
        """Cria um resumo dos dados coletados."""
        summary = {
            "collection_timestamp": datetime.now().isoformat(),
            "candles_collected": len(self.binance_collector.candles_data),
            "orderbook_snapshots": len(self.orderbook_collector.orderbook_snapshots),
            "config_used": self.config,
        }

        summary_file = f"{self.data_dir}/collection_summary.json"
        import json

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Resumo da coleta salvo em {summary_file}")

    async def full_data_collection(self, realtime_duration: int = 300):
        """Executa uma coleta completa de dados (históricos + tempo real)."""
        logger.info("Iniciando coleta completa de dados...")

        # Coletar dados históricos
        historical_data = await self.collect_historical_data()

        # Salvar dados históricos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for data_type, df in historical_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                file_path = f"{self.data_dir}/processed/{data_type}_{timestamp}.parquet"
                os.makedirs(f"{self.data_dir}/processed", exist_ok=True)
                df.to_parquet(file_path, index=False)
                logger.info(f"Dados históricos {data_type} salvos em {file_path}")

        # Coletar dados em tempo real
        realtime_data = await self.collect_realtime_data(
            duration_seconds=realtime_duration
        )

        # Salvar todos os dados
        self.save_all_data()

        logger.info("Coleta completa de dados finalizada")
        return historical_data, realtime_data


async def main():
    """Função principal para teste."""
    logging.basicConfig(level=logging.INFO)

    # Criar diretórios necessários
    os.makedirs("../../data", exist_ok=True)
    os.makedirs("../../config", exist_ok=True)

    manager = DataManager(config_path="../../config/config.yaml", data_dir="../../data")

    # Executar coleta completa (históricos + 60 segundos de tempo real)
    await manager.full_data_collection(realtime_duration=60)


if __name__ == "__main__":
    asyncio.run(main())

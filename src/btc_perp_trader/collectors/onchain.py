"""
Coletor de dados on-chain para Bitcoin.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


class OnChainDataCollector:
    """Coletor de dados on-chain para Bitcoin."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.base_urls = {
            "blockchain_info": "https://api.blockchain.info",
            "blockchair": "https://api.blockchair.com/bitcoin",
            "mempool": "https://mempool.space/api",
        }

        # Criar diretórios se não existirem
        os.makedirs(f"{data_dir}/onchain", exist_ok=True)

    async def get_blockchain_info_stats(self) -> Dict:
        """Coleta estatísticas básicas da blockchain.info."""
        try:
            url = f"{self.base_urls['blockchain_info']}/stats"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "timestamp": datetime.now(),
                            "source": "blockchain_info",
                            "total_bitcoins": data.get("totalbc", 0)
                            / 100000000,  # Converter de satoshis
                            "market_price_usd": data.get("market_price_usd", 0),
                            "hash_rate": data.get("hash_rate", 0),
                            "difficulty": data.get("difficulty", 0),
                            "minutes_between_blocks": data.get(
                                "minutes_between_blocks", 0
                            ),
                            "n_btc_mined": data.get("n_btc_mined", 0) / 100000000,
                            "n_tx": data.get("n_tx", 0),
                            "n_blocks_mined": data.get("n_blocks_mined", 0),
                            "total_fees_btc": data.get("total_fees_btc", 0) / 100000000,
                            "trade_volume_btc": data.get("trade_volume_btc", 0),
                            "trade_volume_usd": data.get("trade_volume_usd", 0),
                        }
                    else:
                        msg = (
                            "Erro ao coletar dados da blockchain.info: "
                            f"{response.status}"
                        )
                        logger.error(msg)
                        return {}
        except Exception as e:
            logger.error(f"Erro ao coletar dados da blockchain.info: {e}")
            return {}

    async def get_mempool_stats(self) -> Dict:
        """Coleta estatísticas do mempool."""
        try:
            url = f"{self.base_urls['mempool']}/mempool"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "timestamp": datetime.now(),
                            "source": "mempool",
                            "mempool_count": data.get("count", 0),
                            "mempool_vsize": data.get("vsize", 0),
                            "mempool_total_fee": data.get("total_fee", 0),
                            "fee_histogram": data.get("fee_histogram", []),
                        }
                    else:
                        logger.error(
                            f"Erro ao coletar dados do mempool: {response.status}"
                        )
                        return {}
        except Exception as e:
            logger.error(f"Erro ao coletar dados do mempool: {e}")
            return {}

    async def get_difficulty_adjustment(self) -> Dict:
        """Coleta informações sobre o ajuste de dificuldade."""
        try:
            url = f"{self.base_urls['mempool']}/difficulty-adjustment"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "timestamp": datetime.now(),
                            "source": "mempool",
                            "progress_percent": data.get("progressPercent", 0),
                            "difficulty_change": data.get("difficultyChange", 0),
                            "estimated_retarget_date": data.get(
                                "estimatedRetargetDate", 0
                            ),
                            "remaining_blocks": data.get("remainingBlocks", 0),
                            "remaining_time": data.get("remainingTime", 0),
                        }
                    else:
                        msg = (
                            "Erro ao coletar dados de difficulty adjustment: "
                            f"{response.status}"
                        )
                        logger.error(msg)
                        return {}
        except Exception as e:
            logger.error(f"Erro ao coletar dados de difficulty adjustment: {e}")
            return {}

    def get_exchange_flows_proxy(self) -> Dict:
        """Proxy para dados de fluxo de exchanges.

        Em um ambiente real, isso seria conectado a APIs como Glassnode,
        CryptoQuant, etc.
        """
        # Simulação de dados de fluxo de exchanges
        # Em produção, isso seria substituído por chamadas reais para APIs
        return {
            "timestamp": datetime.now(),
            "source": "proxy",
            "exchange_inflow_btc": 0,  # Placeholder
            "exchange_outflow_btc": 0,  # Placeholder
            "exchange_netflow_btc": 0,  # Placeholder
            "exchange_balance_btc": 0,  # Placeholder
            "note": "Dados simulados - substituir por API real em produção",
        }

    def calculate_onchain_features(self, data_list: List[Dict]) -> pd.DataFrame:
        """Calcula features derivadas dos dados on-chain."""
        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list)

        # Features básicas
        if "hash_rate" in df.columns:
            df["hash_rate_ma_7d"] = df["hash_rate"].rolling(window=7).mean()
            df["hash_rate_change"] = df["hash_rate"].pct_change()

        if "difficulty" in df.columns:
            df["difficulty_change"] = df["difficulty"].pct_change()

        if "mempool_count" in df.columns:
            df["mempool_congestion"] = (
                df["mempool_count"] / df["mempool_count"].rolling(window=24).mean()
            )

        # Features de sentiment on-chain
        if "exchange_netflow_btc" in df.columns:
            df["exchange_flow_sentiment"] = (
                df["exchange_netflow_btc"].rolling(window=7).mean()
            )

        return df

    async def collect_all_onchain_data(self) -> List[Dict]:
        """Coleta todos os dados on-chain disponíveis."""
        logger.info("Coletando dados on-chain...")

        tasks = [
            self.get_blockchain_info_stats(),
            self.get_mempool_stats(),
            self.get_difficulty_adjustment(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtrar resultados válidos
        valid_data = []
        for result in results:
            if isinstance(result, dict) and result:
                valid_data.append(result)

        # Adicionar dados de proxy de exchanges
        exchange_data = self.get_exchange_flows_proxy()
        if exchange_data:
            valid_data.append(exchange_data)

        logger.info(f"Coletados {len(valid_data)} conjuntos de dados on-chain")
        return valid_data

    def save_data(self, data_list: List[Dict]):
        """Salva os dados on-chain coletados."""
        if not data_list:
            logger.warning("Nenhum dado on-chain para salvar")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Salvar dados brutos
        raw_df = pd.DataFrame(data_list)
        raw_file = f"{self.data_dir}/onchain/onchain_raw_{timestamp}.parquet"
        raw_df.to_parquet(raw_file, index=False)
        logger.info(f"Dados on-chain brutos salvos em {raw_file}")

        # Calcular e salvar features
        features_df = self.calculate_onchain_features(data_list)
        if not features_df.empty:
            features_file = (
                f"{self.data_dir}/onchain/onchain_features_{timestamp}.parquet"
            )
            features_df.to_parquet(features_file, index=False)
            logger.info(f"Features on-chain salvas em {features_file}")

    async def collect_and_save(self):
        """Coleta dados on-chain e salva."""
        data = await self.collect_all_onchain_data()
        self.save_data(data)
        return data


async def main():
    """Função principal para teste."""
    logging.basicConfig(level=logging.INFO)

    collector = OnChainDataCollector(data_dir="../../data")
    await collector.collect_and_save()


if __name__ == "__main__":
    asyncio.run(main())

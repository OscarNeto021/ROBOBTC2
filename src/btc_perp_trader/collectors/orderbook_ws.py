"""
Coletor especializado para dados de order book via WebSocket.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd
from cryptofeed import FeedHandler
from cryptofeed.defines import L2_BOOK
from cryptofeed.exchanges import OKX, Binance, Bybit

logger = logging.getLogger(__name__)


class OrderBookCollector:
    """Coletor especializado para dados de order book."""

    def __init__(
        self,
        exchange: str = "binance",
        symbol: str = "BTC-USDT",
        depth: int = 50,
        data_dir: str = "data",
    ):
        self.exchange = exchange.lower()
        self.symbol = symbol
        self.depth = depth
        self.data_dir = data_dir
        self.orderbook_snapshots: List[Dict] = []

        # Mapear exchanges
        self.exchange_map = {"binance": Binance, "bybit": Bybit, "okx": OKX}

        # Criar diretórios se não existirem
        os.makedirs(f"{data_dir}/raw", exist_ok=True)

    async def orderbook_callback(self, book, receipt_timestamp):
        """Callback para processar snapshots do order book."""
        # Capturar apenas os primeiros N níveis (depth)
        bids = list(book.book.bids.items())[: self.depth]
        asks = list(book.book.asks.items())[: self.depth]

        # Calcular métricas de microestrutura
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        spread = best_ask - best_bid if best_bid and best_ask else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0

        # Calcular imbalance (desequilíbrio)
        total_bid_volume = sum([volume for _, volume in bids])
        total_ask_volume = sum([volume for _, volume in asks])
        imbalance = (
            (total_bid_volume - total_ask_volume)
            / (total_bid_volume + total_ask_volume)
            if (total_bid_volume + total_ask_volume) > 0
            else 0
        )

        snapshot = {
            "timestamp": book.timestamp,
            "symbol": book.symbol,
            "exchange": self.exchange,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "mid_price": mid_price,
            "imbalance": imbalance,
            "total_bid_volume": total_bid_volume,
            "total_ask_volume": total_ask_volume,
            "bids": bids,
            "asks": asks,
            "receipt_timestamp": receipt_timestamp,
        }

        self.orderbook_snapshots.append(snapshot)
        logger.debug(
            f"Order book snapshot: spread={spread:.2f}, imbalance={imbalance:.4f}"
        )

    def calculate_microstructure_features(self, snapshots: List[Dict]) -> pd.DataFrame:
        """Calcula features de microestrutura a partir dos snapshots."""
        if not snapshots:
            return pd.DataFrame()

        df = pd.DataFrame(snapshots)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        # Features de microestrutura
        df["spread_bps"] = (
            df["spread"] / df["mid_price"]
        ) * 10000  # Spread em basis points
        df["price_impact_bid"] = df["best_bid"] - df["mid_price"]
        df["price_impact_ask"] = df["best_ask"] - df["mid_price"]

        # Rolling features (janela de 10 snapshots)
        window = 10
        df["spread_ma"] = df["spread"].rolling(window=window).mean()
        df["imbalance_ma"] = df["imbalance"].rolling(window=window).mean()
        df["mid_price_volatility"] = df["mid_price"].rolling(window=window).std()

        # Features de momentum do order book
        df["bid_volume_change"] = df["total_bid_volume"].pct_change()
        df["ask_volume_change"] = df["total_ask_volume"].pct_change()
        df["imbalance_change"] = df["imbalance"].diff()

        return df

    async def start_collection(self, duration_seconds: int = 300):
        """Inicia a coleta de dados do order book."""
        logger.info(f"Iniciando coleta de order book por {duration_seconds} segundos")

        if self.exchange not in self.exchange_map:
            raise ValueError(f"Exchange {self.exchange} não suportada")

        # Configurar o feed handler
        fh = FeedHandler()
        exchange_class = self.exchange_map[self.exchange]

        fh.add_feed(
            exchange_class(
                symbols=[self.symbol],
                channels=[L2_BOOK],
                callbacks={L2_BOOK: self.orderbook_callback},
            )
        )

        # Executar por um tempo determinado
        try:
            await asyncio.wait_for(fh.run(), timeout=duration_seconds)
        except asyncio.TimeoutError:
            logger.info("Coleta de order book finalizada")

    def save_data(self, include_microstructure: bool = True):
        """Salva os dados coletados em arquivos Parquet."""
        if not self.orderbook_snapshots:
            logger.warning("Nenhum dado de order book para salvar")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Salvar snapshots brutos
        raw_df = pd.DataFrame(self.orderbook_snapshots)
        name = self.symbol.replace('-', '_')
        raw_file = (
            f"{self.data_dir}/raw/orderbook_raw_{self.exchange}_{name}_{timestamp}.parquet"
        )
        raw_df.to_parquet(raw_file, index=False)
        logger.info(f"Snapshots brutos salvos em {raw_file}")

        # Calcular e salvar features de microestrutura
        if include_microstructure:
            microstructure_df = self.calculate_microstructure_features(
                self.orderbook_snapshots
            )
            if not microstructure_df.empty:
                name = self.symbol.replace('-', '_')
                microstructure_file = (
                    f"{self.data_dir}/processed/orderbook_microstructure_{self.exchange}_{name}_{timestamp}.parquet"
                )
                os.makedirs(f"{self.data_dir}/processed", exist_ok=True)
                microstructure_df.to_parquet(microstructure_file, index=False)
                logger.info(
                    f"Features de microestrutura salvas em {microstructure_file}"
                )

    async def collect_and_save(self, duration_seconds: int = 300):
        """Coleta dados do order book e salva."""
        await self.start_collection(duration_seconds=duration_seconds)
        self.save_data()


async def main():
    """Função principal para teste."""
    logging.basicConfig(level=logging.INFO)

    collector = OrderBookCollector(
        exchange="binance", symbol="BTC-USDT", depth=50, data_dir="../../data"
    )
    await collector.collect_and_save(duration_seconds=60)


if __name__ == "__main__":
    asyncio.run(main())

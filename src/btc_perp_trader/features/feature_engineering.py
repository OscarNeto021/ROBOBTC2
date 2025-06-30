"""
Módulo de engenharia de features para o robô de trading BTC-PERP.
Inclui indicadores técnicos tradicionais e inovações do agente.
"""

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from numba import jit

# Suprimir warnings de pandas
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Classe principal para engenharia de features."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.features_cache = {}

    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria indicadores técnicos tradicionais."""
        logger.info("Criando indicadores técnicos tradicionais...")

        # Verificar se as colunas necessárias existem
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colunas faltando no DataFrame: {missing_cols}")

        df = df.copy()

        # RSI (Relative Strength Index) - Usando Numba otimizado
        df["rsi_14"] = calculate_rsi_numba(df["close"].values, length=14)
        df["rsi_21"] = calculate_rsi_numba(df["close"].values, length=21)

        # Médias móveis
        df["sma_20"] = df["close"].rolling(20, min_periods=1).mean()
        df["sma_50"] = df["close"].rolling(50, min_periods=1).mean()
        df["ema_12"] = ta.ema(df["close"], length=12)
        df["ema_26"] = ta.ema(df["close"], length=26)

        # MACD
        macd_data = ta.macd(df["close"])
        if macd_data is not None:
            df["macd"] = macd_data["MACD_12_26_9"]
            df["macd_signal"] = macd_data["MACDs_12_26_9"]
            df["macd_histogram"] = macd_data["MACDh_12_26_9"]
        else:
            df["macd"] = 0
            df["macd_signal"] = 0
            df["macd_histogram"] = 0

        # Bollinger Bands
        bb_data = ta.bbands(df["close"], length=20)
        if bb_data is not None:
            df["bb_upper"] = bb_data["BBU_20_2.0"]
            df["bb_middle"] = bb_data["BBM_20_2.0"]
            df["bb_lower"] = bb_data["BBL_20_2.0"]
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            df["bb_position"] = (df["close"] - df["bb_lower"]) / (
                df["bb_upper"] - df["bb_lower"]
            )

        # ATR (Average True Range)
        df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # Stochastic
        stoch_data = ta.stoch(df["high"], df["low"], df["close"])
        if stoch_data is not None:
            df["stoch_k"] = stoch_data["STOCHk_14_3_3"]
            df["stoch_d"] = stoch_data["STOCHd_14_3_3"]

        # Volume indicators
        df["volume_sma"] = ta.sma(df["volume"], length=20)
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # OBV (On Balance Volume)
        df["obv"] = ta.obv(df["close"], df["volume"])

        count = len([col for col in df.columns if col not in required_cols])
        logger.info(f"Criados {count} indicadores técnicos")
        return df

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features baseadas em preço."""
        logger.info("Criando features baseadas em preço...")

        df = df.copy()

        # Retornos
        df["returns_1"] = df["close"].pct_change(1, fill_method=None)
        df["returns_5"] = df["close"].pct_change(5, fill_method=None)
        df["returns_15"] = df["close"].pct_change(15, fill_method=None)

        # Log returns
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Volatilidade realizada (rolling std dos retornos)
        df["volatility_10"] = df["returns_1"].rolling(window=10).std()
        df["volatility_20"] = df["returns_1"].rolling(window=20).std()

        # High-Low spread
        df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
        df["hl_spread_ma"] = df["hl_spread"].rolling(window=10).mean()

        # Gaps
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        # Preço relativo às médias
        df["price_vs_sma20"] = df["close"] / df["sma_20"] - 1
        df["price_vs_sma50"] = df["close"] / df["sma_50"] - 1

        # Momentum features
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1

        logger.info("Features de preço criadas")
        return df

    def create_microstructure_features(
        self, orderbook_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Cria features de microestrutura do mercado."""
        logger.info("Criando features de microestrutura...")

        if orderbook_df.empty:
            logger.warning("DataFrame de order book vazio")
            return pd.DataFrame()

        df = orderbook_df.copy()

        # Features básicas já calculadas no coletor
        # Adicionar features derivadas

        # Spread normalizado
        df["spread_normalized"] = df["spread"] / df["mid_price"]

        # Momentum do imbalance
        df["imbalance_momentum"] = df["imbalance"].diff()
        df["imbalance_ma_5"] = df["imbalance"].rolling(window=5).mean()
        df["imbalance_std_5"] = df["imbalance"].rolling(window=5).std()

        # Volatilidade do mid price
        df["mid_price_returns"] = df["mid_price"].pct_change(fill_method=None)
        df["mid_price_volatility"] = df["mid_price_returns"].rolling(window=10).std()

        # Pressure indicators
        df["buy_pressure"] = df["total_bid_volume"] / (
            df["total_bid_volume"] + df["total_ask_volume"]
        )
        df["sell_pressure"] = df["total_ask_volume"] / (
            df["total_bid_volume"] + df["total_ask_volume"]
        )

        # Spread momentum
        df["spread_momentum"] = df["spread"].diff()
        df["spread_acceleration"] = df["spread_momentum"].diff()

        logger.info("Features de microestrutura criadas")
        return df

    def create_onchain_features(self, onchain_df: pd.DataFrame) -> pd.DataFrame:
        """Cria features baseadas em dados on-chain."""
        logger.info("Criando features on-chain...")

        if onchain_df.empty:
            logger.warning("DataFrame on-chain vazio")
            return pd.DataFrame()

        df = onchain_df.copy()

        # Hash rate features
        if "hash_rate" in df.columns:
            df["hash_rate_ma_7"] = df["hash_rate"].rolling(window=7).mean()
            df["hash_rate_momentum"] = df["hash_rate"].pct_change(7, fill_method=None)
            df["hash_rate_vs_ma"] = df["hash_rate"] / df["hash_rate_ma_7"] - 1

        # Difficulty features
        if "difficulty" in df.columns:
            df["difficulty_momentum"] = df["difficulty"].pct_change(fill_method=None)

        # Mempool features
        if "mempool_count" in df.columns:
            df["mempool_congestion"] = (
                df["mempool_count"] / df["mempool_count"].rolling(window=24).mean()
            )
            df["mempool_fee_pressure"] = df["mempool_total_fee"] / df["mempool_count"]

        # Network activity
        if "n_tx" in df.columns:
            df["tx_momentum"] = df["n_tx"].pct_change(fill_method=None)
            df["tx_ma_7"] = df["n_tx"].rolling(window=7).mean()

        logger.info("Features on-chain criadas")
        return df

    def create_sentiment_features(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Inovação do Agente 2: Cria features baseadas em dados de sentimento.
        """
        logger.info("Criando features de sentimento...")

        if sentiment_df.empty:
            logger.warning("DataFrame de sentimento vazio")
            return pd.DataFrame()

        df = sentiment_df.copy()

        # Exemplo de features de sentimento (placeholders)
        if "sentiment_score" in df.columns:
            df["sentiment_ma_7"] = df["sentiment_score"].rolling(window=7).mean()
            df["sentiment_momentum"] = df["sentiment_score"].diff()

        if "news_volume" in df.columns:
            df["news_volume_ma_7"] = df["news_volume"].rolling(window=7).mean()
            df["news_volume_change"] = df["news_volume"].pct_change()

        logger.info("Features de sentimento criadas")
        return df

    def create_regime_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inovação do Agente 1: Detecção de Regimes de Mercado.
        Cria features para identificar diferentes regimes de mercado.
        """
        logger.info("Criando features de detecção de regimes de mercado...")

        df = df.copy()

        # Volatilidade regime
        df["volatility_regime"] = self._classify_volatility_regime(df["volatility_20"])

        # Trend regime
        df["trend_regime"] = self._classify_trend_regime(df)

        # Volume regime
        df["volume_regime"] = self._classify_volume_regime(df["volume_ratio"])

        # Regime score combinado
        df["market_regime_score"] = (
            df["volatility_regime"] * 0.4
            + df["trend_regime"] * 0.4
            + df["volume_regime"] * 0.2
        )

        logger.info("Features de detecção de regimes criadas")
        return df

    def _classify_volatility_regime(self, volatility: pd.Series) -> pd.Series:
        """Classifica regime de volatilidade."""
        # Calcular quantis de forma correta
        vol_33 = volatility.rolling(window=100).quantile(0.33)
        vol_67 = volatility.rolling(window=100).quantile(0.67)

        regime = pd.Series(index=volatility.index, dtype=float)
        regime[volatility <= vol_33] = 0  # Baixa volatilidade
        regime[(volatility > vol_33) & (volatility <= vol_67)] = 1  # Média
        regime[volatility > vol_67] = 2  # Alta volatilidade

        return regime.fillna(1)  # Default para regime médio

    def _classify_trend_regime(self, df: pd.DataFrame) -> pd.Series:
        """Classifica regime de tendência."""
        # Usar MACD e médias móveis para determinar tendência
        trend_score = pd.Series(index=df.index, dtype=float)

        # Condições de alta
        uptrend_condition = (
            (df["close"] > df["sma_20"])
            & (df["sma_20"] > df["sma_50"])
            & (df["macd"] > df["macd_signal"])
        )

        # Condições de baixa
        downtrend_condition = (
            (df["close"] < df["sma_20"])
            & (df["sma_20"] < df["sma_50"])
            & (df["macd"] < df["macd_signal"])
        )

        trend_score[uptrend_condition] = 2  # Uptrend
        trend_score[downtrend_condition] = 0  # Downtrend
        trend_score[~(uptrend_condition | downtrend_condition)] = 1  # Sideways

        return trend_score.fillna(1)

    def _classify_volume_regime(self, volume_ratio: pd.Series) -> pd.Series:
        """Classifica regime de volume."""
        regime = pd.Series(index=volume_ratio.index, dtype=float)

        regime[volume_ratio <= 0.8] = 0  # Baixo volume
        regime[(volume_ratio > 0.8) & (volume_ratio <= 1.5)] = 1  # Volume normal
        regime[volume_ratio > 1.5] = 2  # Alto volume

        return regime.fillna(1)

    def create_target_variables(
        self, df: pd.DataFrame, horizons: List[int] = [1, 3, 5]
    ) -> pd.DataFrame:
        """Cria variáveis target para diferentes horizontes de previsão."""
        logger.info(f"Criando variáveis target para horizontes: {horizons}")

        df = df.copy()

        for horizon in horizons:
            # Retorno futuro
            df[f"target_return_{horizon}"] = (
                df["close"].shift(-horizon) / df["close"] - 1
            )

            # Direção (classificação)
            df[f"target_direction_{horizon}"] = (
                df[f"target_return_{horizon}"] > 0
            ).astype(int)

            # Volatilidade futura
            future_returns = (
                df["returns_1"].shift(-horizon).rolling(window=horizon).std()
            )
            df[f"target_volatility_{horizon}"] = future_returns

        logger.info("Variáveis target criadas")
        return df

    def process_full_pipeline(
        self,
        candles_df: pd.DataFrame,
        orderbook_df: Optional[pd.DataFrame] = None,
        onchain_df: Optional[pd.DataFrame] = None,
        sentiment_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Pipeline completo de engenharia de features."""
        logger.info("Iniciando pipeline completo de engenharia de features...")

        # Features técnicas básicas
        df = self.create_technical_indicators(candles_df)
        df = self.create_price_features(df)

        # Features de regime (Inovação do Agente)
        df = self.create_regime_detection_features(df)

        # Features de microestrutura (se disponível)
        if orderbook_df is not None and not orderbook_df.empty:
            microstructure_features = self.create_microstructure_features(orderbook_df)
            if not microstructure_features.empty:
                # Merge por timestamp (assumindo que ambos têm timestamp)
                df = self._merge_features_by_timestamp(
                    df, microstructure_features, "microstructure"
                )

        # Features on-chain (se disponível)
        if onchain_df is not None and not onchain_df.empty:
            onchain_features = self.create_onchain_features(onchain_df)
            if not onchain_features.empty:
                df = self._merge_features_by_timestamp(df, onchain_features, "onchain")

        # Features de sentimento (se disponível)
        if sentiment_df is not None and not sentiment_df.empty:
            sentiment_features = self.create_sentiment_features(sentiment_df)
            if not sentiment_features.empty:
                df = self._merge_features_by_timestamp(
                    df, sentiment_features, "sentiment"
                )

        # Criar targets
        df = self.create_target_variables(df)

        # Limpeza final
        df = self._clean_features(df)

        logger.info(f"Pipeline completo finalizado. Shape final: {df.shape}")
        return df

    def _merge_features_by_timestamp(
        self, main_df: pd.DataFrame, features_df: pd.DataFrame, prefix: str
    ) -> pd.DataFrame:
        """Merge features por timestamp com tratamento de frequências diferentes."""
        if "timestamp" not in main_df.columns or "timestamp" not in features_df.columns:
            logger.warning(f"Timestamp não encontrado para merge de features {prefix}")
            return main_df

        # Preparar DataFrames para merge
        main_df = (
            main_df.set_index("timestamp")
            if "timestamp" in main_df.columns
            else main_df
        )
        features_df = (
            features_df.set_index("timestamp")
            if "timestamp" in features_df.columns
            else features_df
        )

        # Merge com forward fill para lidar com frequências diferentes
        merged = main_df.join(features_df, how="left", rsuffix=f"_{prefix}")
        merged = merged.fillna(
            method="ffill"
        )  # Forward fill para dados de menor frequência

        return merged.reset_index()

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpeza final das features."""
        logger.info("Realizando limpeza final das features...")

        # Remover colunas com muitos NaNs
        nan_threshold = 0.5  # 50% de NaNs
        cols_to_drop = []
        for col in df.columns:
            if df[col].isna().sum() / len(df) > nan_threshold:
                cols_to_drop.append(col)

        if cols_to_drop:
            logger.warning(f"Removendo colunas com muitos NaNs: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        # Substituir infinitos por NaN e depois por 0
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        logger.info(f"Limpeza finalizada. Shape final: {df.shape}")
        return df


@jit(nopython=True)
def calculate_rsi_numba(prices: np.ndarray, length: int = 14) -> np.ndarray:
    """Cálculo otimizado do RSI usando Numba."""
    n = len(prices)
    rsi = np.full(n, np.nan)

    if n < length + 1:
        return rsi

    # Calcular mudanças de preço
    deltas = np.diff(prices)

    # Separar ganhos e perdas
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Primeira média
    avg_gain = np.mean(gains[:length])
    avg_loss = np.mean(losses[:length])

    if avg_loss == 0:
        rsi[length] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[length] = 100.0 - (100.0 / (1.0 + rs))

    # Calcular RSI para o resto dos dados
    for i in range(length + 1, n):
        gain = gains[i - 1]
        loss = losses[i - 1]

        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def main():
    """Função principal para teste."""
    import asyncio
    import sys

    sys.path.append("..")

    from btc_perp_trader.collectors.binance_ws import BinanceDataCollector

    async def test_feature_engineering():
        # Coletar dados de teste
        collector = BinanceDataCollector(symbol="BTC-USDT", data_dir="../data")
        df = await collector.collect_historical_candles(limit=100, interval="5m")

        if df.empty:
            print("Não foi possível coletar dados para teste")
            return

        # Criar um DataFrame de sentimento de exemplo
        sentiment_data = {
            "timestamp": df["timestamp"],
            "sentiment_score": np.random.rand(len(df)) * 2 - 1,  # Scores entre -1 e 1
            "news_volume": np.random.randint(10, 100, len(df)),
        }
        sentiment_df = pd.DataFrame(sentiment_data)

        # Testar engenharia de features
        engineer = FeatureEngineer()
        features_df = engineer.process_full_pipeline(df, sentiment_df=sentiment_df)

        print(f"Shape original: {df.shape}")
        print(f"Shape com features: {features_df.shape}")
        print(f"Colunas criadas: {features_df.shape[1] - df.shape[1]}")
        print("\nPrimeiras features criadas:")
        new_cols = [col for col in features_df.columns if col not in df.columns]
        print(new_cols[:10])

    asyncio.run(test_feature_engineering())


if __name__ == "__main__":
    main()

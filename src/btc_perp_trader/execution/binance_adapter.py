import logging

import ccxt

from btc_perp_trader.tools.trade_logger import log_open


class BinanceAdapter:
    def __init__(self, key: str, secret: str, *, testnet: bool = False) -> None:
        self.logger = logging.getLogger("BinanceAdapter")
        params = {"options": {"defaultType": "future"}}
        self.ex = ccxt.binance(params | {"apiKey": key, "secret": secret})
        # ccxt ajusta automaticamente as URLs quando o sandbox é ativado
        self.ex.set_sandbox_mode(testnet)

        # ativa modo Isolated e alavancagem 20×
        try:
            self.ex.set_margin_mode("isolated", "BTCUSDT")
            self.set_leverage(20, "BTCUSDT")
        except Exception as e:
            self.logger.warning("Falha ao setar margin/leverage: %s", e)

    # ----------------------------------------------------------- #
    def get_balance(self) -> float:
        bal = self.ex.fetch_balance()
        return bal["USDT"]["free"]

    def update_balance(self) -> float:
        return self.get_balance()

    # ---------------- orders & leverage ------------------------ #
    def market_buy(self, symbol: str, qty: float) -> dict:
        try:
            res = self.ex.create_market_buy_order(symbol, qty)
            log_open(symbol, "buy", qty, float(res["average"] or res["price"]))
            return res
        except Exception as e:
            self.logger.warning("Saldo insuficiente para comprar %s: %s", symbol, e)
            return {}

    def market_sell(self, symbol: str, qty: float) -> dict:
        try:
            res = self.ex.create_market_sell_order(symbol, qty)
            log_open(symbol, "sell", qty, float(res["average"] or res["price"]))
            return res
        except Exception as e:
            self.logger.warning("Saldo insuficiente para vender %s: %s", symbol, e)
            return {}

    def set_leverage(self, leverage: int, symbol: str) -> None:
        self.ex.set_leverage(leverage, symbol)

    # --------------- novas utilidades -----------------
    def get_position(self, symbol: str = "BTCUSDT") -> dict | None:
        """Busca posições via endpoint /fapi/v2/positionRisk.

        V1 encontra-se obsoleto na testnet.
        """
        try:
            positions = self.ex.fapiPrivateV2GetPositionRisk()
        except Exception:
            # fallback se a versão do ccxt não tiver o alias v2
            positions = self.ex.fapiPrivateGetPositionRisk({"apiVersion": "v2"})
        for p in positions:
            if p["symbol"] == symbol and float(p["positionAmt"]) != 0.0:
                return p
        return None

    def close_position(self, symbol: str, qty: float) -> None:
        """Close an open position using a market order."""
        side = "sell" if qty > 0 else "buy"
        self.ex.create_order(symbol, "MARKET", side, abs(qty))

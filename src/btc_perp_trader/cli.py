from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from btc_perp_trader.collectors.binance_user_ws import BinanceUserStream
from btc_perp_trader.collectors.binance_ws import BinanceWebsocket
from btc_perp_trader.config import BINANCE_API_KEY, BINANCE_API_SECRET
from btc_perp_trader.execution.binance_adapter import BinanceAdapter
from btc_perp_trader.models.online_model import ONLINE_MODEL
from btc_perp_trader.risk.position_sizing import atr_position_size
from btc_perp_trader.strategy import entry_signal
from btc_perp_trader.tools.bandit_rr import sample_rr, update_rr
from btc_perp_trader.tools.state_manager import load_position, save_position
from btc_perp_trader.tools.trade_logger import log_close, log_open, log_realized

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / ".." / "config"


def _read_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)


def _logger(level: str = "INFO") -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(stream=sys.stdout, format=fmt, level=level)
    logging.getLogger("ccxt").setLevel(logging.WARNING)


@click.group()
def cli() -> None:
    """BTC-PERP Trader CLI."""


@cli.command()
@click.option("--mode", type=click.Choice(["live", "demo", "backtest"]), default="demo")
@click.option("--from", "date_from", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--to", "date_to", type=click.DateTime(formats=["%Y-%m-%d"]))
def run(mode: str, date_from: Optional[datetime], date_to: Optional[datetime]) -> None:
    """Run live/demo or backtest."""
    _logger()
    if mode == "backtest":
        if not (date_from and date_to):
            raise click.BadParameter("--from/--to obrigatórios em backtest")
        return _run_backtest(date_from, date_to)

    api_key = BINANCE_API_KEY
    api_sec = BINANCE_API_SECRET
    adapter = BinanceAdapter(api_key, api_sec, testnet=(mode == "demo"))

    # ---------- sincroniza posição aberta (se houver) ----------
    on_exchange = adapter.get_position("BTCUSDT")
    stored = load_position()

    if on_exchange:
        pnl = float(on_exchange["unRealizedProfit"])
        qty = float(on_exchange["positionAmt"])
        entry = float(on_exchange["entryPrice"])
        logging.getLogger(__name__).info(
            "Posição DETECTADA %s BTC @ %.2f (PnL %.2f)", qty, entry, pnl
        )
        if stored is None or stored.get("positionAmt") != on_exchange["positionAmt"]:
            save_position(on_exchange)
    else:
        logging.getLogger(__name__).info("Nenhuma posição aberta na exchange.")
        if stored:
            save_position({})

    # change leverage if desired (default set to 20× in adapter)
    # adapter.set_leverage(10, "BTCUSDT")
    feed = BinanceWebsocket(symbol="BTCUSDT", testnet=(mode == "demo"))
    user_stream = BinanceUserStream(adapter.ex, log_realized)
    balance = adapter.get_balance()
    risk_fn = atr_position_size

    async def loop() -> None:  # pragma: no cover
        nonlocal balance
        user_task = asyncio.create_task(user_stream.loop())
        pos = adapter.get_position("BTCUSDT")
        side_now = (
            "long"
            if pos and float(pos["positionAmt"]) > 0
            else "short" if pos and float(pos["positionAmt"]) < 0 else None
        )
        entry_price = float(pos["entryPrice"]) if pos else None
        qty_open = float(pos["positionAmt"]) if pos else 0.0
        atr_mul_sl = 1.0
        bandit_idx, rr = sample_rr()  # 1.5 / 2 / 3 baseado em bandit
        stop_price = None
        take_price = None

        try:
            async for candle in feed.stream():
                feats = candle.to_features()
                if "headline" not in feats:
                    feats["headline"] = ""  # evita KeyError no modelo textual
                long_p = ONLINE_MODEL.predict(feats)
                short_p = 1 - long_p
                price = candle.close

                # ---------- risk_pct dinâmico 0.001 → 0.005 --------------------
                trend_strength = abs(candle.close - candle.open) / candle.close  # 0‒1
                risk_pct_dyn = 0.001 + 0.004 * min(
                    trend_strength * 5, 1
                )  # escala rápido

                signal = entry_signal(long_p, short_p)

                if side_now:
                    if stop_price is None:
                        if candle.atr is None:
                            logger.debug(
                                "ATR ainda não calculado — aguardando próximo candle."
                            )
                            await asyncio.sleep(0)
                            continue
                        if side_now == "long":
                            stop_price = entry_price - candle.atr * atr_mul_sl
                            take_price = entry_price + candle.atr * atr_mul_sl * rr
                        else:
                            stop_price = entry_price + candle.atr * atr_mul_sl
                            take_price = entry_price - candle.atr * atr_mul_sl * rr

                    hit_tp = (
                        price >= take_price
                        if side_now == "long"
                        else price <= take_price
                    )
                    hit_sl = (
                        price <= stop_price
                        if side_now == "long"
                        else price >= stop_price
                    )
                    reverse = (
                        signal.side
                        and signal.side != side_now
                        and signal.confidence > 0.55
                    )

                    if hit_tp or hit_sl or reverse:
                        if side_now == "long":
                            adapter.market_sell("BTCUSDT", qty_open)
                        else:
                            adapter.market_buy("BTCUSDT", abs(qty_open))
                        pnl = (
                            (price - entry_price)
                            * qty_open
                            * (1 if side_now == "long" else -1)
                        )
                        log_close("BTCUSDT", qty_open, entry_price, price)
                        # ---------- online learning ----------
                        y = 1 if pnl > 0 else 0
                        ONLINE_MODEL.learn(feats, y)
                        update_rr(bandit_idx, pnl)
                        side_now = None
                        entry_price = None
                        stop_price = None
                        take_price = None
                        balance = adapter.update_balance()

                if not side_now and signal.side:
                    qty = risk_fn(
                        balance, candle.atr, price=price, risk_pct=risk_pct_dyn
                    )
                    if qty > 0:
                        if signal.side == "long":
                            adapter.market_buy("BTCUSDT", qty)
                        else:
                            adapter.market_sell("BTCUSDT", qty)
                        log_open("BTCUSDT", signal.side, qty, price)
                        side_now = signal.side
                        entry_price = price
                        qty_open = qty
                        stop_price = None  # será definido na iteração seguinte
                await asyncio.sleep(0)
        finally:
            user_task.cancel()

    try:
        asyncio.run(loop())
    finally:
        pos = adapter.get_position("BTCUSDT")
        if pos:
            save_position(pos)


def _run_backtest(date_from: datetime, date_to: datetime) -> None:
    from btc_perp_trader.backtest.generate_report import Backtester
    from btc_perp_trader.models.ensemble_model import EnsembleModel

    bt = Backtester("BTCUSDT", date_from, date_to, EnsembleModel, atr_position_size)
    res = bt.run()
    path = bt.generate_html_report(res, out_dir=ROOT / ".." / "reports")
    print(f"✅ Relatório em {path.resolve()}")


if __name__ == "__main__":
    cli()

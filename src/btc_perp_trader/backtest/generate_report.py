"""Gera um relatório HTML com estatísticas + lista de trades.

Uso:
    poetry run python -m btc_perp_trader.backtest.generate_report
"""

from pathlib import Path

import pandas as pd  # noqa: WPS433

from btc_perp_trader.backtest.vectorbt_backtest import run_backtest


def main() -> None:
    pf = run_backtest()
    stats = pf.stats()

    # métricas extras ---------------------------------------------------------
    stats["Total Trades"] = int(pf.trades.count())
    stats["Winning Trades"] = int(pf.trades.winning.count())
    stats["Losing Trades"] = int(pf.trades.losing.count())
    stats["Accuracy (%)"] = stats.get("Win Rate [%]", 0.0)

    initial_cash = pf.init_cash
    final_value = pf.portfolio_value().iloc[-1]
    stats["Return (%)"] = (
        (final_value - initial_cash) / initial_cash * 100 if initial_cash else 0
    )

    html = "\n".join(
        [
            "<h1>ROBOBTC2 – Relatório de Back-test</h1>",
            "<h2>Resumo de métricas</h2>",
            stats.to_frame(name="value").to_html(),
            "<h2>Trades</h2>",
            pf.trades.records_readable.to_html(index=False),
        ]
    )

    out = Path(__file__).parent / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"✅ Relatório salvo em {out}")  # noqa: WPS421


if __name__ == "__main__":  # pragma: no cover
    main()

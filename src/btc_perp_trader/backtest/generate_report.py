from pathlib import Path
from btc_perp_trader.backtest.vectorbt_backtest import run_backtest
import pandas as pd


def main():
    # Executa back-test
    pf = run_backtest()

    # Define frequência de 5 minutos (evita warnings de métricas)
    # vectorbt <0.26 não possui set_freq; ajuste diretamente
    pf.wrapper._freq = pd.Timedelta("5min")

    # pf.stats() → Series. Converta em DataFrame p/ HTML (Pandas ≥2)
    stats = pf.stats()
    
    # Métricas adicionais
    total_trades = stats.get("Total Trades", 0)
    winning_trades = stats.get("Win Trades", 0)
    losing_trades = stats.get("Loss Trades", 0)
    accuracy = stats.get("Win Rate [%]", 0)
    
    # Rendimento em % do valor investido
    initial_cash = pf.init_cash
    final_value = pf.final_value()
    if isinstance(final_value, pd.Series):
        final_value = final_value.iloc[-1]
    return_percentage = ((final_value - initial_cash) / initial_cash) * 100 if initial_cash > 0 else 0

    # Adicionar ao DataFrame de estatísticas
    stats["Total Trades"] = total_trades
    stats["Winning Trades"] = winning_trades
    stats["Losing Trades"] = losing_trades
    stats["Accuracy (%)"] = accuracy
    stats["Return (%)"] = return_percentage

    html = stats.to_frame(name="value").to_html()

    out = Path(__file__).parent / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"✅ Relatório salvo em {out}")


if __name__ == "__main__":
    main()

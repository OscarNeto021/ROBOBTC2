from pathlib import Path

import pandas as pd
import vectorbt as vbt

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "btc_1m_2025-06-30.csv"


def main():
    # Leitura do CSV completo e index temporal
    df = pd.read_csv(CSV, index_col="ts", parse_dates=True)
    price = df["close"]

    # Geração de sinais simples de média móvel
    fast_ma = price.ewm(span=10).mean()
    slow_ma = price.ewm(span=50).mean()

    entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

    # Crie o portfólio informando que cada passo é 1 minuto
    pf = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        init_cash=100_000,
        freq="1T",
    )

    # Define frequência de 1 minuto (evita warnings de métricas)
    # vectorbt <0.26 não possui set_freq; ajuste diretamente
    pf.wrapper._freq = pd.Timedelta("1min")

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

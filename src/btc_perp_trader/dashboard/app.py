"""Dashboard Streamlit para monitorar dados e trades em tempo real.

Execute:
    poetry run streamlit run src/btc_perp_trader/dashboard/app.py
"""

import streamlit as st
from btc_perp_trader.backtest.vectorbt_backtest import run_backtest
from btc_perp_trader.models.train_ensemble import load_dataset

st.set_page_config(layout="wide")
st.title("\ud83d\udcc8 ROBOBTC2 – Monitor de Performance")

# Dataset --------------------------------------------------------------------
df = load_dataset()
st.markdown("### Amostra de dados mais recentes")
st.dataframe(df.tail(10))

# Back-test -------------------------------------------------------------------
st.markdown("### Estatísticas do back-test (vectorbt)")
pf = run_backtest()
st.dataframe(pf.stats().to_frame("value"))

st.markdown("### Últimos trades")
st.dataframe(pf.trades.records_readable.tail(50))

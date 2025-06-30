import streamlit as st, pandas as pd, plotly.express as px
from pathlib import Path
from dotenv import dotenv_values, set_key

LOG_PATH = Path("logs/trades.csv")
ENV_PATH = Path(".env")

st.set_page_config(page_title="BTC-PERP Dashboard", layout="wide")
st.title("\U0001F4CA BTC-PERP Trader – Painel de Desempenho")

# ---------- Sidebar: API keys -------------------------------------------------
with st.sidebar:
    st.header("\U0001F511 Chaves Binance")
    mode = st.radio("Modo da conta", ["Demo (Testnet)", "Live"], index=0)
    demo = mode.startswith("Demo")
    cfg  = dotenv_values(ENV_PATH)
    cur_key = cfg.get("BINANCE_API_KEY_DEMO" if demo else "BINANCE_API_KEY", "")
    cur_sec = cfg.get("BINANCE_SECRET_KEY_DEMO" if demo else "BINANCE_SECRET_KEY", "")
    api_key = st.text_input("API Key", value=cur_key, type="password")
    api_sec = st.text_input("API Secret", value=cur_sec, type="password")
    if st.button("\U0001F4BE Salvar"):
        if demo:
            set_key(ENV_PATH, "BINANCE_API_KEY_DEMO", api_key)
            set_key(ENV_PATH, "BINANCE_SECRET_KEY_DEMO", api_sec)
        else:
            set_key(ENV_PATH, "BINANCE_API_KEY", api_key)
            set_key(ENV_PATH, "BINANCE_SECRET_KEY", api_sec)
        st.success("Chaves gravadas! Reinicie o robô para aplicar.")

# ---------- Carregar trades ---------------------------------------------------
if not LOG_PATH.exists() or LOG_PATH.stat().st_size < 10:
    st.info("Nenhum trade registrado ainda.")
    st.stop()

df = pd.read_csv(LOG_PATH, parse_dates=["ts"]).sort_values("ts")
df["cum_pnl"] = df["pnl"].cumsum()

col1, col2 = st.columns([3,2])

with col1:
    st.subheader("Equity Curve")
    st.plotly_chart(px.line(df, x="ts", y="cum_pnl"), use_container_width=True)

with col2:
    total = len(df); wins = (df["pnl"] > 0).sum()
    st.metric("Win-rate", f"{wins/total*100:.2f}%")
    if total >= 10:
        df["roll_win"] = df["pnl"].gt(0).rolling(50, min_periods=1).mean()*100
        st.subheader("Rolling 50-trade Accuracy")
        st.plotly_chart(px.line(df, x="ts", y="roll_win"), use_container_width=True)

st.subheader("Últimos trades")
st.dataframe(df.tail(20).iloc[::-1].reset_index(drop=True))

"""
Coletor Cryptopanic
-------------------
Lê o token do módulo `config` (que por sua vez pode vir do .env ou estar
hard-coded).  Se a API falhar, apenas devolve lista vazia e registra um aviso.
"""
import logging, requests

from btc_perp_trader.config import CRYPTOPANIC_TOKEN

URL = (
    "https://cryptopanic.com/api/v1/posts/"
    f"?auth_token={CRYPTOPANIC_TOKEN}&kind=news&filter=rising"
)


def fetch_cryptopanic() -> list[str]:
    try:
        r = requests.get(URL, timeout=10)
        r.raise_for_status()
        js = r.json()
        return [p["title"] for p in js.get("results", [])]
    except Exception as e:
        logging.warning(f"[cryptopanic] falhou: {e}")
        return []

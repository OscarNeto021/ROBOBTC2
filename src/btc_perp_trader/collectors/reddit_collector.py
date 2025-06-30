import requests, datetime, logging, json

URL = (
    "https://api.pushshift.io/reddit/search/submission/"
    "?subreddit=CryptoCurrency&after={after}&before={before}&size=500"
)


def _strip_z(iso):  # reaproveita função segura para “Z”
    return iso[:-1] if iso.endswith("Z") else iso


def fetch_reddit(after_iso, before_iso):
    after = int(datetime.datetime.fromisoformat(_strip_z(after_iso)).timestamp())
    before = int(datetime.datetime.fromisoformat(_strip_z(before_iso)).timestamp())
    try:
        r = requests.get(URL.format(after=after, before=before), timeout=20)
        if r.status_code != 200:
            logging.warning("Pushshift HTTP %s – ignorado", r.status_code)
            return []
        js = r.json()          # pode falhar se não for JSON
    except (json.JSONDecodeError, requests.exceptions.RequestException) as e:
        logging.warning("Pushshift decode/conn falhou: %s – ignorado", e)
        return []
    return [p.get("title", "") for p in js.get("data", [])]

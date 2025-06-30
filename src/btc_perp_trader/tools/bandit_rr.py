import random, math, pathlib, json

B_FILE = pathlib.Path("state/bandit_rr.json")
RRS = [1.5, 2.0, 3.0]


def _load():
    if B_FILE.exists():
        return json.loads(B_FILE.read_text())
    return {"alpha": [1] * len(RRS), "beta": [1] * len(RRS)}


def _save(state):
    B_FILE.write_text(json.dumps(state))


def sample_rr():
    st = _load()
    idx = max(
        range(len(RRS)), key=lambda i: random.betavariate(st["alpha"][i], st["beta"][i])
    )
    return idx, RRS[idx]


def update_rr(idx, reward):
    st = _load()
    st["alpha"][idx] += max(reward, 0)
    st["beta"][idx] += max(-reward, 0)
    _save(st)

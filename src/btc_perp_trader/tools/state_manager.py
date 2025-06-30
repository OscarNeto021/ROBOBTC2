from pathlib import Path
import json, logging

STATE_DIR = Path("state"); STATE_DIR.mkdir(exist_ok=True)
POS_FILE = STATE_DIR / "open_position.json"

log = logging.getLogger("StateManager")


def save_position(data: dict) -> None:
    """Persist position data to disk."""
    with POS_FILE.open("w") as fh:
        json.dump(data, fh)
    log.info("Posição salva em %s", POS_FILE)


def load_position() -> dict | None:
    """Load position data if available."""
    if not POS_FILE.exists():
        return None
    try:
        return json.loads(POS_FILE.read_text())
    except Exception as e:  # pragma: no cover - just logging
        log.warning("Falha ao ler estado: %s", e)
        return None


def clear_position() -> None:
    """Remove stored position if present."""
    if POS_FILE.exists():
        POS_FILE.unlink()
        log.info("🗑️  open_position.json removido.")


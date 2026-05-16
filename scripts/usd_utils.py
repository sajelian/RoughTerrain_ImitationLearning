from __future__ import annotations

import yaml
from pathlib import Path

USD_DATA_ROOT = "data/terrains/prepared"


def load_usd_config(usd_name: str) -> tuple[str, dict]:
    """Return (usd_path, config_dict) for the named USD terrain."""
    folder = Path(USD_DATA_ROOT) / usd_name
    usd_path = str(folder / f"{usd_name}.usd")
    cfg_path = folder / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"USD config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
    return usd_path, cfg

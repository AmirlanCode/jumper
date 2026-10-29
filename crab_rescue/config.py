"""Load config.yaml into a plain dict with a few typed accessors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).with_name("config.yaml")
ROLES = ("vlm_primary", "vlm_cheap", "planner", "planner_fast", "planner_escalation")


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))
    for role in ROLES:
        if role not in cfg["models"]:
            raise ValueError(f"config.yaml: models.{role} is missing")
    return cfg

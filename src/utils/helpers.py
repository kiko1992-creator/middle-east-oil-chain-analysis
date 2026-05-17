"""Shared utility functions."""

import logging
import logging.config
from pathlib import Path

import yaml


def load_config(path: str | Path = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(path: str | Path = "config/logging.yaml") -> None:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    Path("outputs").mkdir(exist_ok=True)
    logging.config.dictConfig(cfg)


def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

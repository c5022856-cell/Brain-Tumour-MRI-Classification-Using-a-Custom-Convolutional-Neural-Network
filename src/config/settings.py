from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class Settings:
    data: dict[str, Any]

    @property
    def project(self) -> dict[str, Any]:
        return self.data["project"]

    @property
    def paths(self) -> dict[str, Any]:
        return self.data["paths"]

    @property
    def dataset(self) -> dict[str, Any]:
        return self.data["dataset"]

    @property
    def training(self) -> dict[str, Any]:
        return self.data["training"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.data["evaluation"]

    @property
    def models(self) -> dict[str, Any]:
        return self.data["models"]


def load_settings(config_path: str | Path) -> Settings:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return Settings(data=data)

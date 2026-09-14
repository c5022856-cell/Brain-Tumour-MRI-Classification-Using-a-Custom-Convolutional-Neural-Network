"""YAML-backed project settings."""
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Settings:
    data: dict
    @property
    def project(self): return self.data['project']
    @property
    def paths(self): return self.data['paths']
    @property
    def dataset(self): return self.data['dataset']
    @property
    def models(self): return self.data['models']
    @property
    def training(self): return self.data['training']
    @property
    def evaluation(self): return self.data['evaluation']

def load_settings(path: str | Path) -> Settings:
    with Path(path).open(encoding='utf-8') as handle:
        return Settings(yaml.safe_load(handle))


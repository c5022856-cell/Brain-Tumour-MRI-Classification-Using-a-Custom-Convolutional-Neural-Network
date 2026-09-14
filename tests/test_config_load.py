from pathlib import Path

from src.config.settings import load_settings


def test_load_settings():
    settings = load_settings(Path("src/config/default.yaml"))
    assert settings.project["name"] == "brain-tumor-detection"
    assert settings.paths["primary_dataset_dir"] == "data/raw/primary"

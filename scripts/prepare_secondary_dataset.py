from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings
from src.experiments.robustness import prepare_secondary_dataset
from src.utils.paths import enforce_local_storage, project_path, resolve_project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize and summarize a secondary dataset.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--source-dir", default="data/raw/secondary/source", help="Raw secondary dataset source dir.")
    parser.add_argument("--normalized-dir", default="data/raw/secondary/normalized", help="Normalized secondary dataset dir.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    settings = load_settings(args.config)
    root = resolve_project_root(settings.paths["root"])
    cache_dir = project_path(root, settings.paths["cache_dir"])
    enforce_local_storage(root, cache_dir)

    result = prepare_secondary_dataset(
        source_dir=project_path(root, args.source_dir),
        normalized_dir=project_path(root, args.normalized_dir),
        metrics_dir=project_path(root, settings.paths["metrics_dir"]),
        figures_dir=project_path(root, settings.paths["figures_dir"]),
        valid_extensions=tuple(ext.lower() for ext in settings.dataset["valid_extensions"]),
    )
    print(result["summary"])


if __name__ == "__main__":
    main()


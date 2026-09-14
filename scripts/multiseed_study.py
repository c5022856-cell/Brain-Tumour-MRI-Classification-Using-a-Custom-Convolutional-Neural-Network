from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings
from src.experiments.multiseed import run_multiseed_study
from src.utils.paths import enforce_local_storage, project_path, resolve_project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-seed study for one or more models.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62], help="Seeds to evaluate.")
    parser.add_argument("--models", nargs="+", default=["baseline_cnn", "tumordetnet"], help="Models to evaluate.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    settings = load_settings(args.config)
    root = resolve_project_root(settings.paths["root"])
    cache_dir = project_path(root, settings.paths["cache_dir"])
    enforce_local_storage(root, cache_dir)

    result = run_multiseed_study(
        settings=settings,
        primary_dataset_dir=project_path(root, settings.paths["primary_dataset_dir"]),
        processed_dir=project_path(root, settings.paths["processed_data_dir"]) / "multiseed",
        models_dir=project_path(root, settings.paths["models_dir"]),
        metrics_dir=project_path(root, settings.paths["metrics_dir"]) / "multiseed",
        figures_dir=project_path(root, settings.paths["figures_dir"]) / "multiseed",
        seeds=args.seeds,
        model_names=args.models,
    )
    print(result["summary_df"])


if __name__ == "__main__":
    main()

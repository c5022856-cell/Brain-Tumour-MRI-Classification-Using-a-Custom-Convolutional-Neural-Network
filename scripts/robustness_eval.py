from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings
from src.experiments.robustness import evaluate_on_secondary_dataset
from src.utils.paths import enforce_local_storage, project_path, resolve_project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a secondary dataset.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--model", required=True, choices=["baseline_cnn", "tumordetnet"], help="Model name.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--primary-results", required=True, help="Primary results JSON path.")
    parser.add_argument("--secondary-dir", default="data/raw/secondary/normalized", help="Normalized secondary dataset dir.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    settings = load_settings(args.config)
    root = resolve_project_root(settings.paths["root"])
    cache_dir = project_path(root, settings.paths["cache_dir"])
    enforce_local_storage(root, cache_dir)

    metrics_dir = project_path(root, settings.paths["metrics_dir"])
    figures_dir = project_path(root, settings.paths["figures_dir"])
    result = evaluate_on_secondary_dataset(
        settings=settings,
        model_name=args.model,
        checkpoint_path=project_path(root, args.checkpoint),
        primary_results_path=project_path(root, args.primary_results),
        secondary_dataset_dir=project_path(root, args.secondary_dir),
        output_metrics_path=metrics_dir / f"{args.model}_secondary_robustness.json",
        output_drift_csv_path=metrics_dir / f"{args.model}_secondary_drift.csv",
        output_confusion_path=figures_dir / f"{args.model}_secondary_confusion_matrix.png",
    )
    print(result["secondary_metrics"])


if __name__ == "__main__":
    main()


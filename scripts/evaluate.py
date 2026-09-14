from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings
from src.data.splits import load_splits
from src.evaluation.reporting import save_confusion_matrix, write_json
from src.models.factory import build_model
from src.training.engine import evaluate_model, resolve_device
from src.utils.paths import enforce_local_storage, project_path, resolve_project_root
from src.utils.seed import set_global_seed
from src.utils.training_io import create_dataloaders, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--model", required=True, choices=["baseline_cnn", "tumordetnet"], help="Model name.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    root = resolve_project_root(settings.paths["root"])
    cache_dir = project_path(root, settings.paths["cache_dir"])
    enforce_local_storage(root, cache_dir)
    set_global_seed(int(settings.project["seed"]))

    split_csv_path = project_path(root, settings.paths["primary_splits_csv"])
    metrics_dir = project_path(root, settings.paths["metrics_dir"])
    figures_dir = project_path(root, settings.paths["figures_dir"])

    if not split_csv_path.exists():
        raise FileNotFoundError(
            f"Missing split CSV at {split_csv_path}. Run scripts/prepare_dataset.py first."
        )

    split_df = load_splits(split_csv_path)
    _, _, test_loader, class_names, _ = create_dataloaders(split_df, settings.dataset)
    device = resolve_device(settings.training["device"])

    model = build_model(
        model_name=args.model,
        num_classes=len(class_names),
        channels=int(settings.dataset["channels"]),
        model_cfg=settings.models[args.model],
    )
    model = model.to(device)
    load_checkpoint(model, Path(args.checkpoint), device)

    test_metrics = evaluate_model(model, test_loader, device, settings.evaluation["average"], class_names)

    metrics_path = metrics_dir / f"{args.model}_evaluation.json"
    confusion_path = figures_dir / f"{args.model}_confusion_matrix.png"
    write_json(test_metrics, metrics_path)
    save_confusion_matrix(test_metrics["confusion_matrix"], class_names, confusion_path, title=f"{args.model} Test Confusion Matrix")

    print(json.dumps({"metrics_path": str(metrics_path), "confusion_path": str(confusion_path)}, indent=2))


if __name__ == "__main__":
    main()


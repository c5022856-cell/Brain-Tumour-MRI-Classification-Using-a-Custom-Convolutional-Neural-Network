from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings
from src.data.splits import load_splits
from src.evaluation.reporting import save_training_curves, write_json
from src.models.factory import build_model
from src.training.engine import compute_class_weights, evaluate_model, resolve_device, run_training
from src.utils.paths import enforce_local_storage, project_path, resolve_project_root
from src.utils.seed import set_global_seed
from src.utils.training_io import create_dataloaders, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a brain tumor MRI classification model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--model", required=True, choices=["baseline_cnn", "tumordetnet"], help="Model name.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    args = parse_args()
    settings = load_settings(args.config)
    root = resolve_project_root(settings.paths["root"])
    cache_dir = project_path(root, settings.paths["cache_dir"])
    enforce_local_storage(root, cache_dir)
    set_global_seed(int(settings.project["seed"]))

    split_csv_path = project_path(root, settings.paths["primary_splits_csv"])
    models_dir = project_path(root, settings.paths["models_dir"])
    figures_dir = project_path(root, settings.paths["figures_dir"])
    metrics_dir = project_path(root, settings.paths["metrics_dir"])

    if not split_csv_path.exists():
        raise FileNotFoundError(
            f"Missing split CSV at {split_csv_path}. Run scripts/prepare_dataset.py first."
        )

    split_df = load_splits(split_csv_path)
    train_loader, val_loader, test_loader, class_names, train_labels = create_dataloaders(split_df, settings.dataset)

    model = build_model(
        model_name=args.model,
        num_classes=len(class_names),
        channels=int(settings.dataset["channels"]),
        model_cfg=settings.models[args.model],
    )

    class_weights = None
    if bool(settings.training["use_weighted_loss"]):
        class_weights = compute_class_weights(train_labels)

    artifacts = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        training_cfg=settings.training,
        evaluation_cfg=settings.evaluation,
        class_names=class_names,
        checkpoint_dir=models_dir,
        checkpoint_stem=args.model,
        class_weights=class_weights,
    )

    curves_path = figures_dir / f"{args.model}_training_curves.png"
    save_training_curves(artifacts.history, curves_path, title=args.model)

    device = resolve_device(settings.training["device"])
    model = model.to(device)
    load_checkpoint(model, artifacts.best_checkpoint_path, device)

    val_metrics = evaluate_model(model, val_loader, device, settings.evaluation["average"], class_names)
    test_metrics = evaluate_model(model, test_loader, device, settings.evaluation["average"], class_names)

    result = {
        "model": args.model,
        "class_names": class_names,
        "best_checkpoint": str(artifacts.best_checkpoint_path),
        "history": artifacts.history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    result_path = metrics_dir / f"{args.model}_results.json"
    write_json(result, result_path)
    print(json.dumps({"result_path": str(result_path), "checkpoint": str(artifacts.best_checkpoint_path)}, indent=2))


if __name__ == "__main__":
    main()

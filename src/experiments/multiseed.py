from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.data.dataset_audit import collect_image_records
from src.data.splits import build_split_dataframe, save_splits
from src.evaluation.reporting import save_training_curves, write_json
from src.models.factory import build_model
from src.training.engine import compute_class_weights, evaluate_model, resolve_device, run_training
from src.utils.seed import set_global_seed
from src.utils.training_io import create_dataloaders, load_checkpoint

LOGGER = logging.getLogger("multiseed")


def run_multiseed_study(
    settings,
    primary_dataset_dir: Path,
    processed_dir: Path,
    models_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    seeds: list[int],
    model_names: list[str],
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    valid_extensions = tuple(ext.lower() for ext in settings.dataset["valid_extensions"])
    records = collect_image_records(primary_dataset_dir, valid_extensions)
    if not records:
        raise FileNotFoundError(f"No primary dataset records found in {primary_dataset_dir}")

    run_rows: list[dict[str, object]] = []
    summary_payload: dict[str, object] = {"runs": []}
    total_runs = len(seeds) * len(model_names)
    completed_runs = 0

    for seed in seeds:
        set_global_seed(seed)
        split_df = build_split_dataframe(
            records=records,
            test_size=float(settings.dataset["test_size"]),
            val_size=float(settings.dataset["val_size"]),
            seed=seed,
        )
        split_path = processed_dir / f"primary_splits_seed_{seed}.csv"
        save_splits(split_df, split_path)

        train_loader, val_loader, test_loader, class_names, train_labels = create_dataloaders(split_df, settings.dataset)

        for model_name in model_names:
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "run_start",
                        "seed": seed,
                        "model_name": model_name,
                        "completed_runs": completed_runs,
                        "total_runs": total_runs,
                    }
                )

            model = build_model(
                model_name=model_name,
                num_classes=len(class_names),
                channels=int(settings.dataset["channels"]),
                model_cfg=settings.models[model_name],
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
                checkpoint_dir=models_dir / "multiseed",
                checkpoint_stem=f"{model_name}_seed_{seed}",
                class_weights=class_weights,
                progress_callback=progress_callback,
                show_epoch_progress=False,
                show_batch_progress=False,
            )

            curves_path = figures_dir / f"{model_name}_seed_{seed}_training_curves.png"
            save_training_curves(artifacts.history, curves_path, title=f"{model_name} seed {seed}")

            history_df = pd.DataFrame(artifacts.history)
            history_df.insert(0, "epoch", range(1, len(history_df) + 1))
            history_path = metrics_dir / f"{model_name}_seed_{seed}_history.csv"
            history_df.to_csv(history_path, index=False)

            device = resolve_device(settings.training["device"])
            model = model.to(device)
            load_checkpoint(model, artifacts.best_checkpoint_path, device)

            val_metrics = evaluate_model(model, val_loader, device, settings.evaluation["average"], class_names)
            test_metrics = evaluate_model(model, test_loader, device, settings.evaluation["average"], class_names)

            result = {
                "model": model_name,
                "seed": seed,
                "class_names": class_names,
                "best_checkpoint": str(artifacts.best_checkpoint_path),
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
            write_json(result, metrics_dir / f"{model_name}_seed_{seed}_results.json")
            summary_payload["runs"].append(result)

            for split_name, metrics in [("validation", val_metrics), ("test", test_metrics)]:
                run_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "split": split_name,
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1_score"],
                        "roc_auc": metrics["roc_auc"],
                        "loss": metrics["loss"],
                    }
                )

            completed_runs += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "run_end",
                        "seed": seed,
                        "model_name": model_name,
                        "completed_runs": completed_runs,
                        "total_runs": total_runs,
                        "validation_metrics": val_metrics,
                        "test_metrics": test_metrics,
                    }
                )

    runs_df = pd.DataFrame(run_rows)
    runs_csv = metrics_dir / "multiseed_runs.csv"
    runs_df.to_csv(runs_csv, index=False)

    summary_df = (
        runs_df.groupby(["model", "split"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_score_mean=("f1_score", "mean"),
            f1_score_std=("f1_score", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            loss_mean=("loss", "mean"),
            loss_std=("loss", "std"),
        )
        .reset_index()
    )
    summary_csv = metrics_dir / "multiseed_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    figures_dir.mkdir(parents=True, exist_ok=True)
    for metric_name in ["accuracy", "f1_score", "roc_auc", "loss"]:
        plt.figure(figsize=(9, 5))
        plot_df = runs_df[runs_df["split"] == "test"].copy()
        sns.boxplot(data=plot_df, x="model", y=metric_name, hue="model", legend=False, palette="deep")
        sns.stripplot(data=plot_df, x="model", y=metric_name, color="black", alpha=0.6, size=4)
        plt.title(f"Multi-seed Test {metric_name.replace('_', ' ').title()}")
        plt.tight_layout()
        plot_path = figures_dir / f"multiseed_test_{metric_name}.png"
        plt.savefig(plot_path)
        plt.close()

    payload_path = metrics_dir / "multiseed_summary.json"
    write_json(summary_payload, payload_path)
    LOGGER.info("Saved multi-seed summary to %s", summary_csv)

    return {
        "runs_csv": runs_csv,
        "summary_csv": summary_csv,
        "payload_path": payload_path,
        "runs_df": runs_df,
        "summary_df": summary_df,
    }

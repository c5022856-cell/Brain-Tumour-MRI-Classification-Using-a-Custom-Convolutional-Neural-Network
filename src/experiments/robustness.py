from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_audit import collect_image_records, summarize_records
from src.data.datasets import MRIDataset
from src.data.transforms import build_transforms
from src.evaluation.reporting import save_confusion_matrix, write_json
from src.models.factory import build_model
from src.training.engine import evaluate_model, resolve_device
from src.utils.training_io import load_checkpoint

LOGGER = logging.getLogger("robustness")

LABEL_ALIASES = {
    "glioma": "glioma",
    "glioma_tumor": "glioma",
    "meningioma": "meningioma",
    "meningioma_tumor": "meningioma",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary",
    "notumor": "notumor",
    "no_tumor": "notumor",
    "no tumor": "notumor",
    "notumour": "notumor",
}


def infer_label_from_path(image_path: Path) -> str | None:
    ignored = {"training", "testing", "train", "test", "val", "validation", "source", "normalized"}
    parent_name = image_path.parent.name
    if parent_name.lower() not in ignored:
        return parent_name

    for part in reversed(image_path.parts[:-1]):
        if part.lower() not in ignored:
            return part
    return None


def canonicalize_label(label: str) -> str | None:
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    return LABEL_ALIASES.get(normalized)


def normalize_secondary_dataset(source_dir: Path, normalized_dir: Path) -> pd.DataFrame:
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [
        path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in valid_extensions
    ]
    if not image_paths:
        raise FileNotFoundError(f"No image files found under {source_dir}")

    normalized_dir.mkdir(parents=True, exist_ok=True)
    for class_dir in [path for path in normalized_dir.iterdir() if path.is_dir()]:
        for item in class_dir.rglob("*"):
            if item.is_file():
                item.unlink()
        for inner in sorted(class_dir.rglob("*"), reverse=True):
            if inner.is_dir():
                inner.rmdir()
        class_dir.rmdir()

    rows: list[dict[str, str]] = []
    for image_path in tqdm(image_paths, desc="Normalizing secondary dataset", ascii=True):
        raw_label = infer_label_from_path(image_path)
        if raw_label is None:
            continue
        label = canonicalize_label(raw_label)
        if label is None:
            continue
        destination_dir = normalized_dir / label
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / f"{image_path.parent.name}_{image_path.name}"
        destination_path.write_bytes(image_path.read_bytes())
        rows.append({"label": label, "path": str(destination_path.resolve())})

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise RuntimeError("Secondary dataset normalization produced no files.")
    return dataframe


def prepare_secondary_dataset(
    source_dir: Path,
    normalized_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    valid_extensions: tuple[str, ...],
) -> dict[str, object]:
    normalized_df = normalize_secondary_dataset(source_dir, normalized_dir)
    inventory_path = metrics_dir / "secondary_normalized_inventory.csv"
    normalized_df.to_csv(inventory_path, index=False)

    records = collect_image_records(normalized_dir, valid_extensions)
    summary = summarize_records(records)
    summary_path = metrics_dir / "secondary_dataset_summary.json"
    write_json(summary, summary_path)

    class_counts = normalized_df["label"].value_counts().sort_index()
    class_counts_df = class_counts.rename_axis("label").reset_index(name="count")
    class_counts_path = metrics_dir / "secondary_class_counts.csv"
    class_counts_df.to_csv(class_counts_path, index=False)

    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=class_counts_df, x="label", y="count", hue="label", legend=False, palette="crest")
    plt.title("Secondary Dataset Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Images")
    plt.xticks(rotation=15)
    plt.tight_layout()
    class_plot_path = figures_dir / "secondary_class_distribution.png"
    plt.savefig(class_plot_path)
    plt.close()

    return {
        "summary": summary,
        "inventory_path": inventory_path,
        "summary_path": summary_path,
        "class_counts_path": class_counts_path,
        "class_plot_path": class_plot_path,
        "normalized_dir": normalized_dir,
    }


def evaluate_on_secondary_dataset(
    settings,
    model_name: str,
    checkpoint_path: Path,
    primary_results_path: Path,
    secondary_dataset_dir: Path,
    output_metrics_path: Path,
    output_drift_csv_path: Path,
    output_confusion_path: Path,
) -> dict[str, object]:
    with primary_results_path.open("r", encoding="utf-8") as handle:
        import json

        primary_data = json.load(handle)

    class_names = primary_data["class_names"]
    label_to_index = {label: index for index, label in enumerate(class_names)}
    valid_extensions = tuple(ext.lower() for ext in settings.dataset["valid_extensions"])

    records = collect_image_records(secondary_dataset_dir, valid_extensions)
    filtered_records = []
    for record in records:
        canonical_label = canonicalize_label(str(record["label"]))
        if canonical_label is None or canonical_label not in label_to_index:
            continue
        normalized_record = dict(record)
        normalized_record["label"] = canonical_label
        filtered_records.append(normalized_record)
    if not filtered_records:
        raise ValueError("No secondary dataset records matched the primary class names.")

    dataframe = pd.DataFrame(filtered_records)
    dataframe["label_index"] = dataframe["label"].map(label_to_index)

    _, eval_transform = build_transforms(int(settings.dataset["image_size"]), settings.dataset["augment"])
    dataset = MRIDataset(dataframe, transform=eval_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=int(settings.dataset["batch_size"]),
        shuffle=False,
        num_workers=int(settings.dataset["num_workers"]),
    )

    device = resolve_device(settings.training["device"])
    model = build_model(
        model_name=model_name,
        num_classes=len(class_names),
        channels=int(settings.dataset["channels"]),
        model_cfg=settings.models[model_name],
    )
    model = model.to(device)
    load_checkpoint(model, checkpoint_path, device)

    secondary_metrics = evaluate_model(model, dataloader, device, settings.evaluation["average"], class_names)
    primary_test_metrics = primary_data["test_metrics"]

    drift_rows = []
    for metric_name in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "loss"]:
        drift_rows.append(
            {
                "model": model_name,
                "metric": metric_name,
                "primary_test": primary_test_metrics[metric_name],
                "secondary_test": secondary_metrics[metric_name],
                "delta_secondary_minus_primary": secondary_metrics[metric_name] - primary_test_metrics[metric_name],
            }
        )

    drift_df = pd.DataFrame(drift_rows)
    output_drift_csv_path.parent.mkdir(parents=True, exist_ok=True)
    drift_df.to_csv(output_drift_csv_path, index=False)

    result = {
        "model": model_name,
        "class_names": class_names,
        "secondary_metrics": secondary_metrics,
        "primary_test_metrics": primary_test_metrics,
        "secondary_dataset_dir": str(secondary_dataset_dir),
    }
    write_json(result, output_metrics_path)
    save_confusion_matrix(
        secondary_metrics["confusion_matrix"],
        class_names,
        output_confusion_path,
        title=f"{model_name} Secondary Dataset Confusion Matrix",
    )
    plt.figure(figsize=(10, 6))
    drift_plot_df = drift_df[drift_df["metric"] != "loss"].copy()
    sns.barplot(data=drift_plot_df, x="metric", y="delta_secondary_minus_primary", palette="flare")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title(f"{model_name} Secondary vs Primary Test Metric Drift")
    plt.ylabel("Secondary - Primary")
    plt.xlabel("Metric")
    plt.tight_layout()
    drift_plot_path = output_confusion_path.parent / f"{model_name}_secondary_drift.png"
    plt.savefig(drift_plot_path)
    plt.close()
    LOGGER.info("Saved secondary robustness metrics to %s", output_metrics_path)
    return result

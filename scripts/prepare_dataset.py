from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings
from src.data.dataset_audit import collect_image_records, summarize_records
from src.data.splits import build_split_dataframe, save_splits
from src.utils.paths import enforce_local_storage, project_path, resolve_project_root
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset splits and dataset summary.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    root = resolve_project_root(settings.paths["root"])
    cache_dir = project_path(root, settings.paths["cache_dir"])
    enforce_local_storage(root, cache_dir)
    set_global_seed(int(settings.project["seed"]))

    dataset_dir = project_path(root, settings.paths["primary_dataset_dir"])
    split_csv_path = project_path(root, settings.paths["primary_splits_csv"])
    metrics_dir = project_path(root, settings.paths["metrics_dir"])
    summary_path = metrics_dir / "primary_dataset_summary.json"

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Primary dataset directory does not exist: {dataset_dir}\n"
            "Place the dataset inside data/raw/primary/<class_name>/..."
        )

    valid_extensions = tuple(ext.lower() for ext in settings.dataset["valid_extensions"])
    records = collect_image_records(dataset_dir, valid_extensions)
    if not records:
        raise ValueError(f"No valid images found under {dataset_dir}")

    summary = summarize_records(records)
    split_df = build_split_dataframe(
        records=records,
        test_size=float(settings.dataset["test_size"]),
        val_size=float(settings.dataset["val_size"]),
        seed=int(settings.project["seed"]),
    )
    save_splits(split_df, split_csv_path)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved dataset summary to {summary_path}")
    print(f"Saved dataset splits to {split_csv_path}")


if __name__ == "__main__":
    main()


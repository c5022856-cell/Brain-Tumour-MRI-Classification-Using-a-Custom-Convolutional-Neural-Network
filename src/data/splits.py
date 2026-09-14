from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def build_split_dataframe(records: list[dict[str, str | int]], test_size: float, val_size: float, seed: int) -> pd.DataFrame:
    dataframe = pd.DataFrame.from_records(records)
    if dataframe.empty:
        raise ValueError("No records found to split.")

    label_names = sorted(dataframe["label"].unique())
    label_to_index = {label: index for index, label in enumerate(label_names)}
    dataframe["label_index"] = dataframe["label"].map(label_to_index)

    train_val_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=seed,
        stratify=dataframe["label"],
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=seed,
        stratify=train_val_df["label"],
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return combined.sort_values(["split", "label", "path"]).reset_index(drop=True)


def save_splits(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def load_splits(split_csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(split_csv_path)


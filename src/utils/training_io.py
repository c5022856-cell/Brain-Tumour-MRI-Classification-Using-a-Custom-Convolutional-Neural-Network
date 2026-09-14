from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.datasets import MRIDataset
from src.data.transforms import build_transforms


def create_dataloaders(split_df: pd.DataFrame, dataset_cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader, list[str], list[int]]:
    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    class_names = sorted(split_df["label"].unique().tolist())
    train_transform, eval_transform = build_transforms(int(dataset_cfg["image_size"]), dataset_cfg["augment"])

    train_dataset = MRIDataset(train_df, transform=train_transform)
    val_dataset = MRIDataset(val_df, transform=eval_transform)
    test_dataset = MRIDataset(test_df, transform=eval_transform)

    batch_size = int(dataset_cfg["batch_size"])
    num_workers = int(dataset_cfg["num_workers"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_labels = train_df["label_index"].astype(int).tolist()

    return train_loader, val_loader, test_loader, class_names, train_labels


def load_checkpoint(model, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

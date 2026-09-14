from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[index]
        image = cv2.imread(str(Path(row["path"])), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {row['path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(row["label_index"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, frame, class_names, transform): self.frame, self.class_names, self.transform = frame.reset_index(drop=True), class_names, transform
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image: image = image.convert('RGB')
        return self.transform(image), self.class_names.index(row.label)

def create_dataloaders(split_df: pd.DataFrame, dataset_cfg: dict):
    classes = sorted(split_df.label.unique().tolist()); size = int(dataset_cfg['image_size'])
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    batch_size = int(dataset_cfg['batch_size']); workers = int(dataset_cfg.get('num_workers', 0))
    loaders = []
    for split in ('train', 'validation', 'test'):
        frame = split_df[split_df.split.eq(split)]
        loaders.append(DataLoader(ImageDataset(frame, classes, transform), batch_size=batch_size, shuffle=split == 'train', num_workers=workers))
    return *loaders, classes, split_df[split_df.split.eq('train')].label.tolist()

def load_checkpoint(model, checkpoint_path, device):
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(payload['model_state_dict'] if isinstance(payload, dict) else payload)
    return model


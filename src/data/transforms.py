from __future__ import annotations

from torchvision import transforms


def build_transforms(image_size: int, augment_cfg: dict[str, float | bool]) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transforms = [
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
    ]

    if augment_cfg.get("horizontal_flip", False):
        train_transforms.append(transforms.RandomHorizontalFlip())

    rotation_degrees = float(augment_cfg.get("rotation_degrees", 0))
    if rotation_degrees > 0:
        train_transforms.append(transforms.RandomRotation(rotation_degrees))

    brightness = float(augment_cfg.get("brightness", 0))
    contrast = float(augment_cfg.get("contrast", 0))
    if brightness > 0 or contrast > 0:
        train_transforms.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))

    train_transforms.extend([transforms.ToTensor(), normalize])

    eval_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return transforms.Compose(train_transforms), eval_transforms


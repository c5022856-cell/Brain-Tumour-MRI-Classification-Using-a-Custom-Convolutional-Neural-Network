from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def collect_image_records(dataset_dir: Path, valid_extensions: tuple[str, ...]) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    for class_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        label = class_dir.name
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in valid_extensions:
                continue
            width, height = get_image_size(image_path)
            records.append(
                {
                    "label": label,
                    "path": str(image_path.resolve()),
                    "width": width,
                    "height": height,
                }
            )
    return records


def get_image_size(image_path: Path) -> tuple[int, int]:
    try:
        with Image.open(image_path) as image:
            return image.size
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or corrupted image file: {image_path}") from exc


def summarize_records(records: list[dict[str, str | int]]) -> dict[str, object]:
    widths = [int(record["width"]) for record in records]
    heights = [int(record["height"]) for record in records]
    labels = [str(record["label"]) for record in records]
    label_counts = Counter(labels)

    return {
        "num_images": len(records),
        "num_classes": len(label_counts),
        "classes": dict(sorted(label_counts.items())),
        "image_width_range": [min(widths), max(widths)] if widths else [0, 0],
        "image_height_range": [min(heights), max(heights)] if heights else [0, 0],
    }


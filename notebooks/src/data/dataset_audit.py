from pathlib import Path

def collect_image_records(dataset_dir: str | Path, valid_extensions: tuple[str, ...]) -> list[dict]:
    root = Path(dataset_dir)
    return [
        {'path': str(path), 'label': path.parent.name, 'bytes': path.stat().st_size}
        for path in sorted(root.rglob('*'))
        if path.is_file() and path.suffix.lower() in valid_extensions
    ]

def summarize_records(records: list[dict]) -> dict:
    labels = sorted({record['label'] for record in records})
    counts = {label: sum(row['label'] == label for row in records) for label in labels}
    return {'total_images': len(records), 'classes': labels, 'class_counts': counts}


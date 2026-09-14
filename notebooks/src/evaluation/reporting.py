from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

def write_json(value, path: str | Path) -> None:
    def encode(item):
        if hasattr(item, 'tolist'): return item.tolist()
        if hasattr(item, 'item'): return item.item()
        if isinstance(item, Path): return str(item)
        raise TypeError(f'Not JSON serializable: {type(item).__name__}')
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, default=encode), encoding='utf-8')

def save_training_curves(history, path: str | Path, title='Training curves') -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 5))
    for key in ('train_loss', 'val_loss', 'test_loss'):
        values = [row[key] for row in history if key in row]
        if values: axis.plot(range(1, len(values) + 1), values, label=key.replace('_', ' '))
    axis.set(title=title, xlabel='Epoch', ylabel='Loss'); axis.legend(); fig.tight_layout(); fig.savefig(destination); plt.close(fig)

def save_confusion_matrix(matrix, class_names, path: str | Path, title='Confusion matrix') -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 6)); sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axis)
    axis.set(title=title, xlabel='Predicted', ylabel='Actual'); fig.tight_layout(); fig.savefig(destination); plt.close(fig)


"""Create reproducible train/validation/test CSV splits from class folders."""
import argparse
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import load_settings

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); args = parser.parse_args()
    settings = load_settings(args.config); root = Path(settings.paths['root']).resolve()
    source = root / settings.paths['primary_dataset_dir']; extensions = tuple(settings.dataset['valid_extensions'])
    rows = [{'path': str(path), 'label': path.parent.name} for path in source.rglob('*') if path.is_file() and path.suffix.lower() in extensions]
    if not rows: raise FileNotFoundError(f'No images found in {source}')
    data = pd.DataFrame(rows); seed = int(settings.project['seed']); test_size = 1 - float(settings.dataset['train_fraction'])
    train, holdout = train_test_split(data, test_size=test_size, random_state=seed, stratify=data.label)
    validation_share = float(settings.dataset['validation_fraction']) / test_size
    validation, test = train_test_split(holdout, test_size=1-validation_share, random_state=seed, stratify=holdout.label)
    result = pd.concat([train.assign(split='train'), validation.assign(split='validation'), test.assign(split='test')], ignore_index=True)
    destination = root / settings.paths['processed_dir'] / 'primary_splits.csv'; destination.parent.mkdir(parents=True, exist_ok=True); result.to_csv(destination, index=False); print(f'Wrote {len(result)} rows to {destination}')
if __name__ == '__main__': main()


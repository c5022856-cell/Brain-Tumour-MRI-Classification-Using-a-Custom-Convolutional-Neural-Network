from pathlib import Path
import pandas as pd

def load_splits(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'path', 'label', 'split'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Split file is missing columns: {sorted(missing)}')
    return df


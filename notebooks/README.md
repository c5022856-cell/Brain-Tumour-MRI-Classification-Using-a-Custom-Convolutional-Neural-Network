# Brain Tumour MRI Classification

Research project for training and evaluating MRI brain-tumour classifiers. The notebooks contain the primary training pipeline and a secondary validation/multi-seed study.

## Quick start

1. Create and activate a Python 3.10+ virtual environment.
2. Install the project and its dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -e .
   ```

3. Select the environment as the `Python 3` kernel in Jupyter.
4. If you use the dataset-download cells, put Kaggle credentials at `developer-only/kaggle/kaggle.json`.
5. Run `01_primary_pipeline_brain_tumor_mri.ipynb` before the secondary-validation notebook.

Project data is stored under `data/`; models, metrics, figures, and caches are stored under `outputs/`. These paths are intentionally ignored by Git.

## Layout

```text
src/            Reusable pipeline code
scripts/        Command-line workflow scripts
notebooks/      Experiment notebooks (this repository's root currently contains them)
data/           Downloaded and processed datasets
outputs/        Model checkpoints and reports
tests/          Automated tests
```


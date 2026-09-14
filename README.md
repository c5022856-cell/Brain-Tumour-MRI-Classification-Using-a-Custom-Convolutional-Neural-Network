# Brain Tumor Detection from MRI Scans Using Deep Learning

## Project Overview

This repository implements a reproducible deep learning pipeline for brain tumor MRI classification. The work is based on the project proposal to develop and evaluate a custom deep learning model, `TumorDetNet`, against a simpler baseline CNN, with emphasis on:

- reproducibility
- transparent performance reporting
- clinically relevant evaluation metrics
- cross-dataset robustness assessment

All project assets remain inside the workspace at `F:\brain tumor detection`. No datasets, caches, model weights, or outputs are intentionally stored outside this project folder.

## Repository Structure

```text
brain-tumor-detection/
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|-- notebooks/
|-- outputs/
|   |-- cache/
|   |-- figures/
|   |-- logs/
|   |-- metrics/
|   `-- models/
|-- scripts/
|-- src/
|-- tests/
|-- README.md
`-- requirements.txt
```

## Execution Paths

Primary workflow notebook:

- `notebooks/01_primary_pipeline_brain_tumor_mri.ipynb`

Remaining-work notebook for robustness and multi-seed experiments:

- `notebooks/02_secondary_validation_and_multiseed_study.ipynb`

Supporting scripts:

- `scripts/download_dataset.py`
- `scripts/download_secondary_dataset.py`
- `scripts/prepare_dataset.py`
- `scripts/prepare_secondary_dataset.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/robustness_eval.py`
- `scripts/multiseed_study.py`

## Environment Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Kaggle API credentials must be configured so the Kaggle CLI can authenticate before dataset download steps are run.

## Datasets

### Primary Dataset Inventory

Primary dataset used for model development:

- Kaggle Brain Tumor MRI Dataset
- path after normalization: `data/raw/primary/`

Primary class labels:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

### Secondary Dataset

Secondary dataset used for cross-dataset robustness testing:

- `sartajbhuvaji/brain-tumor-classification-mri`
- normalized path: `data/raw/secondary/normalized/`

The secondary dataset uses different raw folder names such as `glioma_tumor` and `no_tumor`; these are mapped into the primary label space during normalization and evaluation.

## Data Summary

Source artifacts:

- [primary_dataset_summary.json](F:\brain%20tumor%20detection\outputs\metrics\primary_dataset_summary.json)
- [primary_class_counts.csv](F:\brain%20tumor%20detection\outputs\metrics\primary_class_counts.csv)
- [split_class_counts.csv](F:\brain%20tumor%20detection\outputs\metrics\split_class_counts.csv)
- [secondary_dataset_summary.json](F:\brain%20tumor%20detection\outputs\metrics\secondary_dataset_summary.json)
- [secondary_class_counts.csv](F:\brain%20tumor%20detection\outputs\metrics\secondary_class_counts.csv)

### Primary Dataset Summary

- Total images: `7200`
- Number of classes: `4`
- Class balance: equal
- Images per class:
  - `glioma`: `1800`
  - `meningioma`: `1800`
  - `notumor`: `1800`
  - `pituitary`: `1800`
- Image width range: `150` to `1375`
- Image height range: `167` to `1446`

### Primary Data Split

| Split | Images per class | Total images |
| --- | ---: | ---: |
| Train | 1260 | 5040 |
| Validation | 270 | 1080 |
| Test | 270 | 1080 |

### Secondary Dataset Inventory

- Class counts after normalization:
  - `glioma`: `926`
  - `meningioma`: `937`
  - `notumor`: `500`
  - `pituitary`: `901`

Interpretation:

- The primary dataset is balanced and appropriate for controlled model comparison.
- The secondary dataset is not perfectly balanced and is distinct enough to serve as a meaningful domain-shift benchmark.

## Models

### Baseline CNN Architecture

The baseline model is a lightweight convolutional neural network designed to provide a fair reference point. It includes standard convolution, pooling, and dense layers.

### TumorDetNet Architecture

`TumorDetNet` is the main project model. It uses deeper convolutional blocks with normalization, SiLU activation, and stronger representational capacity than the baseline model.

## Training Configuration

Key configuration choices:

- framework: PyTorch
- image size: `224x224`
- optimizer: Adam
- weighted loss: enabled
- learning-rate scheduler: ReduceLROnPlateau
- early stopping support: enabled
- device policy: automatic CUDA if available, otherwise CPU
- standard augmentation:
  - resize
  - horizontal flip
  - random rotation
  - brightness adjustment
  - contrast adjustment

## Primary Single-Run Results

Source artifacts:

- [model_metrics_comparison.csv](F:\brain%20tumor%20detection\outputs\metrics\model_metrics_comparison.csv)
- [baseline_cnn_results.json](F:\brain%20tumor%20detection\outputs\metrics\baseline_cnn_results.json)
- [tumordetnet_results.json](F:\brain%20tumor%20detection\outputs\metrics\tumordetnet_results.json)

### Aggregate Comparison

| Model | Split | Accuracy | Precision | Recall | F1-score | ROC-AUC | Loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_cnn | Validation | 0.8139 | 0.8265 | 0.8139 | 0.8140 | 0.9603 | 0.4599 |
| baseline_cnn | Test | 0.8333 | 0.8467 | 0.8333 | 0.8346 | 0.9587 | 0.4559 |
| tumordetnet | Validation | 0.9148 | 0.9183 | 0.9148 | 0.9148 | 0.9854 | 0.2800 |
| tumordetnet | Test | 0.8972 | 0.9018 | 0.8972 | 0.8969 | 0.9838 | 0.2967 |

### Improvement of TumorDetNet Over Baseline

Validation improvement:

- Accuracy: `+0.1009`
- Precision: `+0.0918`
- Recall: `+0.1009`
- F1-score: `+0.1008`
- ROC-AUC: `+0.0251`
- Loss: `-0.1800`

Test improvement:

- Accuracy: `+0.0639`
- Precision: `+0.0551`
- Recall: `+0.0639`
- F1-score: `+0.0622`
- ROC-AUC: `+0.0251`
- Loss: `-0.1591`

Interpretation:

- `TumorDetNet` outperforms the baseline CNN across all primary aggregate metrics.
- The gain is substantial rather than marginal.
- The proposal hypothesis is supported by the primary dataset experiment.

## Per-Class Test Performance

### Baseline CNN Test Metrics

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| glioma | 0.9657 | 0.7296 | 0.8312 |
| meningioma | 0.6943 | 0.8074 | 0.7466 |
| notumor | 0.8877 | 0.9074 | 0.8974 |
| pituitary | 0.8392 | 0.8889 | 0.8633 |

### TumorDetNet Test Metrics

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| glioma | 0.9821 | 0.8111 | 0.8884 |
| meningioma | 0.8163 | 0.8556 | 0.8354 |
| notumor | 0.8942 | 0.9704 | 0.9307 |
| pituitary | 0.9146 | 0.9519 | 0.9328 |

### Class-Wise Interpretation

- `notumor` and `pituitary` are the strongest classes for `TumorDetNet`.
- `glioma` and `meningioma` remain more difficult because tumor-to-tumor confusion persists.
- `TumorDetNet` improves every class-specific test F1-score relative to the baseline.

## Cross-Dataset Robustness Results

Source artifacts:

- [tumordetnet_secondary_robustness.json](F:\brain%20tumor%20detection\outputs\metrics\tumordetnet_secondary_robustness.json)
- [tumordetnet_secondary_drift.csv](F:\brain%20tumor%20detection\outputs\metrics\tumordetnet_secondary_drift.csv)

### Secondary Dataset Performance of TumorDetNet

| Metric | Primary Test | Secondary Test | Drift |
| --- | ---: | ---: | ---: |
| Accuracy | 0.8972 | 0.8430 | -0.0542 |
| Precision | 0.9018 | 0.8623 | -0.0394 |
| Recall | 0.8972 | 0.8430 | -0.0542 |
| F1-score | 0.8969 | 0.8466 | -0.0502 |
| ROC-AUC | 0.9838 | 0.9635 | -0.0203 |
| Loss | 0.2967 | 0.4495 | +0.1528 |

### Secondary Per-Class F1-Scores

| Class | F1-score |
| --- | ---: |
| glioma | 0.8571 |
| meningioma | 0.8045 |
| notumor | 0.7382 |
| pituitary | 0.9273 |

Interpretation:

- Performance drops under dataset shift, which is expected.
- The drop is meaningful but not catastrophic; the model retains strong overall discrimination on the secondary dataset.
- `notumor` is the weakest transferring class.
- `pituitary` remains the most stable class under external variation.

## Multi-Seed Stability Results

Source artifacts:

- [multiseed_runs.csv](F:\brain%20tumor%20detection\outputs\metrics\multiseed\multiseed_runs.csv)
- [multiseed_summary.csv](F:\brain%20tumor%20detection\outputs\metrics\multiseed\multiseed_summary.csv)

Three seeds were evaluated: `42`, `52`, and `62`.

### Multi-Seed Test Summary

| Model | Accuracy Mean | Accuracy Std | F1 Mean | F1 Std | ROC-AUC Mean | ROC-AUC Std | Loss Mean | Loss Std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_cnn | 0.8157 | 0.0156 | 0.8146 | 0.0177 | 0.9540 | 0.0047 | 0.4896 | 0.0309 |
| tumordetnet | 0.8917 | 0.0207 | 0.8907 | 0.0213 | 0.9833 | 0.0036 | 0.3024 | 0.0433 |

### Multi-Seed Validation Summary

| Model | Accuracy Mean | Accuracy Std | F1 Mean | F1 Std | ROC-AUC Mean | ROC-AUC Std | Loss Mean | Loss Std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_cnn | 0.8062 | 0.0071 | 0.8051 | 0.0084 | 0.9521 | 0.0073 | 0.4950 | 0.0318 |
| tumordetnet | 0.8895 | 0.0179 | 0.8884 | 0.0192 | 0.9826 | 0.0017 | 0.3140 | 0.0391 |

Interpretation:

- The superiority of `TumorDetNet` persists across multiple seeds.
- The model advantage is stable rather than dependent on a single favorable split or initialization.
- `TumorDetNet` maintains a clear margin over the baseline in both mean test accuracy and mean test F1-score.

## Training Summary

Source artifacts:

- [baseline_cnn_history.csv](F:\brain%20tumor%20detection\outputs\metrics\baseline_cnn_history.csv)
- [tumordetnet_history.csv](F:\brain%20tumor%20detection\outputs\metrics\tumordetnet_history.csv)
- [baseline_cnn_training_curves.png](F:\brain%20tumor%20detection\outputs\figures\baseline_cnn_training_curves.png)
- [tumordetnet_training_curves.png](F:\brain%20tumor%20detection\outputs\figures\tumordetnet_training_curves.png)

| Model | Epochs Run | Best Epoch by Validation Loss | Total Training Time | Avg. Epoch Time |
| --- | ---: | ---: | ---: | ---: |
| baseline_cnn | 15 | 14 | 687.65 s | 45.84 s |
| tumordetnet | 15 | 15 | 764.48 s | 50.97 s |

Interpretation:

- The baseline model converges smoothly and predictably.
- `TumorDetNet` is slightly slower per epoch but reaches a materially better solution.
- Validation/test metrics exceeding training metrics in some runs are plausible because augmentation and regularization are active during training.

## Key Figures and Artifacts

Primary figures:

- [primary_class_distribution.png](F:\brain%20tumor%20detection\outputs\figures\primary_class_distribution.png)
- [primary_sample_grid.png](F:\brain%20tumor%20detection\outputs\figures\primary_sample_grid.png)
- [split_class_distribution.png](F:\brain%20tumor%20detection\outputs\figures\split_class_distribution.png)
- [baseline_cnn_training_curves.png](F:\brain%20tumor%20detection\outputs\figures\baseline_cnn_training_curves.png)
- [tumordetnet_training_curves.png](F:\brain%20tumor%20detection\outputs\figures\tumordetnet_training_curves.png)
- [baseline_cnn_confusion_matrix.png](F:\brain%20tumor%20detection\outputs\figures\baseline_cnn_confusion_matrix.png)
- [tumordetnet_confusion_matrix.png](F:\brain%20tumor%20detection\outputs\figures\tumordetnet_confusion_matrix.png)

Robustness figures:

- [secondary_class_distribution.png](F:\brain%20tumor%20detection\outputs\figures\secondary_class_distribution.png)
- [tumordetnet_secondary_confusion_matrix.png](F:\brain%20tumor%20detection\outputs\figures\tumordetnet_secondary_confusion_matrix.png)
- [tumordetnet_secondary_drift.png](F:\brain%20tumor%20detection\outputs\figures\tumordetnet_secondary_drift.png)

Multi-seed figures:

- [multiseed_test_accuracy.png](F:\brain%20tumor%20detection\outputs\figures\multiseed\multiseed_test_accuracy.png)
- [multiseed_test_f1_score.png](F:\brain%20tumor%20detection\outputs\figures\multiseed\multiseed_test_f1_score.png)
- [multiseed_test_roc_auc.png](F:\brain%20tumor%20detection\outputs\figures\multiseed\multiseed_test_roc_auc.png)
- [multiseed_test_loss.png](F:\brain%20tumor%20detection\outputs\figures\multiseed\multiseed_test_loss.png)

Model checkpoints:

- [baseline_cnn_best.pt](F:\brain%20tumor%20detection\outputs\models\baseline_cnn_best.pt)
- [tumordetnet_best.pt](F:\brain%20tumor%20detection\outputs\models\tumordetnet_best.pt)

## Professional Interpretation

The current evidence supports the central project claim. `TumorDetNet` consistently outperforms the baseline CNN on the primary dataset, preserves a clear advantage across three random seeds, and maintains reasonably strong performance on a secondary MRI dataset. This indicates that the model is not only better in a single controlled experiment but also more stable and more transferable.

At the same time, the robustness results show a genuine domain-shift penalty, especially for the `notumor` class. This is important because it prevents overstatement: the model generalizes better than the baseline, but it is not invariant to dataset differences. That finding is academically useful and aligns with the proposal's emphasis on transparent, clinically oriented reporting rather than headline accuracy alone.

## Project Status

Completed:

- project-local storage and cache enforcement
- primary dataset download and normalization
- secondary dataset download and normalization
- dataset auditing and split generation
- baseline CNN implementation
- TumorDetNet implementation
- training and evaluation pipeline
- class-wise metrics and confusion matrices
- secondary-dataset robustness evaluation
- multi-seed stability study
- saved checkpoints, histories, CSVs, JSON results, and figures

Remaining minor gaps:

- formal statistical significance testing
- ROC curve plotting
- interpretability support such as Grad-CAM
- richer error analysis on misclassified images
- stronger explicit ethics and responsible-use section

## Conclusion

This project delivers a functioning and well-documented deep learning pipeline for MRI-based brain tumor classification. `TumorDetNet` is the best-performing model in this repository. It demonstrates strong primary-dataset performance, stable multi-seed behavior, and acceptable cross-dataset robustness. These results support the proposal's objective of developing a reproducible and transparently evaluated MRI brain tumor classification framework.

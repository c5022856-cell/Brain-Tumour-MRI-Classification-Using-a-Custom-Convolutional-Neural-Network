from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.evaluation.metrics import compute_classification_metrics

LOGGER = logging.getLogger("training")


@dataclass(slots=True)
class TrainingArtifacts:
    history: dict[str, list[float]]
    best_checkpoint_path: Path


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    classes, counts = np.unique(labels, return_counts=True)
    weights = len(labels) / (len(classes) * counts)
    tensor = torch.ones(int(classes.max()) + 1, dtype=torch.float32)
    for class_index, weight in zip(classes, weights):
        tensor[int(class_index)] = float(weight)
    return tensor


def run_training(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    training_cfg: dict,
    evaluation_cfg: dict,
    class_names: list[str],
    checkpoint_dir: Path,
    checkpoint_stem: str,
    class_weights: torch.Tensor | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    show_epoch_progress: bool = True,
    show_batch_progress: bool = False,
) -> TrainingArtifacts:
    device = resolve_device(training_cfg["device"])
    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "test_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1_score": [],
        "train_roc_auc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1_score": [],
        "val_roc_auc": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1_score": [],
        "test_roc_auc": [],
        "epoch_seconds": [],
    }

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / f"{checkpoint_stem}_best.pt"
    latest_checkpoint_path = checkpoint_dir / f"{checkpoint_stem}_latest.pt"

    best_val_loss = float("inf")
    patience_counter = 0

    total_epochs = int(training_cfg["epochs"])
    LOGGER.info("Training %s for up to %s epochs on device %s", checkpoint_stem, total_epochs, device)

    epoch_iterator = range(total_epochs)
    if show_epoch_progress:
        epoch_iterator = tqdm(epoch_iterator, desc=f"Epochs[{checkpoint_stem}]", ascii=True)

    for epoch in epoch_iterator:
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "epoch_start",
                    "epoch": epoch + 1,
                    "total_epochs": total_epochs,
                    "model_name": checkpoint_stem,
                }
            )
        epoch_start = perf_counter()
        train_loss, train_accuracy, train_true, train_pred, train_prob = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            phase_name="train",
            epoch_index=epoch + 1,
            total_epochs=total_epochs,
            progress_callback=progress_callback,
            show_batch_progress=show_batch_progress,
        )
        val_loss, val_accuracy, val_true, val_pred, val_prob = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            train=False,
            phase_name="val",
            epoch_index=epoch + 1,
            total_epochs=total_epochs,
            progress_callback=progress_callback,
            show_batch_progress=show_batch_progress,
        )
        test_loss, test_accuracy, test_true, test_pred, test_prob = run_epoch(
            model,
            test_loader,
            criterion,
            optimizer=None,
            device=device,
            train=False,
            phase_name="test",
            epoch_index=epoch + 1,
            total_epochs=total_epochs,
            progress_callback=progress_callback,
            show_batch_progress=show_batch_progress,
        )
        scheduler.step(val_loss)

        train_metrics = compute_classification_metrics(
            train_true, train_pred, train_prob, class_names, evaluation_cfg["average"]
        )
        val_metrics = compute_classification_metrics(
            val_true, val_pred, val_prob, class_names, evaluation_cfg["average"]
        )
        test_metrics = compute_classification_metrics(
            test_true, test_pred, test_prob, class_names, evaluation_cfg["average"]
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["test_accuracy"].append(test_accuracy)
        history["train_precision"].append(train_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["train_f1_score"].append(train_metrics["f1_score"])
        history["train_roc_auc"].append(train_metrics["roc_auc"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1_score"].append(val_metrics["f1_score"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        history["test_precision"].append(test_metrics["precision"])
        history["test_recall"].append(test_metrics["recall"])
        history["test_f1_score"].append(test_metrics["f1_score"])
        history["test_roc_auc"].append(test_metrics["roc_auc"])
        history["epoch_seconds"].append(perf_counter() - epoch_start)

        LOGGER.info(
            "Epoch %s/%s | train loss %.4f acc %.4f prec %.4f rec %.4f f1 %.4f auc %.4f | "
            "val loss %.4f acc %.4f prec %.4f rec %.4f f1 %.4f auc %.4f | "
            "test loss %.4f acc %.4f prec %.4f rec %.4f f1 %.4f auc %.4f",
            epoch + 1,
            total_epochs,
            train_loss,
            train_accuracy,
            train_metrics["precision"],
            train_metrics["recall"],
            train_metrics["f1_score"],
            train_metrics["roc_auc"],
            val_loss,
            val_accuracy,
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1_score"],
            val_metrics["roc_auc"],
            test_loss,
            test_accuracy,
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1_score"],
            test_metrics["roc_auc"],
        )

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "epoch_end",
                    "epoch": epoch + 1,
                    "total_epochs": total_epochs,
                    "model_name": checkpoint_stem,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "test_loss": test_loss,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                    "test_accuracy": test_accuracy,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                }
            )

        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1}, latest_checkpoint_path)

        if val_loss + float(training_cfg["min_delta"]) < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1}, best_checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= int(training_cfg["patience"]):
            LOGGER.info("Early stopping triggered at epoch %s", epoch + 1)
            break

    return TrainingArtifacts(history=history, best_checkpoint_path=best_checkpoint_path)


def evaluate_model(model: nn.Module, data_loader, device: torch.device, average: str, class_names: list[str]) -> dict[str, object]:
    criterion = nn.CrossEntropyLoss()
    loss, accuracy, y_true, y_pred, y_prob = run_epoch(
        model,
        data_loader,
        criterion,
        optimizer=None,
        device=device,
        train=False,
        phase_name="eval",
        epoch_index=0,
        total_epochs=0,
        progress_callback=None,
        show_batch_progress=False,
    )
    metrics = compute_classification_metrics(y_true, y_pred, y_prob, class_names, average)
    metrics["loss"] = float(loss)
    metrics["accuracy_from_epoch"] = float(accuracy)
    return metrics


def run_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    train: bool,
    phase_name: str,
    epoch_index: int,
    total_epochs: int,
    progress_callback: Callable[[dict[str, object]], None] | None,
    show_batch_progress: bool,
):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    total_examples = 0
    correct_predictions = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob_batches: list[np.ndarray] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        total_batches = len(data_loader)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "phase_start",
                    "phase": phase_name,
                    "epoch": epoch_index,
                    "total_epochs": total_epochs,
                    "total_batches": total_batches,
                }
            )

        iterator = data_loader
        if show_batch_progress:
            iterator = tqdm(
                data_loader,
                desc=phase_name,
                leave=False,
                ascii=True,
            )

        for batch_index, (inputs, targets) in enumerate(iterator, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            probabilities = torch.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_examples += batch_size
            correct_predictions += (predictions == targets).sum().item()

            y_true.extend(targets.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            y_prob_batches.append(probabilities.detach().cpu().numpy())

            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "phase_progress",
                        "phase": phase_name,
                        "epoch": epoch_index,
                        "total_epochs": total_epochs,
                        "batch": batch_index,
                        "total_batches": total_batches,
                    }
                )

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "phase_end",
                    "phase": phase_name,
                    "epoch": epoch_index,
                    "total_epochs": total_epochs,
                }
            )

    average_loss = running_loss / max(total_examples, 1)
    accuracy = correct_predictions / max(total_examples, 1)
    y_prob = np.concatenate(y_prob_batches, axis=0) if y_prob_batches else np.empty((0, 0))
    return average_loss, accuracy, y_true, y_pred, y_prob

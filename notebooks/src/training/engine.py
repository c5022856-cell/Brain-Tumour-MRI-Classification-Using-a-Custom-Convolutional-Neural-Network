from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

@dataclass
class TrainingArtifacts:
    history: list[dict]
    best_checkpoint_path: Path

def resolve_device(value='auto'):
    return torch.device('cuda' if value == 'auto' and torch.cuda.is_available() else ('cpu' if value == 'auto' else value))

def compute_class_weights(labels):
    _, counts = np.unique(labels, return_counts=True)
    return torch.tensor(len(labels) / (len(counts) * counts), dtype=torch.float32)

def evaluate_model(model, loader, device, average, class_names):
    model.eval(); losses = []; predicted = []; actual = []; probabilities = []; criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device); logits = model(images)
            losses.append(criterion(logits, labels).item()); predicted.extend(logits.argmax(1).cpu().tolist()); actual.extend(labels.cpu().tolist()); probabilities.extend(torch.softmax(logits, 1).cpu().tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average=average, zero_division=0)
    try: auc = roc_auc_score(actual, probabilities, multi_class='ovr', average=average)
    except ValueError: auc = float('nan')
    return {'loss': float(np.mean(losses)) if losses else float('nan'), 'accuracy': accuracy_score(actual, predicted) if actual else float('nan'), 'precision': precision, 'recall': recall, 'f1_score': f1, 'roc_auc': auc, 'confusion_matrix': confusion_matrix(actual, predicted, labels=range(len(class_names))).tolist()}

def run_training(model, train_loader, val_loader, test_loader, training_cfg, evaluation_cfg, class_names, checkpoint_dir, checkpoint_stem, class_weights=None, progress_callback=None, show_epoch_progress=False, show_batch_progress=False):
    device = resolve_device(training_cfg.get('device', 'auto')); model.to(device)
    weights = class_weights.to(device) if class_weights is not None else None; criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg['learning_rate']), weight_decay=float(training_cfg.get('weight_decay', 0)))
    checkpoint = Path(checkpoint_dir) / f'{checkpoint_stem}_best.pt'; checkpoint.parent.mkdir(parents=True, exist_ok=True); history = []; best_loss = float('inf'); epochs = int(training_cfg['epochs'])
    for epoch in range(1, epochs + 1):
        if progress_callback: progress_callback({'event': 'epoch_start', 'epoch': epoch, 'total_epochs': epochs})
        model.train(); losses = []; hits = total = 0
        if progress_callback: progress_callback({'event': 'phase_start', 'phase': 'train', 'epoch': epoch, 'total_batches': len(train_loader)})
        for batch, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device); optimizer.zero_grad(); logits = model(images); loss = criterion(logits, labels); loss.backward(); optimizer.step(); losses.append(loss.item()); hits += (logits.argmax(1) == labels).sum().item(); total += len(labels)
            if progress_callback: progress_callback({'event': 'phase_progress', 'batch': batch})
        if progress_callback: progress_callback({'event': 'phase_end'})
        train_metrics = evaluate_model(model, train_loader, device, evaluation_cfg['average'], class_names); val_metrics = evaluate_model(model, val_loader, device, evaluation_cfg['average'], class_names); test_metrics = evaluate_model(model, test_loader, device, evaluation_cfg['average'], class_names)
        row = {'train_loss': float(np.mean(losses)), 'train_accuracy': hits / total, 'val_loss': val_metrics['loss'], 'val_accuracy': val_metrics['accuracy'], 'test_loss': test_metrics['loss'], 'test_accuracy': test_metrics['accuracy']}; history.append(row)
        if val_metrics['loss'] < best_loss: best_loss = val_metrics['loss']; torch.save({'model_state_dict': model.state_dict()}, checkpoint)
        if progress_callback: progress_callback({'event': 'epoch_end', 'epoch': epoch, 'total_epochs': epochs, **row, 'train_metrics': train_metrics, 'val_metrics': val_metrics, 'test_metrics': test_metrics})
    return TrainingArtifacts(history, checkpoint)


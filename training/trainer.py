from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Subset

from config.config import Config
from data.data_loader import BrainConnectivityDataset
from models.model import GLGAN
from utils.seed import set_seed


@dataclass(frozen=True)
class CrossValidationResult:
    fold_metrics: list[dict[str, float]]
    aggregate_metrics: dict[str, float]


def run_cross_validation(
    fmri: np.ndarray,
    dti: np.ndarray,
    labels: np.ndarray,
    config: Config,
) -> CrossValidationResult:
    set_seed(config.random_seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.save_checkpoints:
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    labels = labels.astype("int64")
    splitter = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.random_seed)
    fold_metrics: list[dict[str, float]] = []

    for fold, (train_index, test_index) in enumerate(splitter.split(fmri, labels), start=1):
        dataset = BrainConnectivityDataset(fmri, dti, labels)
        train_loader = DataLoader(Subset(dataset, train_index), batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_index), batch_size=config.batch_size, shuffle=False)

        model = GLGAN(config).to(config.torch_device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        best_metrics: dict[str, float] | None = None
        best_accuracy = -1.0
        for _ in range(config.epochs):
            _train_one_epoch(model, train_loader, criterion, optimizer, config)
            metrics = evaluate(model, test_loader, criterion, config)
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_metrics = metrics
                if config.save_checkpoints:
                    torch.save(model.state_dict(), output_dir / "checkpoints" / f"fold_{fold}_best.pt")

        assert best_metrics is not None
        best_metrics = {"fold": float(fold), **best_metrics}
        fold_metrics.append(best_metrics)

    aggregate_metrics = _aggregate_metrics(fold_metrics)
    _write_metrics_csv(output_dir / "fold_metrics.csv", fold_metrics)
    _write_aggregate_csv(output_dir / "aggregate_metrics.csv", aggregate_metrics)
    return CrossValidationResult(fold_metrics=fold_metrics, aggregate_metrics=aggregate_metrics)


def evaluate(model: GLGAN, loader: DataLoader, criterion: nn.Module, config: Config) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    targets: list[int] = []

    with torch.no_grad():
        for fmri, dti, labels in loader:
            fmri = fmri.to(config.torch_device)
            dti = dti.to(config.torch_device)
            labels = labels.to(config.torch_device)
            logits, _ = model(fmri, dti)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            losses.append(float(loss.detach().cpu()))
            predictions.extend(torch.argmax(probs, dim=1).cpu().numpy().tolist())
            probabilities.extend(probs[:, 1].cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())

    return _classification_metrics(targets, predictions, probabilities, float(np.mean(losses)))


def _train_one_epoch(
    model: GLGAN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
) -> None:
    model.train()
    for fmri, dti, labels in loader:
        fmri = fmri.to(config.torch_device)
        dti = dti.to(config.torch_device)
        labels = labels.to(config.torch_device)
        optimizer.zero_grad()
        logits, _ = model(fmri, dti)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()


def _classification_metrics(
    targets: list[int], predictions: list[int], probabilities: list[float], loss: float
) -> dict[str, float]:
    labels = np.asarray(targets)
    preds = np.asarray(predictions)
    probs = np.asarray(probabilities)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0
    return {
        "loss": loss,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "specificity": specificity,
        "sensitivity": sensitivity,
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": auc,
    }


def _aggregate_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    aggregate_metrics: dict[str, float] = {}
    metric_names = [name for name in fold_metrics[0] if name != "fold"]
    for name in metric_names:
        values = np.asarray([row[name] for row in fold_metrics], dtype=float)
        aggregate_metrics[f"{name}_mean"] = float(values.mean())
        aggregate_metrics[f"{name}_std"] = float(values.std())
    return aggregate_metrics


def _write_metrics_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_aggregate_csv(path: Path, aggregate_metrics: dict[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in aggregate_metrics.items():
            writer.writerow([key, value])

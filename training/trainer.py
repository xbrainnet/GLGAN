import copy
import csv
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from config.config import Config
from data.data_loader import BrainConnectivityDataset
from models.model import GLGAN
from utils.seed import set_seed


logger = logging.getLogger("GLGAN.training")


@dataclass(frozen=True)
class CrossValidationResult:
    fold_metrics: list[dict[str, float]]
    aggregate_metrics: dict[str, float]


def run_cross_validation(
    fmri: np.ndarray,
    dti: np.ndarray,
    labels: np.ndarray,
    config: Config,
    *,
    subject_ids: np.ndarray | None = None,
) -> CrossValidationResult:
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1 to select a model on the validation set.")
    if config.n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    if config.repeat_split_mode not in ("resplit", "fixed"):
        raise ValueError("repeat_split_mode must be 'resplit' or 'fixed'.")
    if config.aggregate_unit not in ("run", "fold"):
        raise ValueError("aggregate_unit must be 'run' or 'fold'.")

    output_dir = config.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = labels.astype("int64")
    if subject_ids is not None:
        subject_ids = np.asarray(subject_ids, dtype=str)
        if subject_ids.shape != labels.shape or np.unique(subject_ids).size != labels.size:
            raise ValueError("Cross-validation requires one unique subject ID per label.")
    dataset = BrainConnectivityDataset(fmri, dti, labels)
    # A failed rerun must not leave old aggregate metrics beside updated folds.
    (output_dir / "aggregate_metrics.csv").unlink(missing_ok=True)
    fold_metrics: list[dict[str, float]] = []
    run_means: list[dict[str, float]] = []

    for run_index in range(config.n_repeats):
        run = run_index + 1
        split_seed = config.random_seed
        training_seed = config.random_seed
        if config.repeat_split_mode == "resplit":
            split_seed += run_index
        else:
            training_seed += run_index
        set_seed(training_seed)
        logger.info("Run %02d/%02d split_seed=%d training_seed=%d", run, config.n_repeats, split_seed, training_seed)
        splitter = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=split_seed)
        run_dir = output_dir / f"run_{run:02d}"
        run_dir.mkdir(exist_ok=True)
        fold_metrics_path = run_dir / "fold_metrics.csv"
        cohort_path = run_dir / "cohort.npz"
        fold_metrics_path.unlink(missing_ok=True)
        cohort_path.unlink(missing_ok=True)
        rows = _run_folds(dataset, labels, splitter.split(fmri, labels), run_dir, split_seed, config)
        if subject_ids is not None:
            np.savez(cohort_path, subject_ids=subject_ids, labels=labels)
        _write_metrics_csv(fold_metrics_path, rows)
        fold_metrics.extend({"run": float(run), **row} for row in rows)
        run_means.append({name: float(np.mean(values)) for name, values in _metric_columns(rows).items()})

    aggregate_rows = run_means if config.aggregate_unit == "run" else fold_metrics
    aggregate_metrics = _aggregate_metrics(aggregate_rows)
    _write_aggregate_csv(output_dir / "aggregate_metrics.csv", aggregate_metrics)
    return CrossValidationResult(fold_metrics=fold_metrics, aggregate_metrics=aggregate_metrics)


def _run_folds(
    dataset: BrainConnectivityDataset,
    labels: np.ndarray,
    splits: Iterator[tuple[np.ndarray, np.ndarray]],
    run_dir: Path,
    split_seed: int,
    config: Config,
) -> list[dict[str, float]]:
    if config.save_checkpoints:
        (run_dir / "checkpoints").mkdir(exist_ok=True)
    fold_metrics: list[dict[str, float]] = []

    for fold, (train_index, test_index) in enumerate(splits, start=1):
        try:
            train_idx, val_idx = train_test_split(
                train_index,
                test_size=0.2,
                random_state=split_seed,
                stratify=labels[train_index],
            )
        except ValueError as error:
            raise ValueError(f"{run_dir.name}, fold {fold} could not form a stratified validation split: {error}") from error

        np.savez(
            run_dir / f"fold_{fold}_split.npz",
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_index,
        )

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=config.batch_size, shuffle=False)

        model = GLGAN(config).to(config.torch_device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        best_accuracy = -1.0
        best_state = None
        for _ in range(config.epochs):
            _train_one_epoch(model, train_loader, criterion, optimizer, config)
            val_metrics = evaluate(model, val_loader, criterion, config)
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError(f"Fold {fold} did not produce a validation-selected model.")
        model.load_state_dict(best_state)
        if config.save_checkpoints:
            torch.save(best_state, run_dir / "checkpoints" / f"fold_{fold}_best.pt")

        test_loader = DataLoader(Subset(dataset, test_index), batch_size=config.batch_size, shuffle=False)
        test_metrics = evaluate(model, test_loader, criterion, config)
        fold_metrics.append({"fold": float(fold), **test_metrics})

    return fold_metrics


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
        if config.contrastive_mode == "class_conditional":
            logits, contrastive_loss = model(fmri, dti, labels=labels)
        else:
            logits, contrastive_loss = model(fmri, dti)
        classification_loss = criterion(logits, labels)
        loss = classification_loss + config.lambda_contrast * contrastive_loss
        optimizer.zero_grad()
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


def _metric_columns(rows: list[dict[str, float]]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray([row[name] for row in rows], dtype=float)
        for name in rows[0]
        if name not in ("run", "fold")
    }


def _aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    aggregate_metrics: dict[str, float] = {}
    for name, values in _metric_columns(rows).items():
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

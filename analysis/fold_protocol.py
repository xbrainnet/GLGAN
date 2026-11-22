"""Read the cohort and split files written by the training runner."""

from pathlib import Path

import numpy as np


def load_cohort(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(run_dir / "cohort.npz", allow_pickle=False) as data:
        subject_ids = data["subject_ids"]
        labels = data["labels"]
    if (
        labels.ndim != 1
        or subject_ids.shape != labels.shape
        or np.unique(subject_ids).size != labels.size
        or not np.array_equal(np.unique(labels), [0, 1])
    ):
        raise ValueError(f"{run_dir}: invalid cohort IDs or binary labels.")
    return subject_ids, labels


def load_folds(run_dir: Path, n_subjects: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read one complete ten-fold run, including each inner validation split."""
    paths = [run_dir / f"fold_{fold}_split.npz" for fold in range(1, 11)]
    if set(run_dir.glob("fold_*_split.npz")) != set(paths):
        raise ValueError(f"{run_dir}: expected exactly fold_1_split.npz through fold_10_split.npz.")
    indices = np.arange(n_subjects)
    folds = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            split = (data["train_idx"], data["val_idx"], data["test_idx"])
        if any(part.ndim != 1 or part.dtype.kind not in "iu" or part.size == 0 for part in split):
            raise ValueError(f"{path}: expected nonempty integer train, validation and test indices.")
        if not np.array_equal(np.sort(np.concatenate(split)), indices):
            raise ValueError(f"{path}: train, validation and test indices must partition the cohort.")
        folds.append(split)
    if not np.array_equal(np.sort(np.concatenate([test for _, _, test in folds])), indices):
        raise ValueError(f"{run_dir}: each subject must be tested in exactly one outer fold.")
    return folds

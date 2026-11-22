"""Compare Table S5 accuracies from a matched pair of ten-fold runs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

from analysis.fold_protocol import load_cohort, load_folds


def read_accuracy(run_dir: Path) -> np.ndarray:
    values: dict[int, float] = {}
    with (run_dir / "fold_metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fold = float(row["fold"])
            accuracy = float(row["accuracy"])
            if fold not in range(1, 11) or fold in values or not 0 <= accuracy <= 1:
                raise ValueError(f"{run_dir}: invalid or duplicate fold number or accuracy.")
            values[int(fold)] = accuracy
    if set(values) != set(range(1, 11)):
        raise ValueError(f"{run_dir}: expected accuracy for all ten outer folds.")
    return np.asarray([values[fold] for fold in range(1, 11)])


def compare_runs(baseline_dir: Path, variant_dir: Path) -> dict[str, float | int | str | None]:
    if baseline_dir.resolve() == variant_dir.resolve():
        raise ValueError("Supply separate baseline and variant run directories.")
    baseline_ids, baseline_labels = load_cohort(baseline_dir)
    variant_ids, variant_labels = load_cohort(variant_dir)
    if not np.array_equal(baseline_ids, variant_ids) or not np.array_equal(baseline_labels, variant_labels):
        raise ValueError("Both runs must have identical ordered subject IDs and labels.")
    baseline_folds = load_folds(baseline_dir, len(baseline_ids))
    variant_folds = load_folds(variant_dir, len(variant_ids))
    for fold, (baseline, variant) in enumerate(zip(baseline_folds, variant_folds), start=1):
        if any(not np.array_equal(np.sort(a), np.sort(b)) for a, b in zip(baseline, variant)):
            raise ValueError(f"Fold {fold} has different train, validation or test membership.")

    baseline_acc = read_accuracy(baseline_dir)
    variant_acc = read_accuracy(variant_dir)
    delta = variant_acc - baseline_acc
    # A t statistic requires nonzero variance in the paired differences.
    p_value = float(ttest_rel(variant_acc, baseline_acc).pvalue) if np.ptp(delta) > 1e-12 else None
    return {
        "test": "two-sided paired t-test",
        "n_folds": len(delta),
        "baseline_accuracy_mean": float(baseline_acc.mean()),
        "baseline_accuracy_std": float(baseline_acc.std(ddof=0)),
        "variant_accuracy_mean": float(variant_acc.mean()),
        "variant_accuracy_std": float(variant_acc.std(ddof=0)),
        "mean_accuracy_difference_variant_minus_baseline": float(delta.mean()),
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_runs(args.baseline, args.variant), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

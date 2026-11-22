"""Estimate Table S3 ROI stability from connectivity edges in each outer training fold."""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from analysis._elastic_net import fit_elastic_net
from analysis.fold_protocol import load_cohort, load_folds
from config.config import DEFAULT_CONFIG
from data.data_loader import BrainConnectivityData, load_brain_connectivity_data


def summarize_roi_stability(
    data: BrainConnectivityData, splits: list[np.ndarray], roi_labels: list[str], *, inner_folds: int, seed: int
) -> list[dict[str, str | int | float]]:
    n_regions = len(roi_labels)
    if len(splits) != 10 or data.fmri.shape != data.dti.shape or data.fmri.shape[1:] != (n_regions, n_regions):
        raise ValueError("Expected ten outer folds and paired connectivity matrices matching the ROI labels.")
    rows, cols = np.triu_indices(n_regions, k=1)
    features = np.column_stack((data.fmri[:, rows, cols], data.dti[:, rows, cols]))
    ranks: list[list[int]] = [[] for _ in roi_labels]
    prominence: list[list[float]] = [[] for _ in roi_labels]

    for train in splits:
        fit = fit_elastic_net(features[train], data.labels[train], positive=False, folds=inner_folds, seed=seed)
        weights = np.abs(fit.best_estimator_.named_steps["elasticnet"].coef_).reshape(2, -1).sum(axis=0)
        scores = np.bincount(rows, weights, minlength=n_regions) + np.bincount(cols, weights, minlength=n_regions)
        if scores.max() == 0:
            raise ValueError("An outer fold yielded only zero coefficients. No ROI ranking can be inferred.")
        scores /= scores.max()
        for rank, roi in enumerate(np.argsort(-scores, kind="stable")[:10], start=1):
            ranks[roi].append(rank)
            prominence[roi].append(float(scores[roi]))

    selected = [roi for roi in range(n_regions) if ranks[roi]]
    selected.sort(key=lambda roi: (-len(ranks[roi]), float(np.mean(ranks[roi])), roi))
    return [
        {
            "roi": roi_labels[roi],
            "frequency": len(ranks[roi]),
            "outer_folds": len(splits),
            "mean_rank": float(np.mean(ranks[roi])),
            "rank_std": float(np.std(ranks[roi], ddof=1)) if len(ranks[roi]) > 1 else 0.0,
            "mean_prominence": float(np.mean(prominence[roi])),
            "prominence_std": float(np.std(prominence[roi], ddof=1)) if len(ranks[roi]) > 1 else 0.0,
            "stable": "Yes" if len(ranks[roi]) >= 7 else "No",
        }
        for roi in selected[:10]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_CONFIG.data_dir)
    parser.add_argument("--fmri-file", default=DEFAULT_CONFIG.fmri_file)
    parser.add_argument("--dti-file", default=DEFAULT_CONFIG.dti_file)
    parser.add_argument("--fmri-key")
    parser.add_argument("--dti-key")
    parser.add_argument("--label-key")
    parser.add_argument("--fmri-id-key", default=DEFAULT_CONFIG.fmri_id_key)
    parser.add_argument("--dti-id-key", default=DEFAULT_CONFIG.dti_id_key)
    parser.add_argument("--label-id-key", default=DEFAULT_CONFIG.label_id_key)
    parser.add_argument("--fmri-time-series", action="store_true")
    parser.add_argument("--negative-classes", type=int, nargs="+", required=True)
    parser.add_argument("--positive-classes", type=int, nargs="+", required=True)
    parser.add_argument("--roi-labels", type=Path, required=True, help="90 AAL ROI abbreviations, one per line in matrix order.")
    parser.add_argument("--run-dir", type=Path, help="Reuse the saved cohort and splits of one ten-fold run.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/analysis/elastic_net_roi_stability.csv"))
    args = parser.parse_args()

    roi_labels = args.roi_labels.read_text(encoding="utf-8-sig").splitlines()
    roi_labels = [label.strip() for label in roi_labels]
    if len(roi_labels) != 90 or len(set(roi_labels)) != 90 or not all(roi_labels):
        parser.error("--roi-labels must specify 90 unique, nonempty ROI abbreviations.")
    if args.inner_folds < 2:
        parser.error("--inner-folds must be at least 2.")
    config = DEFAULT_CONFIG.with_updates(
        data_dir=args.data_dir, fmri_file=args.fmri_file, dti_file=args.dti_file,
        fmri_key=args.fmri_key, dti_key=args.dti_key, label_key=args.label_key,
        fmri_id_key=args.fmri_id_key, dti_id_key=args.dti_id_key, label_id_key=args.label_id_key,
        fmri_time_series=args.fmri_time_series,
        negative_classes=args.negative_classes, positive_classes=args.positive_classes,
    )
    data = load_brain_connectivity_data(config)
    if args.run_dir:
        subject_ids, labels = load_cohort(args.run_dir)
        if not np.array_equal(subject_ids, data.subject_ids) or not np.array_equal(labels, data.labels):
            raise ValueError("The saved cohort must match the aligned connectivity data and task labels.")
        folds = load_folds(args.run_dir, len(labels))
        splits = [np.sort(np.concatenate((train, validation))) for train, validation, _ in folds]
    else:
        if np.bincount(data.labels, minlength=2).min() < 10:
            raise ValueError("Each class needs at least ten subjects for ten-fold cross-validation.")
        outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
        splits = [train for train, _ in outer.split(data.fmri, data.labels)]

    records = summarize_roi_stability(data, splits, roi_labels, inner_folds=args.inner_folds, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()

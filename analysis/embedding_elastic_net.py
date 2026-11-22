"""Fit the separate non-negative Elastic Net analysis of pre-softmax GLGAN features."""

import argparse
import csv
from pathlib import Path

import numpy as np

from analysis._elastic_net import fit_elastic_net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="NumPy matrix of GLGAN embeddings [subjects, features].")
    parser.add_argument("--labels", type=Path, required=True, help="NumPy binary labels in the same subject order.")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/analysis/embedding_elastic_net.csv"))
    args = parser.parse_args()

    features = np.load(args.features, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    if features.ndim != 2 or features.shape[1] == 0 or labels.shape != (features.shape[0],):
        parser.error("Expected one row of embedding features and one binary label per subject.")
    if features.dtype.kind not in "iuf" or not np.isfinite(features).all():
        parser.error("Embedding features must be real and finite.")
    if args.inner_folds < 2:
        parser.error("--inner-folds must be at least 2.")
    fit = fit_elastic_net(features, labels, positive=True, folds=args.inner_folds, seed=args.seed)
    model = fit.best_estimator_.named_steps["elasticnet"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("embedding_feature_index", "coefficient_on_standardized_feature", "alpha", "l1_ratio"))
        for column, coefficient in enumerate(model.coef_):
            writer.writerow((column, coefficient, model.alpha, model.l1_ratio))


if __name__ == "__main__":
    main()

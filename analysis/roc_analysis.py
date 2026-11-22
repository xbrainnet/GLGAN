import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import roc_curve


def compute_roc_table(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    fpr, tpr, thresholds = roc_curve(labels.astype(int), scores.astype(float))
    table = np.zeros(len(fpr), dtype=[("fpr", "f8"), ("tpr", "f8"), ("threshold", "f8")])
    table["fpr"] = fpr
    table["tpr"] = tpr
    table["threshold"] = thresholds
    return table, float(sklearn_auc(fpr, tpr))


def write_roc_csv(path: str | Path, table: np.ndarray, auc_value: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["auc", auc_value])
        writer.writerow(["fpr", "tpr", "threshold"])
        for row in table:
            writer.writerow([row["fpr"], row["tpr"], row["threshold"]])


def _load_prediction_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    scores: list[float] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
    return np.asarray(labels), np.asarray(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ROC points from a CSV with label and score columns.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/analysis/roc_points.csv")
    args = parser.parse_args()

    labels, scores = _load_prediction_csv(args.input)
    table, auc_value = compute_roc_table(labels, scores)
    write_roc_csv(args.output, table, auc_value)


if __name__ == "__main__":
    main()

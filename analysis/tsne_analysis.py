import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE


def compute_tsne_embedding(features: np.ndarray, perplexity: float = 30.0, random_state: int = 7) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a 2D array shaped [n_samples, n_features].")
    safe_perplexity = min(float(perplexity), max(1.0, features.shape[0] - 1.0))
    return TSNE(n_components=2, perplexity=safe_perplexity, random_state=random_state, init="random").fit_transform(features)


def write_embedding_csv(path: str | Path, embedding: np.ndarray, labels: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "label"])
        if labels is None:
            labels = np.full(embedding.shape[0], "")
        for point, label in zip(embedding, labels):
            writer.writerow([point[0], point[1], label])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute t-SNE coordinates from a NumPy feature matrix.")
    parser.add_argument("--features", required=True, help=".npy file shaped [n_samples, n_features]")
    parser.add_argument("--labels", default=None, help="Optional .npy label vector")
    parser.add_argument("--output", default="results/analysis/tsne_coordinates.csv")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    features = np.load(args.features)
    labels = np.load(args.labels) if args.labels else None
    embedding = compute_tsne_embedding(features, perplexity=args.perplexity, random_state=args.seed)
    write_embedding_csv(args.output, embedding, labels)


if __name__ == "__main__":
    main()

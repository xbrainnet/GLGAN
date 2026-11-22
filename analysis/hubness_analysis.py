import argparse
import csv
from pathlib import Path

import numpy as np

from models.model import pagerank_centrality
import torch


def compute_hubness(adjacency: np.ndarray, pagerank_alpha: float = 0.85) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype="float32")
    if adjacency.ndim == 3:
        adjacency = adjacency.mean(axis=0)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be [n_regions, n_regions] or [n_subjects, n_regions, n_regions].")
    degree = adjacency.sum(axis=1)
    ranks = pagerank_centrality(torch.as_tensor(adjacency[None, :, :]), alpha=pagerank_alpha).squeeze(0).numpy()
    table = np.zeros(adjacency.shape[0], dtype=[("roi", "i4"), ("degree", "f8"), ("pagerank", "f8")])
    table["roi"] = np.arange(1, adjacency.shape[0] + 1)
    table["degree"] = degree
    table["pagerank"] = ranks
    return table


def write_hubness_csv(path: str | Path, table: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["roi", "degree", "pagerank"])
        for row in table:
            writer.writerow([row["roi"], row["degree"], row["pagerank"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute mean-degree and PageRank hubness from a .npy adjacency array.")
    parser.add_argument("--adjacency", required=True)
    parser.add_argument("--output", default="results/analysis/hubness.csv")
    args = parser.parse_args()

    table = compute_hubness(np.load(args.adjacency))
    write_hubness_csv(args.output, table)


if __name__ == "__main__":
    main()

import argparse
import csv
from pathlib import Path

from config.config import DEFAULT_CONFIG
from models.model import GLGAN


def count_glgan_parameters(n_regions: int = 90, hidden_dim: int = 90) -> int:
    config = DEFAULT_CONFIG.with_updates(n_regions=n_regions, hidden_dim=hidden_dim, device="cpu")
    model = GLGAN(config)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def write_parameter_count(path: str | Path, count: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "trainable_parameters"])
        writer.writerow(["GLGAN", count])


def main() -> None:
    parser = argparse.ArgumentParser(description="Count trainable GLGAN parameters.")
    parser.add_argument("--n-regions", type=int, default=90)
    parser.add_argument("--hidden-dim", type=int, default=90)
    parser.add_argument("--output", default="results/analysis/parameter_count.csv")
    args = parser.parse_args()

    count = count_glgan_parameters(n_regions=args.n_regions, hidden_dim=args.hidden_dim)
    write_parameter_count(args.output, count)


if __name__ == "__main__":
    main()

import argparse
import csv
from pathlib import Path


def compute_sensitivity_statistics(
    input_csv: str | Path, output_csv: str | Path, parameter_column: str, metric_column: str
) -> None:
    groups: dict[str, list[float]] = {}
    with Path(input_csv).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(row[parameter_column], []).append(float(row[metric_column]))

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([parameter_column, f"{metric_column}_mean", f"{metric_column}_std"])
        for value, metrics in groups.items():
            mean_value = sum(metrics) / len(metrics)
            variance = sum((metric - mean_value) ** 2 for metric in metrics) / len(metrics)
            writer.writerow([value, mean_value, variance ** 0.5])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute grouped statistics for sensitivity-analysis results.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/analysis/sensitivity_statistics.csv")
    parser.add_argument("--parameter-column", required=True)
    parser.add_argument("--metric-column", default="accuracy")
    args = parser.parse_args()

    compute_sensitivity_statistics(args.input, args.output, args.parameter_column, args.metric_column)


if __name__ == "__main__":
    main()

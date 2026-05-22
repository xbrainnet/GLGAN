import argparse
import csv
from pathlib import Path


def rank_rois(input_csv: str | Path, output_csv: str | Path, top_k: int = 10, score_column: str = "pagerank") -> None:
    with Path(input_csv).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row[score_column]), reverse=True)
    selected = rows[:top_k]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected[0].keys() if selected else ["roi", score_column])
        writer.writeheader()
        writer.writerows(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank ROI records by a selected connectivity or hubness score.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/analysis/top_rois.csv")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--score-column", default="pagerank")
    args = parser.parse_args()

    rank_rois(args.input, args.output, top_k=args.top_k, score_column=args.score_column)


if __name__ == "__main__":
    main()
